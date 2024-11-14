import numpy as np
import torch
import os
import json
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_features(feature_dir='cifar_features'):
    """
    Load CNN features, semantic features, and metadata
    """
    # Load metadata
    with open(os.path.join(feature_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load semantic features
    semantic_features = np.load(os.path.join(feature_dir, 'semantic_features.npy'))
    
    # Load CNN features
    cnn_features = []
    for item in metadata:
        feature_path = os.path.join(feature_dir, item['feature_file'])
        feature = torch.load(feature_path)
        cnn_features.append(feature.flatten().numpy())
    
    cnn_features = np.array(cnn_features)
    
    return cnn_features, semantic_features, metadata

def compute_similarity_matrices(cnn_features, semantic_features):
    """
    Compute similarity matrices for both CNN and semantic features
    """
    # CNN similarities
    cnn_similarities = cosine_similarity(cnn_features)
    
    # Semantic similarities
    semantic_similarities = cosine_similarity(semantic_features)
    
    return cnn_similarities, semantic_similarities

def select_diverse_images(similarity_matrix, n_select=20, method='ward'):
    """
    Select diverse images using hierarchical clustering
    """
    # Convert similarities to distances and ensure diagonal is exactly 0
    distances = 1 - similarity_matrix
    np.fill_diagonal(distances, 0)  # Force diagonal to be exactly 0
    
    # Convert to condensed form (required by linkage)
    condensed_distances = squareform(distances)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=method)
    
    # Cut tree to get n_select clusters
    clusters = fcluster(linkage_matrix, n_select, criterion='maxclust')
    
    # Select one image from each cluster (the one closest to cluster center)
    selected_indices = []
    for i in range(1, n_select + 1):
        cluster_indices = np.where(clusters == i)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Find the image closest to cluster center
        cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
        mean_similarities = np.mean(cluster_similarities, axis=1)
        center_idx = cluster_indices[np.argmax(mean_similarities)]
        
        selected_indices.append(center_idx)
    
    return selected_indices, clusters

def visualize_clustering(similarity_matrix, metadata, selected_indices=None):
    """
    Create dendrogram and heatmap visualizations
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert similarities to distances and ensure diagonal is exactly 0
    distances = 1 - similarity_matrix
    np.fill_diagonal(distances, 0)  # Force diagonal to be exactly 0
    
    # Convert to condensed form for linkage
    condensed_distances = squareform(distances)
    
    # Compute linkage matrix
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Plot dendrogram
    labels = [f"{m['class_name']}_{m['index']}" for m in metadata]
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax1
    )
    ax1.set_title("Hierarchical Clustering Dendrogram")
    
    # Plot heatmap
    mask = np.eye(len(similarity_matrix), dtype=bool)
    sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=False,
        xticklabels=False,
        yticklabels=False,
        ax=ax2
    )
    ax2.set_title("Similarity Matrix Heatmap")
    
    if selected_indices is not None:
        # Highlight selected images in heatmap
        for idx in selected_indices:
            ax2.axhline(y=idx, color='g', alpha=0.3)
            ax2.axvline(x=idx, color='g', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_comparison(cnn_similarity, sbert_similarity, metadata, selected_indices):
    """
    Create side-by-side visualizations comparing CNN and SBERT similarities
    for the selected subset of images
    """
    # Extract the submatrices for selected indices
    reduced_cnn = cnn_similarity[np.ix_(selected_indices, selected_indices)]
    reduced_sbert = sbert_similarity[np.ix_(selected_indices, selected_indices)]
    reduced_metadata = [metadata[i] for i in selected_indices]
    
    # Create figure with four subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Labels for both visualizations
    labels = [f"{m['class_name']}_{m['index']}" for m in reduced_metadata]
    
    # CNN Dendrogram
    distances_cnn = 1 - reduced_cnn
    np.fill_diagonal(distances_cnn, 0)
    condensed_distances_cnn = squareform(distances_cnn)
    linkage_matrix_cnn = linkage(condensed_distances_cnn, method='ward')
    
    dendrogram(
        linkage_matrix_cnn,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax1
    )
    ax1.set_title("CNN Features Dendrogram")
    
    # SBERT Dendrogram
    distances_sbert = 1 - reduced_sbert
    np.fill_diagonal(distances_sbert, 0)
    condensed_distances_sbert = squareform(distances_sbert)
    linkage_matrix_sbert = linkage(condensed_distances_sbert, method='ward')
    
    dendrogram(
        linkage_matrix_sbert,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax2
    )
    ax2.set_title("SBERT Features Dendrogram")
    
    # CNN Heatmap
    mask = np.eye(len(reduced_cnn), dtype=bool)
    sns.heatmap(
        reduced_cnn,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax3
    )
    ax3.set_title("CNN Similarity Matrix")
    
    # SBERT Heatmap
    sns.heatmap(
        reduced_sbert,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax4
    )
    ax4.set_title("SBERT Similarity Matrix")
    
    plt.tight_layout()
    plt.show()

def analyze_pairs(similarity_matrix, metadata, selected_indices, top_n=10):
    """
    Analyze the most and least similar pairs among selected images
    """
    # Get the reduced similarity matrix for selected images
    reduced_similarity = similarity_matrix[np.ix_(selected_indices, selected_indices)]
    reduced_metadata = [metadata[i] for i in selected_indices]
    
    # Create a mask for the upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(reduced_similarity), k=1).astype(bool)
    
    # Get pairs and their similarities
    pairs = []
    for i in range(len(reduced_similarity)):
        for j in range(i + 1, len(reduced_similarity)):
            if mask[i, j]:
                pairs.append((i, j, reduced_similarity[i, j]))
    
    # Sort pairs by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Print most similar pairs
    print("\nMost Similar Pairs:")
    print("-" * 80)
    for i, j, sim in pairs[:top_n]:
        img1 = reduced_metadata[i]
        img2 = reduced_metadata[j]
        print(f"Similarity: {sim:.3f}")
        print(f"Image 1: Class '{img1['class_name']}' (Index: {img1['index']})")
        print(f"Image 2: Class '{img2['class_name']}' (Index: {img2['index']})")
        print("-" * 80)
    
    # Print least similar pairs
    print("\nLeast Similar Pairs:")
    print("-" * 80)
    for i, j, sim in pairs[-top_n:]:
        img1 = reduced_metadata[i]
        img2 = reduced_metadata[j]
        print(f"Similarity: {sim:.3f}")
        print(f"Image 1: Class '{img1['class_name']}' (Index: {img1['index']})")
        print(f"Image 2: Class '{img2['class_name']}' (Index: {img2['index']})")
        print("-" * 80)
    
    return pairs

def main():
    # Load features
    print("Loading features...")
    cnn_features, semantic_features, metadata = load_features()
    
    # Compute similarity matrices
    print("Computing similarities...")
    cnn_similarities, semantic_similarities = compute_similarity_matrices(
        cnn_features, semantic_features
    )
    
    # Select diverse images using ONLY CNN features
    print("Selecting diverse images using CNN features...")
    n_select = 20  # Using 20 images as specified
    selected_indices, clusters = select_diverse_images(
        cnn_similarities, n_select=n_select
    )
    
    # Visualize full CNN-based clustering
    print("Creating full dataset visualization (CNN-based)...")
    visualize_clustering(cnn_similarities, metadata, selected_indices)
    
    # Compare CNN and SBERT for selected images
    print("Creating comparison visualizations for selected images...")
    visualize_comparison(cnn_similarities, semantic_similarities, metadata, selected_indices)
    
    # Analyze and print most/least similar pairs (CNN-based)
    print("\nAnalyzing CNN-based similarity pairs...")
    pairs = analyze_pairs(cnn_similarities, metadata, selected_indices, top_n=10)
    
    # Print selected images info
    print("\nSelected Images:")
    for idx in selected_indices:
        print(f"Class: {metadata[idx]['class_name']}, Index: {metadata[idx]['index']}")
    
    # Save results
    results = {
        'selected_indices': selected_indices,
        'selected_images': [metadata[idx] for idx in selected_indices],
        'cnn_most_similar_pairs': [(metadata[selected_indices[i]]['index'], 
                                   metadata[selected_indices[j]]['index'], 
                                   float(sim)) for i, j, sim in pairs[:10]],
        'cnn_least_similar_pairs': [(metadata[selected_indices[i]]['index'], 
                                    metadata[selected_indices[j]]['index'], 
                                    float(sim)) for i, j, sim in pairs[-10:]]
    }
    
    with open('selection_results.json', 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nSelected {len(selected_indices)} diverse images")
    print("Results saved to selection_results.json")

if __name__ == "__main__":
    main() 