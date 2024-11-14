import numpy as np
import torch
import os
import json
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

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

def select_diverse_images(similarity_matrix, n_select=100, method='ward'):
    """
    Select diverse images using hierarchical clustering
    """
    # Convert similarities to distances
    distances = 1 - similarity_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distances), method=method)
    
    # Cut tree to get n_select clusters
    clusters = fcluster(linkage_matrix, n_select, criterion='maxclust')
    
    # Select one image from each cluster (the one closest to cluster center)
    selected_indices = []
    for i in range(1, n_select + 1):
        cluster_indices = np.where(clusters == i)[0]
        
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
    
    # Compute linkage matrix
    distances = 1 - similarity_matrix
    linkage_matrix = linkage(squareform(distances), method='ward')
    
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

def main():
    # Load features
    print("Loading features...")
    cnn_features, semantic_features, metadata = load_features()
    
    # Compute similarity matrices
    print("Computing similarities...")
    cnn_similarities, semantic_similarities = compute_similarity_matrices(
        cnn_features, semantic_features
    )
    
    # Select diverse images using CNN features
    print("Selecting diverse images...")
    n_select = 100  # Number of images to select
    selected_indices, clusters = select_diverse_images(
        cnn_similarities, n_select=n_select
    )
    
    # Visualize results
    print("Creating visualizations...")
    visualize_clustering(cnn_similarities, metadata, selected_indices)
    
    # Print selected images info
    print("\nSelected Images:")
    for idx in selected_indices:
        print(f"Class: {metadata[idx]['class_name']}, "
              f"Index: {metadata[idx]['index']}")
    
    # Save selection results
    results = {
        'selected_indices': selected_indices,
        'selected_images': [metadata[idx] for idx in selected_indices]
    }
    
    with open('selection_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSelected {len(selected_indices)} diverse images")
    print("Results saved to selection_results.json")

if __name__ == "__main__":
    main() 