import numpy as np
import torch
import os
import json
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from load_cifar100 import load_cifar100

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

def visualize_clustering(similarity_matrix, metadata, selected_indices=None, save_dir='output_graphs'):
    """
    Create dendrogram and heatmap visualizations and save them
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert similarities to distances and ensure diagonal is exactly 0
    distances = 1 - similarity_matrix
    np.fill_diagonal(distances, 0)
    
    # Convert to condensed form for linkage
    condensed_distances = squareform(distances)
    
    # Compute linkage matrix
    linkage_matrix = linkage(condensed_distances, method='ward')
    
    # Create labels with both class name and index for better tracking
    labels = [f"{m['class_name']}_{m['index']}" for m in metadata]
    
    # Plot dendrogram
    dendrogram_dict = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax1
    )
    ax1.set_title("Hierarchical Clustering Dendrogram")
    
    # Get the order of leaves from dendrogram
    leaf_order = dendrogram_dict['leaves']
    
    # Reorder similarity matrix to match dendrogram
    ordered_sim = similarity_matrix[leaf_order][:, leaf_order]
    ordered_labels = [labels[i] for i in leaf_order]
    
    # Plot heatmap with ordered indices
    mask = np.eye(len(ordered_sim), dtype=bool)
    sns.heatmap(
        ordered_sim,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=False,
        xticklabels=ordered_labels,
        yticklabels=ordered_labels,
        ax=ax2
    )
    ax2.set_title("Similarity Matrix Heatmap")
    
    # Rotate x-axis labels for better readability
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    if selected_indices is not None:
        # Convert selected indices to positions in ordered matrix
        ordered_positions = [leaf_order.index(idx) for idx in selected_indices]
        # Highlight selected images in heatmap
        for pos in ordered_positions:
            ax2.axhline(y=pos, color='g', alpha=0.3)
            ax2.axvline(x=pos, color='g', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(os.path.join(save_dir, 'full_dataset_visualization.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    plt.show()
    return fig

def visualize_comparison(cnn_similarity, sbert_similarity, metadata, selected_indices, save_dir='output_graphs'):
    """
    Create side-by-side visualizations comparing CNN and SBERT similarities
    and save them
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract the submatrices for selected indices
    reduced_cnn = cnn_similarity[np.ix_(selected_indices, selected_indices)]
    reduced_sbert = sbert_similarity[np.ix_(selected_indices, selected_indices)]
    reduced_metadata = [metadata[i] for i in selected_indices]
    
    # Create figure with four subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Create consistent labels for all plots
    labels = [f"{m['class_name']}_{m['index']}" for m in reduced_metadata]
    
    # CNN Dendrogram
    # Ensure distances are non-negative and diagonal is exactly zero
    distances_cnn = 1 - reduced_cnn
    min_dist_cnn = np.min(distances_cnn[~np.eye(len(distances_cnn), dtype=bool)])  # Exclude diagonal
    if min_dist_cnn < 0:
        distances_cnn[~np.eye(len(distances_cnn), dtype=bool)] -= min_dist_cnn  # Adjust non-diagonal elements
    np.fill_diagonal(distances_cnn, 0.0)  # Ensure diagonal is exactly zero
    
    condensed_distances_cnn = squareform(distances_cnn)
    linkage_matrix_cnn = linkage(condensed_distances_cnn, method='ward')
    
    # Plot CNN dendrogram and get leaf order
    dendrogram_dict_cnn = dendrogram(
        linkage_matrix_cnn,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax1
    )
    ax1.set_title("CNN Features Dendrogram")
    
    # Get CNN leaf order
    cnn_leaf_order = dendrogram_dict_cnn['leaves']
    ordered_labels_cnn = [labels[i] for i in cnn_leaf_order]
    
    # SBERT Dendrogram
    # Ensure distances are non-negative and diagonal is exactly zero
    distances_sbert = 1 - reduced_sbert
    min_dist_sbert = np.min(distances_sbert[~np.eye(len(distances_sbert), dtype=bool)])  # Exclude diagonal
    if min_dist_sbert < 0:
        distances_sbert[~np.eye(len(distances_sbert), dtype=bool)] -= min_dist_sbert  # Adjust non-diagonal elements
    np.fill_diagonal(distances_sbert, 0.0)  # Ensure diagonal is exactly zero
    
    condensed_distances_sbert = squareform(distances_sbert)
    linkage_matrix_sbert = linkage(condensed_distances_sbert, method='ward')
    
    # Plot SBERT dendrogram and get leaf order
    dendrogram_dict_sbert = dendrogram(
        linkage_matrix_sbert,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax2
    )
    ax2.set_title("SBERT Features Dendrogram")
    
    # Get SBERT leaf order
    sbert_leaf_order = dendrogram_dict_sbert['leaves']
    ordered_labels_sbert = [labels[i] for i in sbert_leaf_order]
    
    # Reorder similarity matrices according to respective dendrograms
    ordered_cnn = reduced_cnn[cnn_leaf_order][:, cnn_leaf_order]
    ordered_sbert = reduced_sbert[sbert_leaf_order][:, sbert_leaf_order]
    
    # Plot heatmaps with ordered indices
    mask = np.eye(len(reduced_cnn), dtype=bool)
    
    # CNN Heatmap
    sns.heatmap(
        ordered_cnn,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        xticklabels=ordered_labels_cnn,
        yticklabels=ordered_labels_cnn,
        ax=ax3
    )
    ax3.set_title("CNN Similarity Matrix")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
    
    # SBERT Heatmap
    sns.heatmap(
        ordered_sbert,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        xticklabels=ordered_labels_sbert,
        yticklabels=ordered_labels_sbert,
        ax=ax4
    )
    ax4.set_title("SBERT Similarity Matrix")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    fig.savefig(os.path.join(save_dir, 'comparison_visualization.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    plt.show()
    return fig

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

def save_selected_images(dataset, selected_indices, metadata, output_dir='selected_images'):
    """
    Save and display the selected images
    
    Args:
        dataset: CIFAR-100 dataset
        selected_indices: indices of selected images
        metadata: metadata for the images
        output_dir: directory to save images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure to display all selected images
    n_images = len(selected_indices)
    n_cols = 5
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Inverse normalization transform
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Save and display each selected image
    for idx, (ax, selected_idx) in enumerate(zip(axes.flat, selected_indices)):
        # Get image and metadata
        meta = metadata[selected_idx]
        actual_idx = meta['index']  # Use the index from metadata
        image, label = dataset[actual_idx]
        
        # Verify label matches metadata
        if label != meta['class_id']:
            print(f"Warning: Label mismatch for index {actual_idx}")
            print(f"Dataset label: {label}, Metadata class_id: {meta['class_id']}")
        
        # Convert tensor to PIL Image
        image = inv_normalize(image)
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        
        # Save image
        filename = f"{idx+1:02d}_{meta['class_name']}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)
        
        # Display image
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f"{meta['class_name']}\n(#{idx+1})", fontsize=8)
    
    # Remove empty subplots
    for idx in range(len(selected_indices), len(axes.flat)):
        axes.flat[idx].remove()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSelected images saved to {output_dir}/")
    return fig

def main():
    # Load CIFAR-100 dataset first
    print("Loading CIFAR-100 dataset...")
    dataset, class_names, class_to_idx, idx_to_class = load_cifar100()
    
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
    
    # Create output directory
    output_dir = 'output_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all visualizations
    print("Creating and saving visualizations...")
    visualize_clustering(cnn_similarities, metadata, selected_indices, output_dir)
    visualize_comparison(cnn_similarities, semantic_similarities, metadata, selected_indices, output_dir)
    
    # Save selected images grid
    fig = save_selected_images(dataset, selected_indices, metadata)
    fig.savefig(os.path.join(output_dir, 'selected_images_grid.png'), 
                bbox_inches='tight', 
                dpi=300)
    
    print(f"\nAll visualizations saved to {output_dir}/")
    
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