import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform, pdist
import os
import random
import shutil
import warnings
import json
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# Set random seed
random.seed(42)
np.random.seed(42)

### Part 1: Load Features (same as original) ###
feature_directory = os.path.join(os.getcwd(), 'src', 'unique_objects_features')
source_directory = os.path.join(os.getcwd(), 'ObjectsAll', 'OBJECTSALL')
output_base_dir = os.path.join(os.getcwd(), 'src', 'selected_clusters_global')  # New output directory

# Load mapping and features (same as original)
mapping_file = os.path.join(feature_directory, 'feature_mapping.json')
with open(mapping_file, 'r') as f:
    feature_mapping = json.load(f)

feature_files = sorted([f['feature_file'] for f in feature_mapping.values()])
cnn_feature_vectors = []
cnn_image_labels = []

for feature_file in feature_files:
    feature_path = os.path.join(feature_directory, feature_file)
    feature_vector = torch.load(feature_path).flatten().numpy()
    cnn_feature_vectors.append(feature_vector)
    cnn_image_labels.append(os.path.splitext(feature_file)[0])

cnn_feature_vectors = np.array(cnn_feature_vectors)

### Part 2: Select 100 Most Diverse Images (same as original) ###
def select_most_diverse(features, num_select=100):
    # (Same implementation as original)
    selected_indices = []
    remaining_indices = set(range(len(features)))
    first = random.choice(list(remaining_indices))
    selected_indices.append(first)
    remaining_indices.remove(first)
    
    for _ in range(1, num_select):
        selected_features = features[selected_indices]
        remaining_features = features[list(remaining_indices)]
        similarities = cosine_similarity(remaining_features, selected_features)
        min_similarities = similarities.min(axis=1)
        most_diverse_idx = np.argmin(min_similarities)
        most_diverse_image = list(remaining_indices)[most_diverse_idx]
        selected_indices.append(most_diverse_image)
        remaining_indices.remove(most_diverse_image)
        if len(selected_indices) % 10 == 0:
            print(f"Selected {len(selected_indices)} diverse images so far...")
    
    return selected_indices

print("Selecting 100 most diverse images...")
selected_indices = select_most_diverse(cnn_feature_vectors, num_select=100)
selected_labels = [cnn_image_labels[i] for i in selected_indices]
selected_features = cnn_feature_vectors[selected_indices]

### Part 3: Global Clustering Approach ###
def global_clustering(features, num_clusters=20, images_per_cluster=5):
    # First use hierarchical clustering to get rough groups
    initial_clusters = 40  # Start with even more clusters for finer granularity
    hierarchical = AgglomerativeClustering(n_clusters=initial_clusters)
    cluster_labels = hierarchical.fit_predict(features)
    
    # Create initial clusters dictionary
    initial_groups = {i: [] for i in range(initial_clusters)}
    
    # Assign indices to initial groups
    for idx, label in enumerate(cluster_labels):
        initial_groups[label].append(idx)
    
    # Create final clusters ensuring maximum diversity
    final_clusters = {}
    used_indices = set()
    
    # For each new cluster we want to form
    for cluster_id in range(num_clusters):
        cluster_indices = []
        available_indices = set(range(len(features))) - used_indices
        
        # Select first image for this cluster
        # Choose from images that are most different from all previously used images
        if used_indices:
            used_features = features[list(used_indices)]
            available_features = features[list(available_indices)]
            similarities = cosine_similarity(available_features, used_features)
            max_similarities = similarities.max(axis=1)  # Maximum similarity to any used image
            start_idx = list(available_indices)[np.argmin(max_similarities)]
        else:
            # For first cluster, start randomly
            start_idx = random.choice(list(available_indices))
            
        cluster_indices.append(start_idx)
        used_indices.add(start_idx)
        available_indices.remove(start_idx)
        
        # Add 4 more images to this cluster
        while len(cluster_indices) < images_per_cluster:
            # Get features for current cluster and all used images
            cluster_features = features[cluster_indices]
            used_features = features[list(used_indices)]
            available_features = features[list(available_indices)]
            
            # Calculate similarities to both cluster and all used images
            cluster_similarities = cosine_similarity(available_features, cluster_features)
            used_similarities = cosine_similarity(available_features, used_features)
            
            # Combine both criteria
            cluster_max_sim = cluster_similarities.max(axis=1)  # Max similarity to cluster
            used_max_sim = used_similarities.max(axis=1)  # Max similarity to used images
            
            # Weight both factors (adjust weights if needed)
            combined_score = 0.7 * cluster_max_sim + 0.3 * used_max_sim
            
            # Select image with lowest combined score
            next_idx = list(available_indices)[np.argmin(combined_score)]
            cluster_indices.append(next_idx)
            used_indices.add(next_idx)
            available_indices.remove(next_idx)
        
        final_clusters[cluster_id] = cluster_indices
        print(f"Formed cluster {cluster_id + 1} with {len(cluster_indices)} images")
    
    return final_clusters

print("Performing global clustering with enhanced diversity...")
clusters = global_clustering(selected_features, num_clusters=20, images_per_cluster=5)
print(f"Created {len(clusters)} clusters with 5 images each")

### Part 4: Visualization and Analysis ###
os.makedirs(output_base_dir, exist_ok=True)

# Save dendrograms and distance matrices (similar to original)
def save_dendrogram(features, labels, output_path, title):
    if len(features) < 2:
        print(f"Cannot create dendrogram for {title} - not enough images")
        return
        
    plt.figure(figsize=(15, 10))
    linkage_matrix = linkage(features, method='ward')
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_distance_info(features, labels, output_path):
    if len(features) < 2:
        print(f"Cannot create distance matrix - not enough images")
        return
        
    distances = cosine_similarity(features)
    with open(output_path, 'w') as f:
        f.write("Distance Matrix (Cosine Similarity):\n\n")
        f.write("," + ",".join(labels) + "\n")
        for i, label in enumerate(labels):
            row = [label] + [f"{dist:.3f}" for dist in distances[i]]
            f.write(",".join(row) + "\n")

# Create main visualizations
main_dendrogram_path = os.path.join(output_base_dir, 'main_dendrogram.png')
save_dendrogram(selected_features, selected_labels, main_dendrogram_path, 
                'Global Hierarchical Clustering of 100 Selected Images')

# Create cluster visualizations and copy images
for cluster_id, indices in clusters.items():
    cluster_folder = os.path.join(output_base_dir, f"{cluster_id + 1}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    # Get cluster data
    cluster_features = selected_features[indices]
    cluster_labels = [selected_labels[i] for i in indices]
    
    # Only create visualizations if we have enough images
    if len(cluster_features) >= 2:
        # Save visualizations
        save_dendrogram(cluster_features, cluster_labels,
                       os.path.join(cluster_folder, 'cluster_dendrogram.png'),
                       f'Cluster {cluster_id + 1} Internal Structure')
        
        save_distance_info(cluster_features, cluster_labels,
                          os.path.join(cluster_folder, 'distance_matrix.csv'))
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        distances = cosine_similarity(cluster_features)
        sns.heatmap(distances, annot=True, fmt='.3f',
                    xticklabels=cluster_labels,
                    yticklabels=cluster_labels)
        plt.title(f'Distance Heatmap for Cluster {cluster_id + 1}')
        plt.tight_layout()
        plt.savefig(os.path.join(cluster_folder, 'distance_heatmap.png'))
        plt.close()
    else:
        print(f"Skipping visualizations for cluster {cluster_id + 1} - not enough images")
    
    # Copy images
    for idx in indices:
        img_label = selected_labels[idx]
        if img_label in feature_mapping:
            img_filename = feature_mapping[img_label]['image_file']
            src_path = os.path.join(source_directory, img_filename)
            dst_path = os.path.join(cluster_folder, img_filename)
            try:
                shutil.copy(src_path, dst_path)
                print(f"Copied {img_filename} to cluster {cluster_id + 1}")
            except Exception as e:
                print(f"Error copying {img_filename}: {str(e)}")

# Save selected images list
output_file = os.path.join(output_base_dir, 'selected_unique_objects.txt')
with open(output_file, 'w') as f:
    for label in selected_labels:
        f.write(f"{label}\n")

print(f"Analysis complete. Results saved in {output_base_dir}") 