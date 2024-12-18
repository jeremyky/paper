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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

### Part 1: Load CNN-based Features ###

# Update paths to be relative to project root
feature_directory = os.path.join(os.getcwd(), 'src', 'unique_objects_features')
source_directory = os.path.join(os.getcwd(), 'ObjectsAll', 'OBJECTSALL')  # This is where the original images are
output_base_dir = os.path.join(os.getcwd(), 'src', 'selected_clusters')

# Load the mapping file first
mapping_file = os.path.join(feature_directory, 'feature_mapping.json')
try:
    with open(mapping_file, 'r') as f:
        feature_mapping = json.load(f)
    print(f"Loaded feature mapping file with {len(feature_mapping)} entries.")
except FileNotFoundError:
    raise FileNotFoundError(f"Mapping file not found at {mapping_file}. Please run feature extraction first.")

# Load feature files using the mapping
feature_files = sorted([f['feature_file'] for f in feature_mapping.values()])
cnn_feature_vectors = []
cnn_image_labels = []

for feature_file in feature_files:
    feature_path = os.path.join(feature_directory, feature_file)
    try:
        feature_vector = torch.load(feature_path).flatten().numpy()
        cnn_feature_vectors.append(feature_vector)
        cnn_image_labels.append(os.path.splitext(feature_file)[0])
    except Exception as e:
        print(f"Error loading feature file {feature_file}: {str(e)}")

cnn_feature_vectors = np.array(cnn_feature_vectors)
print(f"Loaded {len(cnn_feature_vectors)} feature vectors.")

### Part 2: Select 100 Most Diverse Images from 2400 ###

# Compute distance matrix using cosine distance
# Since AgglomerativeClustering does not support precomputed distance for large datasets efficiently,
# we'll use a different approach to select 100 diverse images.

# Approach: Max-Min Distance Selection

def select_most_diverse(features, num_select=100):
    selected_indices = []
    remaining_indices = set(range(len(features)))

    # Start by selecting a random image
    first = random.choice(list(remaining_indices))
    selected_indices.append(first)
    remaining_indices.remove(first)

    # Iteratively select the image that has the maximum minimum distance to the selected set
    for _ in range(1, num_select):
        # Compute cosine similarity between the remaining and selected
        selected_features = features[selected_indices]
        remaining_features = features[list(remaining_indices)]
        similarities = cosine_similarity(remaining_features, selected_features)
        # Compute the minimum similarity for each remaining image
        min_similarities = similarities.min(axis=1)
        # Select the image with the least similarity (most diverse)
        most_diverse_idx = np.argmin(min_similarities)
        most_diverse_image = list(remaining_indices)[most_diverse_idx]
        selected_indices.append(most_diverse_image)
        remaining_indices.remove(most_diverse_image)
        if len(selected_indices) % 10 == 0:
            print(f"Selected {len(selected_indices)} diverse images so far...")
    
    return selected_indices

print("Selecting 100 most diverse images from 2400...")
selected_indices = select_most_diverse(cnn_feature_vectors, num_select=100)
selected_labels = [cnn_image_labels[i] for i in selected_indices]
selected_features = cnn_feature_vectors[selected_indices]

print(f"Selected {len(selected_indices)} diverse images for analysis.")

### Part 3: Random Selection of 100 Images for Comparison ###

random_indices = random.sample(range(len(cnn_feature_vectors)), 100)
random_labels = [cnn_image_labels[i] for i in random_indices]
random_features = cnn_feature_vectors[random_indices]

print("Randomly selected 100 images for comparison.")

### Part 4: Statistical Comparison Using Monte Carlo Simulation ###

def average_min_distance(features):
    # Compute pairwise cosine distances
    pairwise_dist = pdist(features, metric='cosine')
    return np.mean(pairwise_dist)


def find_min_distance_amongst_100(features):
    # Compute pairwise cosine distances
    pairwise_dist = pdist(features, metric='cosine')
    return np.min(pairwise_dist)

# Calculate average minimum distance for clustered selection
clustered_avg_distance = average_min_distance(selected_features)

# Calculate average minimum distance for random selections
print("Running Monte Carlo simulation...")
num_simulations = 1000
random_avg_distances = []
random_min_distances = []
for i in range(num_simulations):
    rand_indices = random.sample(range(len(cnn_feature_vectors)), 100)
    rand_features = cnn_feature_vectors[rand_indices]
    rand_avg = average_min_distance(rand_features)
    random_avg_distances.append(rand_avg)
    rand_min = find_min_distance_amongst_100(rand_features)
    random_min_distances.append(rand_min)
    if (i+1) % 100 == 0:
        print("Mean dist of random: " + str(np.mean(random_avg_distances)))
        print("Min dist of random: " + str(np.min(random_avg_distances)))
        print("Max dist of random: " + str(np.max(random_avg_distances)))
        print("RANDOM Min dist amongst 100: " + str(find_min_distance_amongst_100(rand_features)))
        print(f"Completed {i+1}/{num_simulations} simulations")

random_avg_distances = np.array(random_avg_distances)
random_min_distances = np.array(random_min_distances)
# Calculate p-value
print("Clust Avg of mine: " + str(clustered_avg_distance))
print("ALGORITHM Min dist amongst 100 : " + str(find_min_distance_amongst_100(selected_features)))
print("# of Random  greater than the algorithm")
print(np.sum(random_avg_distances >= clustered_avg_distance))
p_value = (np.sum(random_avg_distances >= clustered_avg_distance)) / num_simulations
print(f"\nFinal P-value: {p_value:.10f}")

random_min_greater_than_algorithm = np.sum(random_min_distances >= find_min_distance_amongst_100(selected_features))
print("# of Random Min dist greater than the algorithm")
print(random_min_greater_than_algorithm)
print("Average of Random Min that are closest to eachother")
print(np.mean(random_min_distances))
# p_value_min = random_min_greater_than_algorithm / num_simulations
# print(f"\nFinal P-value Min: {p_value_min:.10f}")


### Part 5: Cluster the 100 Selected Images into 20 Sub-clusters of 5 ###

# To maximize intra-cluster diversity, we'll again use a Max-Min approach within each cluster.

def cluster_into_subsets(features, labels, num_subclusters=20, images_per_cluster=5):
    clusters = {i: [] for i in range(num_subclusters)}
    remaining = set(range(len(features)))

    for cluster_id in range(num_subclusters):
        if not remaining:
            break
        # Select the first image randomly
        first = random.choice(list(remaining))
        clusters[cluster_id].append(first)
        remaining.remove(first)

        while len(clusters[cluster_id]) < images_per_cluster and remaining:
            selected_features = features[clusters[cluster_id]]
            remaining_features = features[list(remaining)]
            similarities = cosine_similarity(remaining_features, selected_features)
            avg_similarities = similarities.mean(axis=1)
            # Select the image with the least average similarity to the cluster
            most_diverse_idx = np.argmin(avg_similarities)
            most_diverse_image = list(remaining)[most_diverse_idx]
            clusters[cluster_id].append(most_diverse_image)
            remaining.remove(most_diverse_image)
        print(f"Formed cluster {cluster_id + 1} with {len(clusters[cluster_id])} images.")

    return clusters

print("Clustering the 100 selected images into 20 sub-clusters of 5...")
clusters = cluster_into_subsets(selected_features, selected_labels, num_subclusters=20, images_per_cluster=5)

# Verify that each cluster has exactly 5 images
for cluster_id, indices in clusters.items():
    if len(indices) != 5:
        print(f"Warning: Cluster {cluster_id + 1} has {len(indices)} images instead of 5.")

### Additional Analysis: Dendrograms and Distance Information ###

print("Generating dendrograms and distance information...")

# Function to create and save dendrogram
def save_dendrogram(features, labels, output_path, title):
    plt.figure(figsize=(15, 10))
    linkage_matrix = linkage(features, method='ward')
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to save distance matrix information
def save_distance_info(features, labels, output_path):
    distances = cosine_similarity(features)
    with open(output_path, 'w') as f:
        f.write("Distance Matrix (Cosine Similarity):\n\n")
        # Write header
        f.write("," + ",".join(labels) + "\n")
        # Write distances
        for i, label in enumerate(labels):
            row = [label] + [f"{dist:.3f}" for dist in distances[i]]
            f.write(",".join(row) + "\n")

# 1. Create main dendrogram for all 100 images
main_dendrogram_path = os.path.join(output_base_dir, 'main_dendrogram.png')
save_dendrogram(selected_features, selected_labels, main_dendrogram_path, 
                'Hierarchical Clustering of 100 Selected Images')

# 2. For each cluster, create dendrogram and distance information
for cluster_id, indices in clusters.items():
    cluster_folder = os.path.join(output_base_dir, f"{cluster_id + 1}")
    
    # Get cluster-specific data
    cluster_features = selected_features[indices]
    cluster_labels = [selected_labels[i] for i in indices]
    
    # Save cluster dendrogram
    cluster_dendrogram_path = os.path.join(cluster_folder, 'cluster_dendrogram.png')
    save_dendrogram(cluster_features, cluster_labels, cluster_dendrogram_path,
                   f'Hierarchical Clustering of Cluster {cluster_id + 1}')
    
    # Save distance information
    distance_info_path = os.path.join(cluster_folder, 'distance_matrix.csv')
    save_distance_info(cluster_features, cluster_labels, distance_info_path)
    
    # Create and save heatmap visualization
    plt.figure(figsize=(10, 8))
    distances = cosine_similarity(cluster_features)
    sns.heatmap(distances, annot=True, fmt='.3f', 
                xticklabels=cluster_labels, 
                yticklabels=cluster_labels)
    plt.title(f'Distance Heatmap for Cluster {cluster_id + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(cluster_folder, 'distance_heatmap.png'))
    plt.close()

print("Generated dendrograms and distance information for all clusters.")

# 3. Create visualization of the 20-way clustering
plt.figure(figsize=(20, 10))
# Compute linkage matrix for the 20-way clustering
linkage_matrix_20 = linkage(selected_features, method='ward')
# Draw dendrogram with colors for 20 clusters
dendrogram(linkage_matrix_20, labels=selected_labels, 
          leaf_rotation=90, 
          color_threshold=linkage_matrix_20[-20, 2])
plt.title('20-way Clustering of 100 Selected Images')
plt.tight_layout()
plt.savefig(os.path.join(output_base_dir, 'twenty_way_clustering.png'))
plt.close()

### Part 6: Save Final Selected Images into Organized Folders ###

print("Organizing selected images into folders...")

# Define source directory where original images are stored - FIXED PATH
source_directory = os.path.join(os.getcwd(), 'ObjectsAll', 'OBJECTSALL')  # Changed from 'src/targets'

if not os.path.isdir(source_directory):
    raise FileNotFoundError(f"Source image directory does not exist: {source_directory}")

print(f"Looking for images in: {source_directory}")

# Define output base directory
output_base_dir = os.path.join(os.getcwd(), 'src', 'selected_clusters')
os.makedirs(output_base_dir, exist_ok=True)

# Create folders 1-20 and copy images
for cluster_id, indices in clusters.items():
    cluster_folder = os.path.join(output_base_dir, f"{cluster_id + 1}")
    os.makedirs(cluster_folder, exist_ok=True)
    
    for img_idx in indices:
        img_label = selected_labels[img_idx]
        
        if img_label not in feature_mapping:
            print(f"Warning: No mapping found for label '{img_label}'")
            continue
            
        # Get the original image filename from the mapping
        img_filename = feature_mapping[img_label]['image_file']
        src_path = os.path.join(source_directory, img_filename)
        
        if not os.path.exists(src_path):
            print(f"Warning: Image file not found at {src_path}")
            print(f"Looking for: {img_filename}")
            continue
            
        dst_path = os.path.join(cluster_folder, img_filename)
        try:
            shutil.copy(src_path, dst_path)
            print(f"Copied {img_filename} to {cluster_folder}")
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {str(e)}")

### Part 7: Save Final Selected Images List ###

output_file = os.path.join(os.getcwd(), 'src', 'selected_unique_objects.txt')
with open(output_file, 'w') as f:
    for label in selected_labels:
        f.write(f"{label}\n")
print(f"Final selected 100 unique objects saved to {output_file}") 