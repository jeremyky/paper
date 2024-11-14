import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster
from sentence_transformers import SentenceTransformer
import matplotlib.colors as mcolors

### Part 1: Initialize and Load Data ###

# Initialize SBERT model - using MiniLM for efficient semantic encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load image descriptions from file
observation_file_path = r'C:\Users\kyjer\Documents\david\dops-rv.exp\combined_descriptors.txt'
with open(observation_file_path, 'r') as file:
    observation_vectors = [line.strip() for line in file]

# Generate SBERT embeddings for each description
sbert_feature_vectors = [model.encode(obs_vector) for obs_vector in observation_vectors]
sbert_feature_vectors = np.array(sbert_feature_vectors)
sbert_image_labels = [f"Image {i+1}" for i in range(len(observation_vectors))]

### Part 2: Similarity Computation ###

# Compute cosine similarity matrix
similarity_matrix = cosine_similarity(sbert_feature_vectors)
distance_matrix = 1 - similarity_matrix  # Convert to distance matrix

### Part 3: Hierarchical Clustering Analysis ###

# Get actual image filenames from the targets folder (assuming they're numbered .jpg files)
image_labels = [f"{i}.jpg" for i in range(1, len(observation_vectors) + 1)]

# Compute linkage matrix using Ward's method
linkage_matrix = linkage(scipy.spatial.distance.pdist(sbert_feature_vectors), method='ward')

# Create simple, clean dendrogram
plt.figure(figsize=(12, 8))
dendrogram(
    linkage_matrix,
    labels=image_labels,
    leaf_rotation=0,  # Keep labels horizontal
    leaf_font_size=10,  # Slightly smaller font
)
plt.title("Hierarchical Clustering Dendrogram (Threshold: 0.30)")
plt.xlabel("")  # Remove x-axis label
plt.ylabel("Distance")

# Add threshold line
plt.axhline(y=0.3, color='r', linestyle='--', label='Threshold: 0.30')
plt.legend()

plt.tight_layout()
plt.show()

# Print cluster assignments
cluster_dict = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = []
    cluster_dict[cluster_id].append(sbert_image_labels[idx])

print(f"\nCluster Assignments (Ward linkage):")
for cluster_id, images in cluster_dict.items():
    print(f"\nCluster {cluster_id}:")
    print(", ".join(images))

### Part 4: Similarity Analysis ###

# Create heatmap using the best linkage method's ordering
final_linkage = linkage(scipy.spatial.distance.pdist(sbert_feature_vectors), method='average')
ordered_indices = leaves_list(final_linkage)
ordered_similarity_matrix = similarity_matrix[ordered_indices][:, ordered_indices]
ordered_labels = [sbert_image_labels[i] for i in ordered_indices]

plt.figure(figsize=(10, 8))
sns.heatmap(
    ordered_similarity_matrix,
    annot=True,
    cmap='coolwarm',
    xticklabels=ordered_labels,
    yticklabels=ordered_labels
)
plt.title("Cosine Similarity Heatmap (SBERT Features)")
plt.xlabel("Image Index")
plt.ylabel("Image Index")
plt.tight_layout()
plt.show()

### Part 5: Top Pairs Analysis ###

# Find most similar pairs
top_n = 5
flattened_similarity = []
for i in range(len(ordered_similarity_matrix)):
    for j in range(i+1, len(ordered_similarity_matrix)):
        flattened_similarity.append((ordered_similarity_matrix[i, j], i, j))

# Display top similar pairs
flattened_similarity.sort(reverse=True, key=lambda x: x[0])
print(f"\nTop {top_n} most related observation vectors:\n")
for score, i, j in flattened_similarity[:top_n]:
    print(f"Similarity: {score:.4f}")
    print(f"Image {ordered_labels[i]} and Image {ordered_labels[j]}:")
    print(f" - {observation_vectors[ordered_indices[i]]}")
    print(f" - {observation_vectors[ordered_indices[j]]}")
    print()
