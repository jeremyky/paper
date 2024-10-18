import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
import os
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import squareform, pdist

### Part 1: SBERT-based Features ###

# Initialize the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the observation vectors from the file
observation_file_path = r'C:\Users\kyjer\Documents\david\dops-rv.exp\combined_descriptors.txt'
with open(observation_file_path, 'r') as file:
    observation_vectors = [line.strip() for line in file]

# Embed the observation vectors using SBERT
sbert_feature_vectors = [model.encode(obs_vector) for obs_vector in observation_vectors]

# Convert to a numpy array for further processing
sbert_feature_vectors = np.array(sbert_feature_vectors)

# Create labels for each observation vector (assuming you want to use line numbers as labels)
sbert_image_labels = [f"Observation {i+1}" for i in range(len(observation_vectors))]

condensed_dist_matrix_cnn = scipy.spatial.distance.pdist(sbert_feature_vectors, metric="correlation")
cnn_distance_matrix = squareform(condensed_dist_matrix_cnn)




linkage_matrix_ward_cnn = linkage(condensed_dist_matrix_cnn, method='ward')
print(linkage_matrix_ward_cnn)
plt.figure(figsize=(12, 8))
dn = dendrogram(
    linkage_matrix_ward_cnn,
    labels=sbert_image_labels,
    leaf_rotation=90,  
    leaf_font_size=12  
)
plt.title("Hierarchical Clustering Dendrogram (SBERT Features - Ward Linkage)")
plt.xlabel("Image Filename")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Retrieve the order of labels from the dendrogram
ivl_order = dn["ivl"]  # ivl should contain the labels

# Get the indices of ivl_order in the original sbert_image_labels list
reordered_indices = [sbert_image_labels.index(label) for label in ivl_order]

# Reorder the distance matrix based on the ivl order from the dendrogram
reordered_distance_matrix = cnn_distance_matrix[np.ix_(reordered_indices, reordered_indices)]

# Set a custom threshold to control the coloring of clusters
threshold = 0.4  # Adjust this value to control grouping

# Plot the dendrogram with the new color threshold
plt.figure(figsize=(12, 8))
dn = dendrogram(
    linkage_matrix_ward_cnn,
    labels=sbert_image_labels,
    leaf_rotation=90,  
    leaf_font_size=12,
    color_threshold=threshold  # Set the threshold here
)
plt.title("Hierarchical Clustering Dendrogram (SBERT Features - Ward Linkage)")
plt.xlabel("Image Filename")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

print(dn["ivl"])

# Plot the reordered heatmap to align with the dendrogram structure
plt.figure(figsize=(10, 8))
sns.heatmap(reordered_distance_matrix, annot=True, cmap='coolwarm',  # Change colormap to 'viridis'
            xticklabels=ivl_order, yticklabels=ivl_order)
plt.title("Reordered Cosine Similarity Heatmap (SBERT Features)")
plt.xlabel("Image Filename")
plt.ylabel("Image Filename")
plt.tight_layout()
plt.show()
