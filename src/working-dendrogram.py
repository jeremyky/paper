import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, fcluster

# Initialize the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the observation vectors from the file
with open(r'C:\Users\kyjer\Documents\david\dops-rv.exp\combined_descriptors.txt', 'r') as file:
    observation_vectors = [line.strip() for line in file]

# Embedding the whole observation vector using SBERT
embeddings_whole = [model.encode(obs_vector) for obs_vector in observation_vectors]
print(len(embeddings_whole))
# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(embeddings_whole)

# Set diagonal elements (self-comparisons) to 0 instead of 1
np.fill_diagonal(similarity_matrix, 0)

# Convert similarity matrix to cosine distance (1 - similarity)
distance_matrix = 1 - similarity_matrix

# Perform hierarchical clustering using the cosine distance matrix
# Use complete-linkage and single-linkage
linkage_matrix_complete = linkage(distance_matrix, method='complete')
linkage_matrix_single = linkage(distance_matrix, method='single')

# Reorder the similarity matrix according to the hierarchical clustering (complete-linkage)
ordered_indices_complete = leaves_list(linkage_matrix_complete)
ordered_similarity_matrix_complete = similarity_matrix[ordered_indices_complete, :][:, ordered_indices_complete]

# Create labels for the images (1-based index)
image_labels = [f"Image {i+1}" for i in range(len(observation_vectors))]
ordered_labels_complete = [image_labels[i] for i in ordered_indices_complete]

# Create a heatmap to visualize the reordered similarity matrix (based on complete-linkage)
plt.figure(figsize=(10, 8))
sns.heatmap(ordered_similarity_matrix_complete, annot=True, cmap='coolwarm', xticklabels=ordered_labels_complete, yticklabels=ordered_labels_complete)
plt.title("Cosine Similarity Heatmap (Hierarchically Clustered - Complete Linkage)")
plt.xlabel("Image Index")
plt.ylabel("Image Index")
plt.tight_layout()  # Ensure the heatmap is neatly displayed
plt.show()

'''# Plot the dendrogram for complete-linkage hierarchical clustering
plt.figure(figsize=(12, 8))
dendrogram(
    linkage_matrix_complete,
    labels=ordered_labels_complete,
    leaf_rotation=90,  # Rotate labels for easier readability
    leaf_font_size=12,  # Increase font size for readability
    color_threshold=0.5  # New threshold for coloring the clusters
)
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage with New Threshold)")
plt.xlabel("Image Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
'''
# ---------------------------------------------------------------

# Number of top related pairs to display
top_n = 5

# Flatten the upper triangle of the similarity matrix (excluding diagonal)
flattened_similarity = []
for i in range(len(ordered_similarity_matrix_complete)):
    for j in range(i+1, len(ordered_similarity_matrix_complete)):
        flattened_similarity.append((ordered_similarity_matrix_complete[i, j], i, j))

# Sort the pairs by similarity score in descending order
flattened_similarity.sort(reverse=True, key=lambda x: x[0])

# Output the top N most related pairs
print(f"Top {top_n} most related observation vectors:\n")
for score, i, j in flattened_similarity[:top_n]:
    print(f"Similarity: {score:.4f}")
    print(f"Image {ordered_labels_complete[i]} and Image {ordered_labels_complete[j]}:")
    print(f" - {observation_vectors[ordered_indices_complete[i]]}")
    print(f" - {observation_vectors[ordered_indices_complete[j]]}")
    print("\n")

# ---------------------------------------------------------------

# Perform hierarchical clustering using the cosine similarity matrix
# Use complete linkage for clusters
distance_matrix = 1 - similarity_matrix
#sqf_dist_matrix = scipy.spatial.distance.squareform(distance_matrix)
condensed_dist_matrix = scipy.spatial.distance.pdist(embeddings_whole)
linkage_matrix_ward= linkage(condensed_dist_matrix, method='ward')
dendrogram(linkage_matrix_ward)
plt.show()
'''# Define a new threshold for the dendrogram at distance 0.5 for tighter clusters
new_threshold = 0.5  # Adjusted threshold value

# Extract cluster labels for each observation
cluster_labels_complete = fcluster(linkage_matrix_complete, new_threshold, criterion='distance')

# Create a dictionary to store images by their cluster
clusters = {}
for i, label in enumerate(cluster_labels_complete):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(f"Image {ordered_indices_complete[i] + 1}")

# Output the clusters
print(f"Clusters formed with threshold {new_threshold} (Complete Linkage):\n")
for cluster_id, images in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(images)}")

# Output the threshold used
print(f"New threshold used for clustering: {new_threshold}")
'''