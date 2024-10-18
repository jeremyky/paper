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

### Part 1: CNN-based Features ###

feature_directory = os.path.join(os.getcwd(), 'features')

feature_files = sorted([f for f in os.listdir(feature_directory) if f.endswith('.pt')])

cnn_feature_vectors = []
cnn_image_labels = [] 

for file in feature_files:
    feature_path = os.path.join(feature_directory, file)
    feature_vector = torch.load(feature_path).flatten().numpy() 
    cnn_feature_vectors.append(feature_vector)
    cnn_image_labels.append(os.path.splitext(file)[0])


cnn_feature_vectors = np.array(cnn_feature_vectors)

condensed_dist_matrix_cnn = scipy.spatial.distance.pdist(cnn_feature_vectors, metric="correlation")
cnn_distance_matrix = squareform(condensed_dist_matrix_cnn)




linkage_matrix_ward_cnn = linkage(condensed_dist_matrix_cnn, method='ward')
print(linkage_matrix_ward_cnn)
plt.figure(figsize=(12, 8))
dn = dendrogram(
    linkage_matrix_ward_cnn,
    labels=cnn_image_labels,
    leaf_rotation=90,  
    leaf_font_size=12  
)
plt.title("Hierarchical Clustering Dendrogram (CNN Features - Ward Linkage)")
plt.xlabel("Image Filename")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Retrieve the order of labels from the dendrogram
ivl_order = dn["ivl"]  # ivl should contain the labels

# Get the indices of ivl_order in the original cnn_image_labels list
reordered_indices = [cnn_image_labels.index(label) for label in ivl_order]

# Reorder the distance matrix based on the ivl order from the dendrogram
reordered_distance_matrix = cnn_distance_matrix[np.ix_(reordered_indices, reordered_indices)]

# Plot the reordered heatmap to align with the dendrogram structure
plt.figure(figsize=(10, 8))
sns.heatmap(reordered_distance_matrix, annot=True, cmap='coolwarm',  # Change colormap to 'viridis'
            xticklabels=ivl_order, yticklabels=ivl_order)
plt.title("Reordered Cosine Similarity Heatmap (CNN Features)")
plt.xlabel("Image Filename")
plt.ylabel("Image Filename")
plt.tight_layout()
plt.show()
