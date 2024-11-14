import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform, pdist
import os
from sklearn.preprocessing import MinMaxScaler

### Part 1: CNN-based Features ###

feature_directory = os.path.join(os.getcwd(), r'C:\Users\kyjer\Documents\remote-viewing-exp\src\features')

feature_files = sorted([f for f in os.listdir(feature_directory) if f.endswith('.pt')])

cnn_feature_vectors = []
cnn_image_labels = [] 

for file in feature_files:
    feature_path = os.path.join(feature_directory, file)
    feature_vector = torch.load(feature_path).flatten().numpy() 
    cnn_feature_vectors.append(feature_vector)
    cnn_image_labels.append(os.path.splitext(file)[0])

cnn_feature_vectors = np.array(cnn_feature_vectors)

# Compute distance matrix
condensed_dist_matrix_cnn = pdist(cnn_feature_vectors, metric="correlation")
cnn_distance_matrix = squareform(condensed_dist_matrix_cnn)

# Compute linkage matrix using Ward's method
linkage_matrix_ward_cnn = linkage(condensed_dist_matrix_cnn, method='ward')

# Create dendrogram
plt.figure(figsize=(12, 8))
dn = dendrogram(
    linkage_matrix_ward_cnn,
    labels=cnn_image_labels,
    leaf_rotation=0,  
    leaf_font_size=10
)
plt.title("Hierarchical Clustering Dendrogram (CNN Features)")
plt.xlabel("")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Get the order of leaves from dendrogram
ordered_indices = leaves_list(linkage_matrix_ward_cnn)

# Convert distance matrix to similarity matrix (1 - distance)
similarity_matrix = cosine_similarity(cnn_feature_vectors)

# Reorder similarity matrix to match dendrogram
ordered_similarity_matrix = similarity_matrix[ordered_indices][:, ordered_indices]
ordered_labels = [cnn_image_labels[i] for i in ordered_indices]

# Create mask for diagonal
mask = np.eye(len(ordered_similarity_matrix), dtype=bool)

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# First heatmap with raw values
sns.heatmap(
    ordered_similarity_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    xticklabels=ordered_labels,
    yticklabels=ordered_labels,
    mask=mask,
    vmin=-1,
    vmax=1,
    center=0,
    ax=ax1
)
ax1.set_title("Raw Similarity Values (CNN Features)")
ax1.set_xlabel("")
ax1.set_ylabel("")

# For second heatmap, interpolate values between 0 and 1 based on actual data range
min_val = ordered_similarity_matrix[~mask].min()  # Get min value excluding diagonal
max_val = ordered_similarity_matrix[~mask].max()  # Get max value excluding diagonal

# Interpolate to [0,1] range
interpolated_matrix = (ordered_similarity_matrix - min_val) / (max_val - min_val)

sns.heatmap(
    interpolated_matrix,
    annot=ordered_similarity_matrix,  # Show original values but use interpolated colors
    fmt='.2f',
    cmap='coolwarm',
    xticklabels=ordered_labels,
    yticklabels=ordered_labels,
    mask=mask,
    vmin=0,    # Set range from 0 to 1 for interpolated values
    vmax=1,
    center=0.5,
    ax=ax2
)
ax2.set_title("Interpolated Similarity Values (CNN Features)")
ax2.set_xlabel("")
ax2.set_ylabel("")

plt.tight_layout()
plt.show()
