import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
import random
import supportfunctions as support
from sklearn.metrics import pairwise_distances

# ---------------------- CNN-based Features ---------------------- #
# Path to the CNN feature directory
cnn_feature_directory = r"C:\Users\kyjer\Documents\david\dops-rv.exp\features"

# Load CNN feature vectors and corresponding labels
cnn_feature_files = sorted([f for f in os.listdir(cnn_feature_directory) if f.endswith('.pt')])

cnn_feature_vectors = []
cnn_image_labels = [] 

for file in cnn_feature_files:
    feature_path = os.path.join(cnn_feature_directory, file)
    feature_vector = torch.load(feature_path).flatten().numpy() 
    cnn_feature_vectors.append(feature_vector)
    cnn_image_labels.append(os.path.splitext(file)[0])

# Convert to numpy array
cnn_feature_vectors = np.array(cnn_feature_vectors)

# ---------------------- SBERT-based Features ---------------------- #
# Load the observation vectors for SBERT embeddings
observation_file_path = r'C:\Users\kyjer\Documents\david\dops-rv.exp\combined_descriptors.txt'

# Initialize the SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load observation vectors
with open(observation_file_path, 'r') as file:
    observation_vectors = [line.strip() for line in file]

# Embed the observation vectors using SBERT
sbert_feature_vectors = [model.encode(obs_vector) for obs_vector in observation_vectors]

# Convert to numpy array
sbert_feature_vectors = np.array(sbert_feature_vectors)

# ---------------------- Sampling to Ensure Same Number of Samples ---------------------- #
# Make sure both sets have the same number of samples
min_samples = min(cnn_feature_vectors.shape[0], sbert_feature_vectors.shape[0])

# Randomly sample min_samples from both sets
sample_indices = random.sample(range(min_samples), min_samples)

cnn_sampled_feature_vectors = cnn_feature_vectors[sample_indices]
sbert_sampled_feature_vectors = sbert_feature_vectors[sample_indices]

# Generate condensed distance matrices
condensed_dist_matrix_cnn = support.generate_condensed_distance_matrix(cnn_sampled_feature_vectors, metric="correlation")
condensed_dist_matrix_sbert = support.generate_condensed_distance_matrix(sbert_sampled_feature_vectors, metric="correlation")

# ---------------------- Similarity Space Measures ---------------------- #

# 1. Pearson Correlation - Measures the linear correlation between the distance matrices
correlation_pearson, p_value_pearson = pearsonr(condensed_dist_matrix_cnn, condensed_dist_matrix_sbert)
print(f"Pearson Correlation between CNN and SBERT similarity spaces: {correlation_pearson}")
print(f"Pearson P-value: {p_value_pearson}")

# 2. Spearman Rank Correlation - Measures the monotonic relationship between distance matrices
correlation_spearman, p_value_spearman = spearmanr(condensed_dist_matrix_cnn, condensed_dist_matrix_sbert)
print(f"Spearman Rank Correlation between CNN and SBERT similarity spaces: {correlation_spearman}")
print(f"Spearman P-value: {p_value_spearman}")

# ^ double check image order input to make sure that we get positive correlation
# ^ (image input into the model for distance matrix)
# create github to dops to be able to send from local pc to hpc

# ---------------------- Procrustes Analysis ---------------------- #
# Option 1: Truncate both sets to have the same number of dimensions
'''min_dimensions = min(cnn_sampled_feature_vectors.shape[1], sbert_sampled_feature_vectors.shape[1])

# Truncate CNN and SBERT feature vectors to the minimum number of dimensions
cnn_sampled_feature_vectors_truncated = cnn_sampled_feature_vectors[:, :min_dimensions]
sbert_sampled_feature_vectors_truncated = sbert_sampled_feature_vectors[:, :min_dimensions]

# Perform Procrustes analysis on truncated feature vectors
matrix1_truncated, matrix2_truncated, disparity_truncated = procrustes(cnn_sampled_feature_vectors_truncated, sbert_sampled_feature_vectors_truncated)
print(f"Procrustes Disparity (Truncated CNN vs SBERT): {disparity_truncated}")

# Option 2: Zero-pad the smaller set to match the larger set
max_dimensions = max(cnn_sampled_feature_vectors.shape[1], sbert_sampled_feature_vectors.shape[1])

# Zero-pad both feature sets to have the same number of dimensions
cnn_sampled_feature_vectors_padded = np.pad(cnn_sampled_feature_vectors, ((0, 0), (0, max_dimensions - cnn_sampled_feature_vectors.shape[1])), 'constant')
sbert_sampled_feature_vectors_padded = np.pad(sbert_sampled_feature_vectors, ((0, 0), (0, max_dimensions - sbert_sampled_feature_vectors.shape[1])), 'constant')

# Perform Procrustes analysis on zero-padded feature vectors
matrix1_padded, matrix2_padded, disparity_padded = procrustes(cnn_sampled_feature_vectors_padded, sbert_sampled_feature_vectors_padded)
print(f"Procrustes Disparity (Padded CNN vs SBERT): {disparity_padded}")

# Option 3: PCA to reduce both to the same dimensionality
num_components = min(cnn_sampled_feature_vectors.shape[1], sbert_sampled_feature_vectors.shape[1])

# Apply PCA to both CNN and SBERT feature vectors
pca = PCA(n_components=num_components)
cnn_sampled_feature_vectors_pca = pca.fit_transform(cnn_sampled_feature_vectors)
sbert_sampled_feature_vectors_pca = pca.fit_transform(sbert_sampled_feature_vectors)

# Perform Procrustes analysis on PCA-reduced feature vectors
matrix1_pca, matrix2_pca, disparity_pca = procrustes(cnn_sampled_feature_vectors_pca, sbert_sampled_feature_vectors_pca)
print(f"Procrustes Disparity (PCA CNN vs SBERT): {disparity_pca}")'''

# ---------------------- Additional Metric: MMD ---------------------- #
'''def rbf_kernel(X, Y, gamma=1.0):
    """
    Computes the RBF kernel between two datasets.
    :param X: First dataset (numpy array)
    :param Y: Second dataset (numpy array)
    :param gamma: Kernel width (default 1.0)
    :return: RBF kernel matrix
    """
    K = np.exp(-gamma * pairwise_distances(X, Y, metric='euclidean') ** 2)
    return K

def compute_mmd(X, Y, gamma=1.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two datasets.
    :param X: First dataset (numpy array)
    :param Y: Second dataset (numpy array)
    :param gamma: Kernel width (default 1.0)
    :return: MMD value
    """
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    mmd_value = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd_value

# Compute MMD between CNN and SBERT
mmd_value = compute_mmd(cnn_sampled_feature_vectors, sbert_sampled_feature_vectors, gamma=1.0)
print(f"Maximum Mean Discrepancy (MMD) between CNN and SBERT: {mmd_value}")
'''