import scipy
import numpy as np

def generate_condensed_distance_matrix(feature_vectors, metric="correlation"):
    """
    Generate a condensed distance matrix using a specified metric.
    
    :param feature_vectors: numpy array of feature vectors (num_samples x num_features)
    :param metric: distance metric to use (default is "correlation")
    :return: condensed distance matrix
    """
    condensed_dist_matrix = scipy.spatial.distance.pdist(feature_vectors, metric=metric)
    return condensed_dist_matrix


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_dendrogram(condensed_dist_matrix, labels, method="ward"):
    """
    Generate a dendrogram for the given condensed distance matrix.
    
    :param condensed_dist_matrix: condensed distance matrix (1D array)
    :param labels: list of labels for each feature vector
    :param method: linkage method for hierarchical clustering (default is "ward")
    :return: the dendrogram object (contains 'ivl' for leaf order)
    """
    # Perform hierarchical clustering (Ward linkage)
    linkage_matrix = linkage(condensed_dist_matrix, method=method)
    
    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dn = dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=90,  
        leaf_font_size=12
    )
    plt.title(f"Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)")
    plt.xlabel("Image Filename")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
    
    return dn


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform

def generate_reordered_heatmap(condensed_dist_matrix, labels, dendrogram_obj, cmap="viridis"):
    """
    Generate and plot a reordered heatmap based on the dendrogram's leaf order.
    
    :param condensed_dist_matrix: condensed distance matrix (1D array)
    :param labels: list of labels for each feature vector
    :param dendrogram_obj: the dendrogram object (contains 'ivl' for leaf order)
    :param cmap: colormap for the heatmap (default is 'viridis')
    """
    # Convert the condensed distance matrix to square form
    distance_matrix = squareform(condensed_dist_matrix)
    
    # Get the order of labels from the dendrogram ('ivl' is the leaf order)
    ivl_order = dendrogram_obj["ivl"]
    
    # Get the indices of ivl_order in the original labels list
    reordered_indices = [labels.index(label) for label in ivl_order]
    
    # Reorder the distance matrix based on the dendrogram leaf order
    reordered_distance_matrix = distance_matrix[np.ix_(reordered_indices, reordered_indices)]
    
    # Plot the reordered heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(reordered_distance_matrix, annot=False, cmap=cmap, 
                xticklabels=ivl_order, yticklabels=ivl_order)
    plt.title("Reordered Cosine Similarity Heatmap")
    plt.xlabel("Image Filename")
    plt.ylabel("Image Filename")
    plt.tight_layout()
    plt.show()
