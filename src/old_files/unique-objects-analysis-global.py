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
from config.experiment_config import (
    CLUSTERING_ALGORITHM,
    ALGORITHM_CONFIG,
    FEATURES_DIR,
    OBJECTS_DIR,
    CLUSTERS_DIR,
    FIGURE_SIZE,
    HEATMAP_COLORMAP,
    DENDROGRAM_LEAF_ROTATION,
    LOGGING_CONFIG
)
import logging
from analysis.comparison_analysis import ClusteringComparison
from algorithms.legacy_algorithms import (
    select_most_diverse_original,
    global_clustering_original
)

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')

# Set random seed
random.seed(42)
np.random.seed(42)

def main():
    logger.info(f"Using {CLUSTERING_ALGORITHM} clustering algorithm")
    
    # Use config paths
    feature_directory = FEATURES_DIR
    source_directory = OBJECTS_DIR
    output_base_dir = CLUSTERS_DIR
    
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

    # Get clustering algorithm
    algorithm = get_clustering_algorithm(CLUSTERING_ALGORITHM, cnn_feature_vectors, ALGORITHM_CONFIG)
    
    # Select diverse images
    print(f"Using {CLUSTERING_ALGORITHM} clustering algorithm")
    print(f"Algorithm description:\n{ALGORITHM_CONFIG['description']}")
    selected_indices = algorithm.select_images()
    
    # Run random selection for comparison
    random_indices = np.random.choice(len(cnn_feature_vectors), len(selected_indices), replace=False)
    
    # Compare selections
    comparison = ClusteringComparison(
        cnn_feature_vectors,
        cnn_image_labels,
        os.path.join(output_base_dir, 'comparison_analysis')
    )
    
    # Run comprehensive validation
    validation_results = comparison.comprehensive_validation(
        selected_indices,
        n_simulations=1000,
        group_size=5
    )
    
    # Save comparison results
    with open(os.path.join(output_base_dir, 'comparison_results.txt'), 'w') as f:
        f.write("Comparison Results\n\n")
        f.write("Algorithm Selection:\n")
        f.write(f"Closest pair: {validation_results['algorithm']['closest_pair']}\n")
        f.write(f"Closest similarity: {validation_results['algorithm']['closest_similarity']:.3f}\n")
        f.write(f"Mean similarity: {validation_results['algorithm']['mean_similarity']:.3f}\n")
        f.write(f"Min similarity: {validation_results['algorithm']['min_similarity']:.3f}\n\n")
        
        f.write("Random Selection:\n")
        f.write(f"Closest pair: {validation_results['random']['closest_pair']}\n")
        f.write(f"Closest similarity: {validation_results['random']['closest_similarity']:.3f}\n")
        f.write(f"Mean similarity: {validation_results['random']['mean_similarity']:.3f}\n")
        f.write(f"Min similarity: {validation_results['random']['min_similarity']:.3f}\n\n")
        
        f.write("Monte Carlo Analysis:\n")
        f.write("Minimum Similarities:\n")
        f.write(f"Algorithm: {validation_results['algorithm']['min_similarity']:.3f}\n")
        f.write(f"Random (mean): {validation_results['monte_carlo']['min_similarities']['mean']:.3f}\n")
        f.write(f"Random (std): {validation_results['monte_carlo']['min_similarities']['std']:.3f}\n")
        f.write(f"Random (best): {validation_results['monte_carlo']['min_similarities']['best']:.3f}\n")
        f.write(f"Random (worst): {validation_results['monte_carlo']['min_similarities']['worst']:.3f}\n")
    
    # Continue with clustering and visualization...

    # Get cluster assignments
    clusters = algorithm.cluster_images(selected_indices)
    
    # Visualization and Analysis
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create visualizations using the selected indices and clusters
    create_visualizations(selected_features, selected_labels, clusters, output_base_dir)

    # Save selected images list
    output_file = os.path.join(output_base_dir, 'selected_unique_objects.txt')
    with open(output_file, 'w') as f:
        for label in selected_labels:
            f.write(f"{label}\n")

    print(f"Analysis complete. Results saved in {output_base_dir}") 