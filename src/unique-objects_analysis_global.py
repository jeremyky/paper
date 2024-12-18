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
import logging

logger = logging.getLogger(__name__)

# Global variables to store metrics
_metrics = {
    'algorithm_metrics': None,
    'random_metrics': None,
    'p_value': None,
    'clusters': {},
    'features': None
}

def get_algorithm_metrics():
    return _metrics['algorithm_metrics']

def get_random_metrics():
    return _metrics['random_metrics']

def get_p_value():
    return _metrics['p_value']

def get_cluster_metrics():
    cluster_metrics = {}
    for cluster_id, indices in _metrics['clusters'].items():
        cluster_features = _metrics['features'][indices]
        distances = 1 - cosine_similarity(cluster_features)
        cluster_metrics[cluster_id] = {
            'mean_dist': float(np.mean(distances[np.triu_indices_from(distances, k=1)])),
            'min_dist': float(np.min(distances[np.triu_indices_from(distances, k=1)])),
            'max_dist': float(np.max(distances[np.triu_indices_from(distances, k=1)]))
        }
    return cluster_metrics

def save_clustering(output_path):
    """Save current clustering state"""
    clustering_data = {
        'clusters': _metrics['clusters'],
        'features': _metrics['features'].tolist() if _metrics['features'] is not None else None,
        'algorithm_metrics': float(_metrics['algorithm_metrics']) if _metrics['algorithm_metrics'] is not None else None,
        'random_metrics': _metrics['random_metrics'].tolist() if _metrics['random_metrics'] is not None else None,
        'p_value': float(_metrics['p_value']) if _metrics['p_value'] is not None else None
    }
    with open(output_path + '.json', 'w') as f:
        json.dump(clustering_data, f)

def run_analysis_from_baseline(baseline_path, output_dir):
    """Run analysis using saved baseline clustering"""
    logger.info(f"Loading baseline from: {baseline_path}")
    
    try:
        # Load baseline data
        with open(baseline_path + '.json', 'r') as f:
            baseline_data = json.load(f)
        
        # Extract data
        features = np.array(baseline_data['features'])
        clusters = baseline_data['clusters']
        
        # Calculate algorithm metrics
        distances = 1 - cosine_similarity(features)
        algorithm_mean_dist = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        # Run Monte Carlo simulation
        n_simulations = 1000
        random_distances = []
        n_samples = len(features)
        
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        for i in range(n_simulations):
            if i % 100 == 0:
                logger.info(f"Simulation progress: {i}/{n_simulations}")
                
            # Randomly select same number of images
            random_indices = np.random.choice(n_samples, size=n_samples, replace=False)
            random_features = features[random_indices]
            
            # Calculate mean distance for random selection
            random_sim = cosine_similarity(random_features)
            random_dist = np.mean(1 - random_sim[np.triu_indices_from(random_sim, k=1)])
            random_distances.append(random_dist)
        
        random_distances = np.array(random_distances)
        p_value = np.mean(random_distances >= algorithm_mean_dist)
        
        # Calculate per-cluster metrics
        cluster_metrics = {}
        for cluster_id, indices in clusters.items():
            cluster_features = features[indices]
            cluster_distances = 1 - cosine_similarity(cluster_features)
            cluster_metrics[cluster_id] = {
                'mean_dist': float(np.mean(cluster_distances[np.triu_indices_from(cluster_distances, k=1)])),
                'min_dist': float(np.min(cluster_distances[np.triu_indices_from(cluster_distances, k=1)])),
                'max_dist': float(np.max(cluster_distances[np.triu_indices_from(cluster_distances, k=1)]))
            }
        
        logger.info("Analysis complete")
        logger.info(f"Algorithm mean distance: {algorithm_mean_dist:.4f}")
        logger.info(f"Random mean distance: {np.mean(random_distances):.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        
        return {
            'algorithm_metrics': algorithm_mean_dist,
            'random_metrics': random_distances,
            'p_value': p_value,
            'cluster_metrics': cluster_metrics
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise

def run_analysis(output_dir):
    """Run the complete analysis with Monte Carlo simulation"""
    global _metrics
    
    # Number of Monte Carlo simulations
    n_simulations = 1000
    
    # Load features and clusters if not already loaded
    if _metrics['features'] is None:
        # Load your features from the selected_clusters_global directory
        feature_dir = os.path.join(os.path.dirname(output_dir), 'selected_clusters_global')
        
        # Load features and cluster assignments from distance matrices
        clusters = {}
        features_list = []
        
        # Read all cluster directories
        for cluster_id in range(1, 21):  # Assuming 20 clusters
            cluster_dir = os.path.join(feature_dir, str(cluster_id))
            if not os.path.exists(cluster_dir):
                continue
                
            # Read distance matrix for this cluster
            matrix_path = os.path.join(cluster_dir, 'distance_matrix.csv')
            if os.path.exists(matrix_path):
                with open(matrix_path, 'r') as f:
                    lines = f.readlines()[2:]  # Skip header lines
                    cluster_features = []
                    for line in lines:
                        values = line.strip().split(',')[1:]  # Skip label column
                        features = np.array([float(v) for v in values])
                        cluster_features.append(features)
                    
                    # Store cluster indices
                    start_idx = len(features_list)
                    indices = list(range(start_idx, start_idx + len(cluster_features)))
                    clusters[cluster_id-1] = indices
                    
                    # Add features to main list
                    features_list.extend(cluster_features)
        
        _metrics['features'] = np.array(features_list)
        _metrics['clusters'] = clusters
    
    # Calculate algorithm metrics (mean distance between all selected images)
    distances = 1 - cosine_similarity(_metrics['features'])
    algorithm_mean_dist = np.mean(distances[np.triu_indices_from(distances, k=1)])
    _metrics['algorithm_metrics'] = algorithm_mean_dist
    
    # Run Monte Carlo simulation
    random_distances = []
    n_samples = len(_metrics['features'])
    
    for _ in range(n_simulations):
        # Randomly select same number of images
        random_indices = np.random.choice(n_samples, size=n_samples, replace=False)
        random_features = _metrics['features'][random_indices]
        
        # Calculate mean distance for random selection
        random_sim = cosine_similarity(random_features)
        random_dist = np.mean(1 - random_sim[np.triu_indices_from(random_sim, k=1)])
        random_distances.append(random_dist)
    
    _metrics['random_metrics'] = np.array(random_distances)
    
    # Calculate p-value
    _metrics['p_value'] = np.mean(_metrics['random_metrics'] >= algorithm_mean_dist)
    
    # Return metrics
    return {
        'algorithm_metrics': get_algorithm_metrics(),
        'random_metrics': get_random_metrics(),
        'p_value': get_p_value(),
        'cluster_metrics': get_cluster_metrics()
    }

if __name__ == "__main__":
    # If run directly, use environment variable
    output_dir = os.path.join(os.environ.get('EXPERIMENT_DIR', '.'), 'global_analysis')
    run_analysis(output_dir) 