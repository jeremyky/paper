import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging

logger = logging.getLogger(__name__)

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
                
            random_indices = np.random.choice(n_samples, size=n_samples, replace=False)
            random_features = features[random_indices]
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
    """Run the complete analysis"""
    # Your existing analysis code here
    pass

if __name__ == "__main__":
    output_dir = os.path.join(os.environ.get('EXPERIMENT_DIR', '.'), 'global_analysis')
    run_analysis(output_dir) 