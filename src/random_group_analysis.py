import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import logging
import os

logger = logging.getLogger(__name__)

def create_random_groups(features, n_groups=20, group_size=5):
    """Create random groups of images"""
    n_images = len(features)
    indices = np.random.permutation(n_images)
    
    groups = {}
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        if end_idx <= len(indices):
            groups[str(i)] = indices[start_idx:end_idx].tolist()
    
    return groups

def analyze_random_groups(features, n_iterations=1000):
    """Analyze random groupings and compare with algorithm clusters"""
    random_metrics = []
    
    logger.info(f"Running {n_iterations} random group analyses...")
    
    for i in range(n_iterations):
        if i % 100 == 0:
            logger.info(f"Iteration {i}/{n_iterations}")
            
        # Create random groups
        random_groups = create_random_groups(features)
        
        # Calculate metrics for each group
        group_metrics = []
        for group_indices in random_groups.values():
            group_features = features[group_indices]
            distances = 1 - cosine_similarity(group_features)
            
            # Get upper triangle of distance matrix (excluding diagonal)
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            
            metrics = {
                'mean_dist': float(np.mean(upper_tri)),
                'min_dist': float(np.min(upper_tri)),
                'max_dist': float(np.max(upper_tri)),
                'std_dist': float(np.std(upper_tri))
            }
            group_metrics.append(metrics)
            
        random_metrics.append(group_metrics)
    
    return random_metrics

def plot_random_vs_algorithm(random_metrics, algorithm_metrics, output_dir):
    """Create comparison plots between random and algorithm groups"""
    # Setup
    plt.style.use('seaborn')
    metrics_dir = os.path.join(output_dir, 'random_analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Mean Distance Distribution
    plt.figure(figsize=(10, 6))
    random_means = [m['mean_dist'] for group in random_metrics for m in group]
    algo_means = [m['mean_dist'] for m in algorithm_metrics]
    
    plt.hist([random_means, algo_means], label=['Random Groups', 'Algorithm Groups'],
             bins=30, alpha=0.6)
    plt.title('Distribution of Mean Distances: Random vs Algorithm')
    plt.xlabel('Mean Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'mean_distance_comparison.png'))
    plt.close()
    
    # 2. Minimum Distance Comparison
    plt.figure(figsize=(10, 6))
    random_mins = [m['min_dist'] for group in random_metrics for m in group]
    algo_mins = [m['min_dist'] for m in algorithm_metrics]
    
    plt.hist([random_mins, algo_mins], label=['Random Groups', 'Algorithm Groups'],
             bins=30, alpha=0.6)
    plt.title('Distribution of Minimum Distances: Random vs Algorithm')
    plt.xlabel('Minimum Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'min_distance_comparison.png'))
    plt.close()
    
    # 3. Temporal Quality Comparison
    plt.figure(figsize=(12, 6))
    
    # Calculate mean distances per position
    random_temporal = np.mean([[m['mean_dist'] for m in group] 
                             for group in random_metrics], axis=0)
    algo_temporal = [m['mean_dist'] for m in algorithm_metrics]
    
    positions = range(len(algo_temporal))
    plt.plot(positions, algo_temporal, 'b-', label='Algorithm Groups')
    plt.plot(positions, random_temporal, 'r--', label='Random Groups (Mean)')
    plt.fill_between(positions, 
                     [np.percentile([group[i]['mean_dist'] for group in random_metrics], 25)
                      for i in range(len(positions))],
                     [np.percentile([group[i]['mean_dist'] for group in random_metrics], 75)
                      for i in range(len(positions))],
                     color='r', alpha=0.2)
    
    plt.title('Temporal Quality: Random vs Algorithm Groups')
    plt.xlabel('Group Position')
    plt.ylabel('Mean Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(metrics_dir, 'temporal_quality_comparison.png'))
    plt.close()
    
    return {
        'random_stats': {
            'mean_dist': float(np.mean(random_means)),
            'std_dist': float(np.std(random_means)),
            'min_dist': float(np.min(random_mins)),
            'max_dist': float(np.max([m['max_dist'] for group in random_metrics for m in group])),
            'quartiles': {
                '25': float(np.percentile(random_means, 25)),
                '50': float(np.percentile(random_means, 50)),
                '75': float(np.percentile(random_means, 75))
            },
            'per_position_stats': {
                'mean': random_temporal.tolist(),
                'q25': [float(np.percentile([group[i]['mean_dist'] for group in random_metrics], 25))
                       for i in range(len(positions))],
                'q75': [float(np.percentile([group[i]['mean_dist'] for group in random_metrics], 75))
                       for i in range(len(positions))]
            }
        },
        'algorithm_stats': {
            'mean_dist': float(np.mean(algo_means)),
            'std_dist': float(np.std(algo_means)),
            'min_dist': float(np.min(algo_mins)),
            'max_dist': float(np.max([m['max_dist'] for m in algorithm_metrics])),
            'temporal_quality': algo_temporal.tolist()
        },
        'comparison': {
            'mean_difference': float(np.mean(algo_means) - np.mean(random_means)),
            'min_difference': float(np.min(algo_mins) - np.min(random_mins)),
            'effect_size': float((np.mean(algo_means) - np.mean(random_means)) / np.std(random_means))
        }
    } 