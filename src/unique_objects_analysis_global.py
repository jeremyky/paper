import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

logger = logging.getLogger(__name__)

def cohen_d(x, y):
    """Calculate Cohen's d effect size"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    
    # Pool standard deviation
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    # Calculate effect size
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

def calculate_distance_metrics(features):
    """Calculate comprehensive distance metrics for a set of features"""
    distances = 1 - cosine_similarity(features)
    
    # Get unique distances (upper triangle only)
    upper_tri = distances[np.triu_indices_from(distances, k=1)]
    
    return {
        'mean_dist': float(np.mean(upper_tri)),
        'min_dist': float(np.min(upper_tri)),
        'max_dist': float(np.max(upper_tri)),
        'std_dist': float(np.std(upper_tri)),
        'median_dist': float(np.median(upper_tri)),
        'quartiles': [float(q) for q in np.percentile(upper_tri, [25, 50, 75])]
    }

def analyze_cluster_temporal_quality(features, clusters):
    """Analyze how cluster quality varies over time"""
    cluster_metrics = []
    
    # Sort clusters by order of creation
    sorted_clusters = sorted(clusters.items(), key=lambda x: int(x[0]))
    
    for cluster_id, indices in sorted_clusters:
        cluster_features = features[indices]
        metrics = calculate_distance_metrics(cluster_features)
        metrics['cluster_id'] = cluster_id
        metrics['size'] = len(indices)
        cluster_metrics.append(metrics)
    
    return cluster_metrics

def plot_cluster_temporal_analysis(cluster_metrics, output_dir):
    """Create visualizations for temporal cluster analysis"""
    metrics_dir = os.path.join(output_dir, 'cluster_analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot mean distances over time
    plt.figure(figsize=(12, 6))
    cluster_ids = [m['cluster_id'] for m in cluster_metrics]
    mean_dists = [m['mean_dist'] for m in cluster_metrics]
    
    plt.plot(cluster_ids, mean_dists, marker='o')
    plt.axhline(y=np.mean(mean_dists), color='r', linestyle='--', 
                label=f'Average: {np.mean(mean_dists):.3f}')
    plt.fill_between(cluster_ids, 
                     [m['mean_dist'] - m['std_dist'] for m in cluster_metrics],
                     [m['mean_dist'] + m['std_dist'] for m in cluster_metrics],
                     alpha=0.2)
    
    plt.title('Cluster Quality Over Time')
    plt.xlabel('Cluster ID (Temporal Order)')
    plt.ylabel('Mean Internal Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(metrics_dir, 'temporal_quality.png'))
    plt.close()

def find_closest_pairs(features, n_pairs=5):
    """Find the n closest pairs of images"""
    distances = 1 - cosine_similarity(features)
    
    # Get indices of upper triangle (excluding diagonal)
    i_upper, j_upper = np.triu_indices_from(distances, k=1)
    
    # Get distances and corresponding indices
    dist_pairs = list(zip(distances[i_upper, j_upper], i_upper, j_upper))
    
    # Sort by distance
    dist_pairs.sort(key=lambda x: x[0])
    
    return dist_pairs[:n_pairs]

def analyze_minimum_distances(features, clusters, image_names=None):
    """Analyze minimum distances between images globally and within clusters"""
    results = {
        'global': {
            'closest_pairs': [],
            'stats': {}
        },
        'clusters': {}
    }
    
    # Global analysis
    closest_pairs = find_closest_pairs(features)
    
    for dist, i, j in closest_pairs:
        pair_info = {
            'distance': float(dist),
            'index_1': int(i),
            'index_2': int(j)
        }
        if image_names:
            pair_info.update({
                'image_1': image_names[i],
                'image_2': image_names[j]
            })
        results['global']['closest_pairs'].append(pair_info)
    
    # Per-cluster analysis
    for cluster_id, indices in clusters.items():
        cluster_features = features[indices]
        cluster_pairs = find_closest_pairs(cluster_features, n_pairs=3)
        
        cluster_results = {
            'closest_pairs': [],
            'stats': calculate_distance_metrics(cluster_features)
        }
        
        for dist, i, j in cluster_pairs:
            pair_info = {
                'distance': float(dist),
                'index_1': int(indices[i]),
                'index_2': int(indices[j])
            }
            if image_names:
                pair_info.update({
                    'image_1': image_names[indices[i]],
                    'image_2': image_names[indices[j]]
                })
            cluster_results['closest_pairs'].append(pair_info)
            
        results['clusters'][cluster_id] = cluster_results
    
    return results

def plot_minimum_distances(min_dist_analysis, output_dir):
    """Create visualizations for minimum distance analysis"""
    metrics_dir = os.path.join(output_dir, 'distance_analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Plot global minimum distances
    global_dists = [p['distance'] for p in min_dist_analysis['global']['closest_pairs']]
    cluster_min_dists = []
    
    for cluster_id, cluster_data in min_dist_analysis['clusters'].items():
        cluster_min_dists.extend([p['distance'] for p in cluster_data['closest_pairs']])
    
    plt.figure(figsize=(10, 6))
    plt.hist([global_dists, cluster_min_dists], label=['Global', 'Within Clusters'],
             bins=20, alpha=0.6)
    plt.title('Distribution of Minimum Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(metrics_dir, 'minimum_distances.png'))
    plt.close()

def run_analysis_from_baseline(baseline_path, output_dir):
    """Run enhanced analysis using saved baseline clustering"""
    logger.info(f"Loading baseline from: {baseline_path}")
    
    try:
        # Load baseline data
        with open(baseline_path + '.json', 'r') as f:
            baseline_data = json.load(f)
        
        # Extract data
        features = np.array(baseline_data['features'])
        clusters = baseline_data['clusters']
        
        # Calculate algorithm metrics
        algorithm_metrics = calculate_distance_metrics(features)
        
        # Run Monte Carlo simulation
        n_simulations = 1000
        random_metrics_list = []
        n_samples = len(features)
        
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        for i in range(n_simulations):
            if i % 100 == 0:
                logger.info(f"Simulation progress: {i}/{n_simulations}")
            
            random_indices = np.random.choice(n_samples, size=n_samples, replace=False)
            random_features = features[random_indices]
            random_metrics_list.append(calculate_distance_metrics(random_features))
        
        # Aggregate random metrics
        random_means = [m['mean_dist'] for m in random_metrics_list]
        random_mins = [m['min_dist'] for m in random_metrics_list]
        
        # Calculate statistical significance
        t_stat, p_value = ttest_ind([algorithm_metrics['mean_dist']], random_means)
        effect_size = cohen_d([algorithm_metrics['mean_dist']], random_means)
        
        # Analyze temporal cluster quality
        cluster_temporal_metrics = analyze_cluster_temporal_quality(features, clusters)
        plot_cluster_temporal_analysis(cluster_temporal_metrics, output_dir)
        
        # Add minimum distance analysis
        min_dist_analysis = analyze_minimum_distances(
            features, 
            clusters,
            image_names=baseline_data.get('image_names')
        )
        plot_minimum_distances(min_dist_analysis, output_dir)
        
        # Prepare comprehensive results
        results = {
            'algorithm_metrics': algorithm_metrics,
            'random_metrics': {
                'mean_dist': float(np.mean(random_means)),
                'std_dist': float(np.std(random_means)),
                'min_dist': float(np.min(random_mins)),
                'max_dist': float(np.max([m['max_dist'] for m in random_metrics_list])),
                'all_means': random_means  # For distribution plotting
            },
            'statistical_tests': {
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'effect_size': float(effect_size)
            },
            'cluster_metrics': {
                'temporal': cluster_temporal_metrics,
                'overall': calculate_distance_metrics(features)
            },
            'minimum_distances': min_dist_analysis
        }
        
        # Log key results
        logger.info("Analysis complete")
        logger.info(f"Algorithm mean distance: {algorithm_metrics['mean_dist']:.4f}")
        logger.info(f"Algorithm min distance: {algorithm_metrics['min_dist']:.4f}")
        logger.info(f"Random mean distance: {results['random_metrics']['mean_dist']:.4f}")
        logger.info(f"P-value: {p_value:.4f}")
        logger.info(f"Effect size: {effect_size:.4f}")
        
        # Log minimum distance findings
        logger.info("\nClosest image pairs globally:")
        for pair in min_dist_analysis['global']['closest_pairs'][:3]:
            if 'image_1' in pair:
                logger.info(f"Distance: {pair['distance']:.4f} between {pair['image_1']} and {pair['image_2']}")
            else:
                logger.info(f"Distance: {pair['distance']:.4f} between indices {pair['index_1']} and {pair['index_2']}")
        
        return results
        
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