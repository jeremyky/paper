import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from matplotlib.colors import LinearSegmentedColormap

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
        # Only add image names if they exist and indices are valid
        if image_names is not None and i < len(image_names) and j < len(image_names):
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
            # Only add image names if they exist and indices are valid
            if image_names is not None and indices[i] < len(image_names) and indices[j] < len(image_names):
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

def plot_distance_matrix(features, output_dir, max_display=100):
    """Plot distance matrix heatmap for features"""
    metrics_dir = os.path.join(output_dir, 'distance_analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # If too many features, sample randomly
    if len(features) > max_display:
        indices = np.random.choice(len(features), max_display, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features
    
    # Calculate distance matrix
    distances = 1 - cosine_similarity(features_subset)
    
    # Create custom colormap
    colors = ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84', '#FC8D59', 
              '#EF6548', '#D7301F', '#B30000', '#7F0000']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(distances, 
                cmap=cmap,
                xticklabels=False, 
                yticklabels=False)
    plt.title('Distance Matrix Heatmap\n(Cosine Distance)')
    plt.xlabel('Images')
    plt.ylabel('Images')
    
    # Add colorbar label
    plt.colorbar(label='Distance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'distance_matrix.png'), dpi=300)
    plt.close()

def plot_cluster_analysis(clusters, features, output_dir):
    """Create comprehensive cluster analysis visualizations"""
    metrics_dir = os.path.join(output_dir, 'cluster_analysis')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Cluster Size Distribution
    plt.figure(figsize=(10, 6))
    sizes = [len(indices) for indices in clusters.values()]
    sns.histplot(sizes, bins=range(min(sizes), max(sizes) + 2, 1))
    plt.title('Cluster Size Distribution')
    plt.xlabel('Number of Images in Cluster')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(metrics_dir, 'cluster_sizes.png'))
    plt.close()
    
    # 2. Inter-cluster Distances
    centroids = []
    cluster_ids = []
    for cluster_id, indices in clusters.items():
        centroid = np.mean(features[indices], axis=0)
        centroids.append(centroid)
        cluster_ids.append(cluster_id)
    
    centroid_distances = 1 - cosine_similarity(centroids)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(centroid_distances,
                cmap='viridis',
                xticklabels=cluster_ids,
                yticklabels=cluster_ids,
                annot=True,
                fmt='.2f',
                square=True)
    plt.title('Inter-cluster Distance Matrix\n(Cosine Distance between Centroids)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'intercluster_distances.png'))
    plt.close()
    
    # 3. Intra-cluster Distance Distribution
    intra_distances = []
    cluster_labels = []
    
    for cluster_id, indices in clusters.items():
        if len(indices) > 1:  # Need at least 2 images for distances
            cluster_features = features[indices]
            distances = 1 - cosine_similarity(cluster_features)
            # Get upper triangle values
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_distances.extend(upper_tri)
            cluster_labels.extend([cluster_id] * len(upper_tri))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=cluster_labels, y=intra_distances)
    plt.title('Intra-cluster Distance Distribution')
    plt.xlabel('Cluster ID')
    plt.ylabel('Cosine Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, 'intracluster_distances.png'))
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
        
        # Add new visualizations
        logger.info("Generating distance matrix visualization...")
        plot_distance_matrix(features, output_dir)
        
        logger.info("Generating cluster analysis visualizations...")
        plot_cluster_analysis(clusters, features, output_dir)
        
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