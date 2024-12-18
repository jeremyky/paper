import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import unique_objects_analysis_global as analysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create synthetic test data"""
    n_images = 100
    n_features = 512
    n_clusters = 20
    images_per_cluster = 5
    
    # Create synthetic features
    features = np.random.rand(n_images, n_features)
    
    # Create synthetic clusters
    clusters = {}
    for i in range(n_clusters):
        start_idx = i * images_per_cluster
        end_idx = start_idx + images_per_cluster
        clusters[str(i)] = list(range(start_idx, end_idx))
    
    # Create synthetic image names
    image_names = [f"test_image_{i}.jpg" for i in range(n_images)]
    
    return features, clusters, image_names

def test_visualizations():
    """Test all visualization functions"""
    logger.info("Creating test data...")
    features, clusters, image_names = create_test_data()
    
    # Create test output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(project_root, "test_outputs", f"test_run_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    
    logger.info("Testing cluster temporal analysis...")
    # Test cluster temporal analysis
    cluster_metrics = analysis.analyze_cluster_temporal_quality(features, clusters)
    analysis.plot_cluster_temporal_analysis(cluster_metrics, test_dir)
    
    logger.info("Testing minimum distance analysis...")
    # Test minimum distance analysis
    min_dist_analysis = analysis.analyze_minimum_distances(features, clusters, image_names)
    analysis.plot_minimum_distances(min_dist_analysis, test_dir)
    
    # Print some test metrics
    logger.info("\nTest Results:")
    logger.info("Cluster Temporal Analysis:")
    for i, metrics in enumerate(cluster_metrics[:3]):  # Show first 3 clusters
        logger.info(f"Cluster {i}:")
        logger.info(f"  Mean Distance: {metrics['mean_dist']:.4f}")
        logger.info(f"  Min Distance: {metrics['min_dist']:.4f}")
        logger.info(f"  Max Distance: {metrics['max_dist']:.4f}")
    
    logger.info("\nMinimum Distance Analysis:")
    logger.info("Global closest pairs:")
    for pair in min_dist_analysis['global']['closest_pairs'][:3]:
        logger.info(f"Distance: {pair['distance']:.4f} between {pair['image_1']} and {pair['image_2']}")
    
    logger.info(f"\nResults saved in: {test_dir}")
    return test_dir

def test_full_analysis():
    """Test the complete analysis pipeline"""
    logger.info("Creating test baseline data...")
    features, clusters, image_names = create_test_data()
    
    # Create test baseline file
    baseline_dir = os.path.join(project_root, "test_outputs", "baseline")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_path = os.path.join(baseline_dir, "test_baseline")
    
    baseline_data = {
        'features': features.tolist(),
        'clusters': clusters,
        'image_names': image_names
    }
    
    with open(baseline_path + '.json', 'w') as f:
        import json
        json.dump(baseline_data, f)
    
    logger.info("Running full analysis...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "test_outputs", f"full_analysis_{timestamp}")
    
    results = analysis.run_analysis_from_baseline(baseline_path, output_dir)
    
    logger.info("\nFull Analysis Results:")
    logger.info(f"Algorithm mean distance: {results['algorithm_metrics']['mean_dist']:.4f}")
    logger.info(f"Random mean distance: {results['random_metrics']['mean_dist']:.4f}")
    logger.info(f"P-value: {results['statistical_tests']['p_value']:.4f}")
    
    logger.info(f"\nResults saved in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    logger.info("Starting visualization tests...")
    viz_dir = test_visualizations()
    
    logger.info("\nStarting full analysis test...")
    analysis_dir = test_full_analysis()
    
    logger.info("\nAll tests complete!")
    logger.info(f"Visualization test results: {viz_dir}")
    logger.info(f"Full analysis results: {analysis_dir}") 