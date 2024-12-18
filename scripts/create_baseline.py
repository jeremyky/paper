import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_baseline_from_existing():
    """Create baseline from existing clustering results"""
    try:
        # Paths
        clusters_dir = os.path.join(project_root, "src", "selected_clusters_global")
        baseline_dir = os.path.join(project_root, "baseline_clusters")
        os.makedirs(baseline_dir, exist_ok=True)
        
        logger.info(f"Reading clustering data from: {clusters_dir}")
        
        # First read the selected objects file to get image names
        selected_objects_file = os.path.join(clusters_dir, "selected_unique_objects.txt")
        with open(selected_objects_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        logger.info(f"Found {len(image_names)} selected images")
        
        # Initialize data structures
        clusters = {}
        current_idx = 0
        
        # Read cluster assignments
        for cluster_id in range(1, 21):  # 20 clusters
            cluster_dir = os.path.join(clusters_dir, str(cluster_id))
            if not os.path.exists(cluster_dir):
                logger.warning(f"Cluster directory {cluster_id} not found")
                continue
            
            # Get images in this cluster
            cluster_images = [img for img in os.listdir(cluster_dir) 
                            if img.endswith(('.jpg', '.jpeg', '.png', '.thl'))]
            
            if cluster_images:
                # Store indices for this cluster
                indices = list(range(current_idx, current_idx + len(cluster_images)))
                clusters[str(cluster_id-1)] = indices
                current_idx += len(cluster_images)
                logger.info(f"Cluster {cluster_id}: {len(cluster_images)} images")
        
        # Create dummy features for now (we'll use real features later)
        n_total_images = sum(len(indices) for indices in clusters.values())
        features = np.random.rand(n_total_images, 512)  # 512-dimensional features
        
        # Calculate initial metrics
        distances = 1 - cosine_similarity(features)
        algorithm_mean_dist = np.mean(distances[np.triu_indices_from(distances, k=1)])
        
        # Quick Monte Carlo simulation
        n_simulations = 100
        random_distances = []
        
        for _ in range(n_simulations):
            random_indices = np.random.choice(n_total_images, size=n_total_images, replace=False)
            random_features = features[random_indices]
            random_sim = cosine_similarity(random_features)
            random_dist = np.mean(1 - random_sim[np.triu_indices_from(random_sim, k=1)])
            random_distances.append(random_dist)
        
        # Save baseline
        baseline_data = {
            'clusters': clusters,
            'image_names': image_names,
            'features': features.tolist(),
            'algorithm_metrics': float(algorithm_mean_dist),
            'random_metrics': random_distances,
            'p_value': float(np.mean(random_distances >= algorithm_mean_dist))
        }
        
        baseline_path = os.path.join(baseline_dir, 'baseline_clusters_v1.json')
        with open(baseline_path, 'w') as f:
            json.dump(baseline_data, f)
        
        logger.info(f"Successfully created baseline at: {baseline_path}")
        logger.info(f"Number of clusters: {len(clusters)}")
        logger.info(f"Total images: {n_total_images}")
        
    except Exception as e:
        logger.error(f"Error creating baseline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    create_baseline_from_existing()