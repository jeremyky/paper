import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
from typing import List, Dict, Tuple
from config.experiment_config import LOGGING_CONFIG
import logging

# Set up logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ClusteringAlgorithm:
    def __init__(self, features: np.ndarray, config: Dict):
        self.features = features
        self.config = config
        self.n_samples = len(features)
        
    def compute_distance_matrix(self, indices: List[int] = None) -> np.ndarray:
        """Compute cosine distance matrix for given indices or all features"""
        if indices is None:
            features = self.features
        else:
            features = self.features[indices]
        return 1 - cosine_similarity(features)
    
    def validate_cluster(self, indices: List[int]) -> Tuple[float, bool]:
        """Validate cluster meets minimum distance threshold"""
        distances = self.compute_distance_matrix(indices)
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        min_distance = np.min(distances[np.triu_indices_from(distances, k=1)])
        meets_threshold = min_distance >= self.config['parameters']['min_distance_threshold']
        return avg_distance, meets_threshold

class MaxMinSequential(ClusteringAlgorithm):
    def select_images(self) -> List[int]:
        """Select diverse images using sequential max-min approach"""
        num_images = self.config['parameters']['num_images']
        max_iterations = self.config['parameters']['max_iterations']
        
        for iteration in range(max_iterations):
            selected_indices = []
            # Start with random image
            start_idx = random.randrange(self.n_samples)
            selected_indices.append(start_idx)
            
            while len(selected_indices) < num_images:
                # Compute distances to selected images
                remaining_indices = list(set(range(self.n_samples)) - set(selected_indices))
                distances = cosine_similarity(
                    self.features[remaining_indices],
                    self.features[selected_indices]
                )
                min_distances = np.min(distances, axis=1)
                next_idx = remaining_indices[np.argmax(min_distances)]
                selected_indices.append(next_idx)
            
            # Validate selection
            avg_distance, meets_threshold = self.validate_cluster(selected_indices)
            if meets_threshold:
                logger.info(f"Found valid clustering on iteration {iteration + 1}")
                return selected_indices
                
            logger.info(f"Iteration {iteration + 1} failed threshold, retrying...")
            
        logger.warning("Failed to meet threshold after max iterations")
        return selected_indices

class GlobalAverageOptimization(ClusteringAlgorithm):
    def select_images(self) -> List[int]:
        """Select diverse images using global average distance optimization"""
        num_images = self.config['parameters']['num_images']
        sample_size = min(self.config['parameters']['sample_size'], self.n_samples)
        max_iterations = self.config['parameters']['max_iterations']
        
        # Initial random selection
        selected_indices = random.sample(range(self.n_samples), num_images)
        best_avg_distance = np.mean(self.compute_distance_matrix(selected_indices))
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try swapping with random samples
            sample_indices = random.sample(range(self.n_samples), sample_size)
            for idx in sample_indices:
                if idx in selected_indices:
                    continue
                    
                for swap_idx in selected_indices:
                    # Try swap
                    new_indices = selected_indices.copy()
                    new_indices[new_indices.index(swap_idx)] = idx
                    new_avg_distance = np.mean(self.compute_distance_matrix(new_indices))
                    
                    if new_avg_distance > best_avg_distance:
                        selected_indices = new_indices
                        best_avg_distance = new_avg_distance
                        improved = True
                        break
                        
            if not improved:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
                
        return selected_indices

class HybridClustering(ClusteringAlgorithm):
    def select_images(self) -> List[int]:
        """Select diverse images using hybrid approach"""
        # Get initial selection using max-min
        maxmin = MaxMinSequential(self.features, self.config)
        selected_indices = maxmin.select_images()
        
        # Refine using global optimization
        refinement_iterations = self.config['parameters']['refinement_iterations']
        improvement_threshold = self.config['parameters']['improvement_threshold']
        
        current_avg_distance = np.mean(self.compute_distance_matrix(selected_indices))
        
        for iteration in range(refinement_iterations):
            # Try to improve by swapping
            improved = False
            for idx in range(self.n_samples):
                if idx in selected_indices:
                    continue
                    
                for swap_idx in selected_indices:
                    new_indices = selected_indices.copy()
                    new_indices[new_indices.index(swap_idx)] = idx
                    new_avg_distance = np.mean(self.compute_distance_matrix(new_indices))
                    
                    if new_avg_distance > current_avg_distance + improvement_threshold:
                        selected_indices = new_indices
                        current_avg_distance = new_avg_distance
                        improved = True
                        break
                        
            if not improved:
                logger.info(f"Refinement converged after {iteration + 1} iterations")
                break
                
        return selected_indices

def get_clustering_algorithm(algorithm_name: str, features: np.ndarray, config: Dict) -> ClusteringAlgorithm:
    """Factory function to get clustering algorithm instance"""
    algorithms = {
        "maxmin_sequential": MaxMinSequential,
        "global_average": GlobalAverageOptimization,
        "hybrid_clustering": HybridClustering
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    return algorithms[algorithm_name](features, config) 