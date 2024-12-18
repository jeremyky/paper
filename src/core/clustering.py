import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Any
import random

logger = logging.getLogger(__name__)

def perform_clustering(features: np.ndarray,
                     algorithm: str,
                     num_images: int,
                     images_per_cluster: int,
                     params: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    Perform clustering using specified algorithm
    
    Args:
        features: Feature vectors for all images
        algorithm: Clustering algorithm to use
        num_images: Number of images to select
        images_per_cluster: Number of images per cluster
        params: Algorithm-specific parameters
    
    Returns:
        Dictionary mapping cluster IDs to lists of image indices
    """
    logger.info(f"Starting clustering with {algorithm}")
    
    if algorithm == "maxmin_sequential":
        selected_indices = maxmin_sequential(
            features,
            num_images,
            params['min_distance_threshold'],
            params['max_iterations']
        )
    elif algorithm == "global_average":
        selected_indices = global_average_optimization(
            features,
            num_images,
            params['improvement_threshold'],
            params['max_iterations'],
            params['sample_size']
        )
    elif algorithm == "hybrid_clustering":
        selected_indices = hybrid_clustering(
            features,
            num_images,
            params['min_distance_threshold'],
            params['refinement_iterations'],
            params['improvement_threshold']
        )
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    # Group selected images into clusters
    num_clusters = num_images // images_per_cluster
    clusters = {}
    
    for i in range(num_clusters):
        start_idx = i * images_per_cluster
        end_idx = start_idx + images_per_cluster
        clusters[i] = selected_indices[start_idx:end_idx]
    
    return clusters

def maxmin_sequential(features: np.ndarray,
                     num_select: int,
                     min_distance_threshold: float,
                     max_iterations: int) -> List[int]:
    """Max-min sequential selection algorithm"""
    n_samples = len(features)
    
    for iteration in range(max_iterations):
        selected_indices = []
        remaining_indices = set(range(n_samples))
        
        # Start with random image
        start_idx = random.choice(list(remaining_indices))
        selected_indices.append(start_idx)
        remaining_indices.remove(start_idx)
        
        while len(selected_indices) < num_select:
            # Compute distances to selected images
            selected_features = features[selected_indices]
            remaining_features = features[list(remaining_indices)]
            similarities = cosine_similarity(remaining_features, selected_features)
            min_similarities = similarities.min(axis=1)
            
            # Select image with minimum similarity to selected set
            next_idx = list(remaining_indices)[np.argmin(min_similarities)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Validate selection
        similarities = cosine_similarity(features[selected_indices])
        min_distance = 1 - np.max(similarities - np.eye(len(similarities)))
        
        if min_distance >= min_distance_threshold:
            logger.info(f"Found valid selection on iteration {iteration + 1}")
            return selected_indices
        
        logger.info(f"Iteration {iteration + 1} failed threshold, retrying...")
    
    logger.warning("Failed to meet threshold after max iterations")
    return selected_indices

def global_average_optimization(features: np.ndarray,
                              num_select: int,
                              improvement_threshold: float,
                              max_iterations: int,
                              sample_size: int) -> List[int]:
    """Global average distance optimization"""
    n_samples = len(features)
    sample_size = min(sample_size, n_samples)
    
    # Initial random selection
    selected_indices = random.sample(range(n_samples), num_select)
    best_avg_distance = np.mean(1 - cosine_similarity(features[selected_indices]))
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try swapping with random samples
        sample_indices = random.sample(range(n_samples), sample_size)
        for idx in sample_indices:
            if idx in selected_indices:
                continue
            
            for swap_idx in selected_indices:
                # Try swap
                new_indices = selected_indices.copy()
                new_indices[new_indices.index(swap_idx)] = idx
                new_avg_distance = np.mean(1 - cosine_similarity(features[new_indices]))
                
                if new_avg_distance > best_avg_distance + improvement_threshold:
                    selected_indices = new_indices
                    best_avg_distance = new_avg_distance
                    improved = True
                    break
        
        if not improved:
            logger.info(f"Converged after {iteration + 1} iterations")
            break
    
    return selected_indices

def hybrid_clustering(features: np.ndarray,
                     num_select: int,
                     min_distance_threshold: float,
                     refinement_iterations: int,
                     improvement_threshold: float) -> List[int]:
    """Hybrid approach combining max-min with global optimization"""
    # Initial selection using max-min
    selected_indices = maxmin_sequential(
        features,
        num_select,
        min_distance_threshold,
        max_iterations=1
    )
    
    # Refine using global optimization
    current_avg_distance = np.mean(1 - cosine_similarity(features[selected_indices]))
    
    for iteration in range(refinement_iterations):
        improved = False
        
        for idx in range(len(features)):
            if idx in selected_indices:
                continue
            
            for swap_idx in selected_indices:
                new_indices = selected_indices.copy()
                new_indices[new_indices.index(swap_idx)] = idx
                new_avg_distance = np.mean(1 - cosine_similarity(features[new_indices]))
                
                if new_avg_distance > current_avg_distance + improvement_threshold:
                    selected_indices = new_indices
                    current_avg_distance = new_avg_distance
                    improved = True
                    break
        
        if not improved:
            logger.info(f"Refinement converged after {iteration + 1} iterations")
            break
    
    return selected_indices
