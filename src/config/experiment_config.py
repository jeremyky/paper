"""
Remote Viewing Experiment Configuration

This file contains all configurable parameters for the experiment.
Edit the values below to modify the experiment behavior.
"""

#------------------------------------------------------------------------------
# Pipeline Configuration
#------------------------------------------------------------------------------

# Pipeline control
PIPELINE_CONFIG = {
    # Start from 'analysis' to skip feature extraction and clustering
    "start_from": "analysis",
    
    # Use existing baseline clustering
    "use_baseline": True,
    
    # Specify which baseline to use
    "baseline_dir": "baseline_clusters",
    "baseline_name": "baseline_clusters_v1"  # Point to your saved clustering
}

#------------------------------------------------------------------------------
# Dataset Configuration
#------------------------------------------------------------------------------

# Which dataset to use for the experiment
# Options: 
#   - "mm_unique": Massive Memory Unique Objects dataset
#   - "cifar100": CIFAR-100 dataset for validation
DATASET = "mm_unique"

# Maximum number of images to process
# - Set to None to process all images
# - Recommended: 2400 for balanced sampling
MAX_IMAGES = 2400

# Input/Output paths
PATHS = {
    "mm_unique": {
        "input": "data/raw/mm_unique_objects",
        "features": "data/features/mm_unique",
        "results": "data/results/mm_unique"
    },
    "cifar100": {
        "input": "data/raw/cifar100",
        "features": "data/features/cifar100",
        "results": "data/results/cifar100"
    }
}

# Set default dataset
DEFAULT_DATASET = "mm_unique"

#------------------------------------------------------------------------------
# Clustering Configuration
#------------------------------------------------------------------------------

# Clustering algorithm to use
# Options:
#   - "maxmin_sequential": Fast, sequential selection (original algorithm)
#   - "global_average": Global optimization (more thorough but slower)
#   - "hybrid_clustering": Combines both approaches
CLUSTERING_ALGORITHM = "hybrid_clustering"

# Number of images to select
# - Must be divisible by IMAGES_PER_CLUSTER
# - Default: 100 for 20 clusters of 5 images
NUM_IMAGES = 100

# Number of images per cluster
# - Affects internal cluster diversity
# - Default: 5 images per cluster
IMAGES_PER_CLUSTER = 5

# Algorithm-specific parameters
CLUSTERING_PARAMS = {
    "maxmin_sequential": {
        "min_distance_threshold": 0.4,  # Minimum acceptable distance between images
        "max_iterations": 5  # Maximum attempts to find diverse selection
    },
    "global_average": {
        "improvement_threshold": 0.01,  # Minimum improvement to continue optimization
        "max_iterations": 100,  # Maximum optimization iterations
        "sample_size": 1000  # Number of random images to sample for comparison
    },
    "hybrid_clustering": {
        "min_distance_threshold": 0.4,
        "refinement_iterations": 20,  # Number of refinement steps
        "improvement_threshold": 0.005  # Minimum improvement during refinement
    }
}

#------------------------------------------------------------------------------
# Validation Configuration
#------------------------------------------------------------------------------

# Number of Monte Carlo iterations for validation
# - Higher numbers give more reliable statistics
# - Recommended: 1000 for publication-quality results
MONTE_CARLO_ITERATIONS = 1000

# Statistical significance threshold
# - Standard p-value threshold
# - Default: 0.05 (95% confidence)
SIGNIFICANCE_THRESHOLD = 0.05

# Minimum improvement threshold over random
# - How much better algorithm must be vs random
# - Default: 0.1 (10% improvement)
MIN_IMPROVEMENT_THRESHOLD = 0.1

#------------------------------------------------------------------------------
# Visualization Configuration
#------------------------------------------------------------------------------

# Figure sizes for different plot types
FIGURE_SIZES = {
    "similarity_matrix": (12, 10),
    "cluster_trends": (12, 6),
    "distributions": (10, 6)
}

# Color scheme for visualizations
VISUALIZATION_PARAMS = {
    "similarity_colormap": "coolwarm",  # Colormap for similarity matrices
    "cluster_colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],  # Colors for cluster plots
    "error_alpha": 0.2  # Transparency for error regions
}

#------------------------------------------------------------------------------
# Logging Configuration
#------------------------------------------------------------------------------

# Logging level
# Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Whether to save logs to file
SAVE_LOGS = True

# Log file format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 