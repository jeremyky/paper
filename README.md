# Remote Viewing Experiment Analysis

This repository contains tools for analyzing and testing image clustering algorithms for remote viewing experiments. The system uses deep learning and semantic analysis to find and group similar images, validating the results through statistical testing.

## Overview

### Feature Extraction Methods
The pipeline uses multiple approaches to understand image content:

1. **CNN Features** (ResNet-50)
   - Deep convolutional neural network pre-trained on ImageNet
   - Extracts visual features like shapes, textures, and patterns
   - Final layer features (2048-dimensional vectors)

2. **SBERT Semantic Features** (Planned)
   - Sentence-BERT model for image caption analysis
   - Will capture semantic meaning and context
   - Uses CLIP model to generate image descriptions
   - 768-dimensional semantic embeddings

### Image Selection Process

1. **Initial Selection (110 Images)**
   - Selects slightly more images than needed
   - Uses distance metrics to maximize diversity
   - Allows for removal of non-interpretable images

2. **Interpretability Filtering**
   - Removes ~10 non-interpretable images
   - Criteria:
     - Image clarity
     - Object recognizability
     - Visual distinctiveness

3. **Final Selection (100 Images)**
   - Maintains maximum diversity
   - Ensures all images are interpretable
   - Balances diversity with quality

### Selection Algorithms

1. **MaxMin Sequential**
   ```python
   CLUSTERING_ALGORITHM = "maxmin_sequential"
   SELECTION_CONFIG = {
       "initial_selection": 110,
       "final_selection": 100,
       "interpretability_threshold": 0.8,
       "distance_metric": "cosine"
   }
   ```
   - Greedy selection maximizing minimum distance
   - Sequential selection with diversity optimization
   - Validates against random selection

2. **Global Average**
   - Optimizes overall distribution
   - Computationally more intensive
   - Better for final refinement

3. **Hybrid Clustering**
   - Combines MaxMin and Global approaches
   - Default recommended option
   - Best balance of speed and quality

### Clustering Methods

1. **Agglomerative Clustering**
   - Creates 20 clusters of 5 images each
   - Maximizes within-cluster diversity
   - Parameters:
   ```python
   CLUSTERING_CONFIG = {
       "n_clusters": 20,
       "linkage": "ward",
       "distance_threshold": None,
       "diversity_weight": 0.7  # Weights diversity vs. coherence
   }
   ```

## Validation Metrics

### Distance Analysis
- Minimum distances between images (algorithm vs. random)
- Mean distances within clusters
- Distribution of distances across all pairs
- Sequential vs. random performance comparison

### Cluster Quality Metrics
- Within-cluster diversity scores
- Between-cluster separation
- Temporal analysis (early vs. late clusters)
- Cluster stability measures

### Statistical Validation
```
=== Analysis Metrics Summary ===

Monte Carlo Simulation Results:
--------------------------------------------------
Number of Simulations: 1000
Algorithm Performance:
  Mean Distance: 0.7523
  Min Distance: 0.5234
  Max Distance: 0.8901
  
Random Selection Stats:
  Mean: 0.6891 ± 0.0234
  Min: 0.6234
  Max: 0.7456
  
Significance:
  P-value: 0.001
  Effect Size: 0.842
```

### Visualization Outputs
1. **Distance Distribution Plots**
   - Algorithm vs. random selection
   - Within-cluster distances
   - Between-cluster distances

2. **Cluster Quality Plots**
   - Mean distances per cluster
   - Temporal cluster quality
   - Diversity distribution

3. **Validation Plots**
   - Monte Carlo simulation results
   - Statistical significance
   - Performance benchmarks

## Pipeline Workflow

1. **Feature Extraction**
   ```python
   # Dataset Configuration
   DATASET = "mm_unique"  # or "cifar100" for validation
   MAX_IMAGES = 2400     # Maximum images to process
   ```

2. **Image Selection**
   - Selects diverse representative images
   - Uses configured selection algorithm
   - Number of images controlled by:
   ```python
   NUM_IMAGES = 100      # Total images to select
   IMAGES_PER_CLUSTER = 5  # Images per group
   ```

3. **Clustering**
   - Groups similar images
   - Generates visualizations:
     - Dendrograms
     - Distance matrices
     - Cluster visualizations

4. **Statistical Validation**
   - Monte Carlo simulations
   - Compares against random selection
   - Generates performance metrics

## Understanding Results

### Feature Analysis
- `distance_matrix.csv`: Pairwise distances between images
- Higher distances indicate more dissimilar images
- Both CNN and semantic features contribute

### Clustering Quality
- Dendrogram shows hierarchical relationships
- Distance matrices show internal cluster cohesion
- Lower internal distances indicate better grouping

### Statistical Validation
The Monte Carlo simulation tests if the algorithm performs better than random:

```
=== Analysis Metrics Summary ===

Monte Carlo Simulation Results:
--------------------------------------------------
Number of Simulations: 1000
Algorithm Mean Distance: 0.7523
Random Selection Stats:
  Mean: 0.6891
  Std: 0.0234
  Min: 0.6234
  Max: 0.7456
P-value: 0.001
```

- Lower p-value indicates significant improvement over random
- Compare algorithm mean distance to random distribution
- Check cluster-specific metrics for consistency

## Configuration Options

Full configuration available in `src/config/experiment_config.py`:

```python
# Dataset selection
DATASET = "mm_unique"
MAX_IMAGES = 2400

# Algorithm selection
CLUSTERING_ALGORITHM = "hybrid_clustering"

# Clustering parameters
NUM_IMAGES = 100
IMAGES_PER_CLUSTER = 5

# Pipeline control
PIPELINE_CONFIG = {
    "start_from": "features",
    "save_baseline": False,
    "use_baseline": False,
    "baseline_dir": "baseline_clusters",
    "baseline_name": "baseline_clusters_v1"
}
```

## Directory Structure

```
remote-viewing-experiment/
├── src/
│   ├── config/                 # Configuration files
│   ├── clustering/             # Clustering algorithms
│   └── unique_objects_features/# Extracted image features
├── experiments/                # Experiment results
│   └── experiment_N_TIMESTAMP/ # Individual experiment outputs
│       ├── selected_clusters/
│       │   ├── selected_clusters_global/  # Cluster results
│       │   │   ├── 1/                    # Cluster directories
│       │   │   │   ├── distance_matrix.csv
│       │   │   │   ├── dendrogram.png
│       │   │   │   └── images/
│       │   │   └── ...
│       │   └── metrics/                  # Analysis results
│       │       ├── analysis_summary/
│       │       │   ├── metrics_summary.txt
│       │       │   ├── monte_carlo_distribution.png
│       │       │   ├── cluster_performance.png
│       │       │   └── raw_metrics.json
│       └── experiment_info.txt
└── baseline_clusters/          # Saved baseline clusterings
```

## Running Experiments

### Full Pipeline Run
```bash
# Edit config to start from features
python scripts/run_analysis.py
```

### Save Baseline Clustering
```python
# In experiment_config.py:
PIPELINE_CONFIG = {
    "start_from": "features",
    "save_baseline": True,
    "baseline_name": "baseline_v1"
}
```

### Run Analysis on Existing Clustering
```python
# In experiment_config.py:
PIPELINE_CONFIG = {
    "start_from": "analysis",
    "use_baseline": True,
    "baseline_name": "baseline_v1"
}
```

## Understanding Results

### Metrics Summary
The `metrics_summary.txt` file contains:
- Monte Carlo simulation results
  - Algorithm mean distance: Average dissimilarity between selected images
  - Random selection stats: Baseline performance metrics
  - P-value: Statistical significance of results
- Per-cluster statistics
  - Mean/Min/Max internal distances: Measure of cluster cohesion

### Visualizations

1. **Monte Carlo Distribution Plot**
   - Histogram of random selection performances
   - Red line shows algorithm performance
   - Lower p-value indicates better algorithm performance

2. **Cluster Performance Plot**
   - Shows mean internal distance for each cluster
   - Lower values indicate more cohesive clusters
   - Helps identify problematic clusters

### Raw Metrics
The `raw_metrics.json` file contains detailed numerical data for:
- Monte Carlo simulation results
- Per-cluster metrics
- Algorithm performance statistics

## Interpreting Results

### Statistical Significance
- P-value < 0.05: Algorithm significantly outperforms random selection
- Lower mean distances indicate more similar images in clusters
- Higher internal cluster cohesion suggests better grouping

### Cluster Quality
- Mean internal distance < random mean: Good cluster cohesion
- Consistent performance across clusters: Stable algorithm
- Outlier clusters may need investigation

## Troubleshooting

Common issues and solutions:
1. Missing baseline: Run with `save_baseline=True` first
2. Experiment directory not created: Check write permissions
3. Feature extraction fails: Verify image directory structure

