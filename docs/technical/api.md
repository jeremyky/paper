# API Reference

## Core Analysis Functions

### Feature Analysis

```python
def calculate_distance_metrics(features: np.ndarray) -> dict:
    """
    Calculate comprehensive distance metrics for a set of features.
    
    Args:
        features (np.ndarray): Feature vectors of shape (n_images, n_features)
    
    Returns:
        dict: Metrics including mean_dist, min_dist, max_dist, etc.
    """
```

### Cluster Analysis

```python
def analyze_cluster_temporal_quality(
    features: np.ndarray,
    clusters: dict
) -> list:
    """
    Analyze how cluster quality varies over time.
    
    Args:
        features: Feature vectors
        clusters: Dictionary mapping cluster_id to image indices
    
    Returns:
        list: Metrics for each cluster
    """
```

### Visualization Functions

```python
def plot_distance_matrix(
    features: np.ndarray,
    output_dir: str,
    max_display: int = 100
):
    """
    Plot distance matrix heatmap for features.
    
    Args:
        features: Feature vectors
        output_dir: Directory to save plot
        max_display: Maximum number of images to display
    """
```

## Complete Pipeline

```python
def run_analysis_from_baseline(
    baseline_path: str,
    output_dir: str
) -> dict:
    """
    Run complete analysis pipeline.
    
    Args:
        baseline_path: Path to baseline data
        output_dir: Directory for outputs
    
    Returns:
        dict: Complete analysis results
    """
```
