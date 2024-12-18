# Architecture Overview

## Project Structure

```
remote-viewing-experiment/
├── src/
│   ├── unique_objects_analysis_global.py  # Core analysis functions
│   ├── selection_process.py               # Image selection logic
│   └── cluster_diversity.py              # Cluster analysis tools
├── scripts/
│   ├── run_analysis.py                   # Main execution script
│   └── test_real_data.py                # Testing utilities
└── data/
    └── raw/
        └── images/                       # Source images
```

## Core Components

### 1. Feature Extraction
```python
def extract_features(images, batch_size=16):
    """Extract ResNet-50 features from images"""
    # Uses ResNet-50 pretrained on ImageNet
    # Returns: numpy array of shape (n_images, 2048)
```

### 2. Distance Calculation
```python
def calculate_distance_metrics(features):
    """Calculate comprehensive distance metrics"""
    # Uses cosine similarity
    # Returns: dict with mean, min, max distances
```

### 3. Cluster Analysis
```python
def analyze_cluster_temporal_quality(features, clusters):
    """Analyze how cluster quality varies over time"""
    # Tracks cluster evolution
    # Returns: list of metrics per cluster
```

## Data Flow

1. **Image Loading**
   - Load images from directory
   - Preprocess for neural network
   - Batch processing for memory efficiency

2. **Feature Extraction**
   - ResNet-50 processing
   - Feature vector generation
   - Dimensionality: 2048 features per image

3. **Analysis Pipeline**
   - Distance calculation
   - Cluster formation
   - Statistical validation

4. **Output Generation**
   - Visualization creation
   - Metrics calculation
   - Results saving
