# CIFAR-100 Analysis Framework for Remote Viewing Experiment

## Overview
This document describes the implementation of CIFAR-100 dataset analysis for the remote viewing experiment. The framework uses both visual (CNN) and semantic (SBERT) features to create a diverse and well-distributed set of target images.

## Dataset Selection
CIFAR-100 was chosen for the following reasons:
- 100 distinct object categories
- Clean, labeled images
- Manageable size for initial testing
- Built-in PyTorch support
- Well-documented classes
- Consistent image quality

## Implementation Components

### 1. Data Loading (`load_cifar100.py`)
- **Purpose**: Manages dataset loading and sampling
- **Key Functions**:
  - `load_cifar100()`: Loads dataset with proper transformations
  - `get_class_samples()`: Creates balanced subset of images
- **Transformations**:
  ```python
  transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
  ])
  ```

### 2. Feature Extraction (`cifar_feature_extraction.py`)
#### Visual Features
- Uses AlexNet's last maxpool layer
- Extracts high-level visual features
- Implementation via PyTorch hooks
- Features saved as individual .pt files

#### Semantic Features
- Generated from class names and descriptions
- Uses SBERT embeddings
- Consistent description format:
  ```python
  description = f"a photograph of a {clean_name}. "
  description += f"This image shows a clear view of a {clean_name}. "
  description += f"The main subject is a {clean_name}."
  ```

#### Metadata Storage
- JSON format for easy access
- Stores:
  - Image index
  - Class ID
  - Class name
  - Description
  - Feature filename

## Data Organization

### Directory Structure
```
cifar_features/
├── metadata.json
├── semantic_features.npy
└── cnn_features/
    ├── 0_23.pt
    ├── 1_45.pt
    └── ...
```

### Feature Types
1. **CNN Features**
   - Format: PyTorch tensors (.pt files)
   - Naming: `{global_index}_{class_id}.pt`
   - Dimension: [256, 6, 6] (AlexNet maxpool output)

2. **Semantic Features**
   - Format: NumPy array (.npy file)
   - Dimension: [n_samples, 384] (SBERT embedding size)

3. **Metadata**
   - Format: JSON
   - Contains mapping between features and classes

## Analysis Pipeline

### 1. Initial Setup
```python
dataset, class_names = load_cifar100()
subset_dataset = get_class_samples(dataset)
```

### 2. Feature Extraction
```python
# CNN Features
alexnet = models.alexnet(pretrained=True)
feature_extractor = FeatureExtractor(alexnet)

# Semantic Features
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
```

### 3. Data Processing
- Batch processing for efficiency
- Parallel extraction of CNN and semantic features
- Metadata generation and storage

## Usage Guidelines

### 1. Dataset Preparation
```python
# Load dataset with 6 samples per class
dataset, class_names = load_cifar100()
subset, indices, labels = get_class_samples(dataset, samples_per_class=6)
```

### 2. Feature Extraction
```python
# Run feature extraction
python cifar_feature_extraction.py
```

### 3. Output Verification
- Check metadata.json for completeness
- Verify feature file existence
- Validate semantic embeddings

## Future Enhancements

### Planned Improvements
1. **Enhanced Semantic Descriptions**
   - Add detailed object attributes
   - Include contextual information
   - Multiple description templates

2. **Feature Analysis**
   - Hierarchical clustering implementation
   - Diversity metrics
   - Similarity visualization

3. **Selection Algorithms**
   - Distance-based sampling
   - Cluster-based selection
   - Hybrid approach combining visual and semantic features

### Next Steps
1. Implement hierarchical clustering
2. Develop image selection algorithm
3. Create visualization tools
4. Add validation metrics

## Notes
- Current implementation uses 6 samples per class (600 total images)
- Features are extracted in batches of 32
- Both CNN and SBERT features are normalized
- Metadata includes full provenance information

## Dependencies
- PyTorch
- torchvision
- sentence-transformers
- numpy
- json 