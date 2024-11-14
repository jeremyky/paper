# Remote Viewing Experiment: Automated Analysis Framework

## Overview
This project introduces an automated, objective methodology for conducting and analyzing remote viewing experiments. The framework addresses traditional challenges in remote viewing research, specifically:
- Subjective human judgment in evaluating viewer accuracy
- Lack of reproducibility in analysis
- Scalability limitations
- Potential experimental bias

## Background

### Traditional Remote Viewing Experiments
In traditional remote viewing experiments:
1. A sender selects a target image
2. A remote viewer attempts to describe the target without seeing it
3. Human judges evaluate the accuracy by comparing the viewer's description to the target
4. Success is determined through statistical analysis of matching accuracy

### Current Limitations
- Human judgment introduces subjectivity
- Inconsistent evaluation criteria between different judges
- Time-consuming analysis process
- Difficult to replicate results across different studies
- Potential for unconscious bias in evaluation

## Proposed Methodology

### Automated Analysis Framework
Our method introduces two parallel analysis approaches:
1. **NLP-Based Analysis**
   - Standardized descriptions for target and decoy images
   - Semantic similarity comparison between viewer descriptions and image descriptions
   - Objective scoring based on linguistic similarity metrics

2. **CNN-Based Visual Analysis**
   - Computer vision analysis of image features
   - Direct comparison of visual similarities between images
   - Clustering-based validation of image relationships

### Validation Strategy
The framework's validity is established by comparing:
- NLP-generated similarity matrices
- CNN-generated similarity matrices
- Visualization through dendrograms and heatmaps
- Correlation between linguistic and visual clustering patterns

### Key Advantages
1. **Objectivity**: Removes human bias from evaluation process
2. **Reproducibility**: Standardized analysis methods ensure consistent results
3. **Scalability**: Automated analysis enables larger-scale experiments
4. **Validation**: Dual-analysis approach (NLP and CNN) provides robust validation
5. **Efficiency**: Reduces time and resources needed for analysis

## Experimental Process

### 1. Image Pool Preparation
- Collection of target and decoy images
- Development of standardized descriptions
- Processing images for CNN analysis

### 2. Remote Viewing Session
- Target selection from image pool
- Remote viewer provides description
- Recording of viewer's description

### 3. Automated Analysis
- NLP processing of viewer descriptions
- CNN analysis of image features
- Generation of similarity matrices
- Creation of dendrograms and heatmaps

### 4. Statistical Analysis
- Comparison of similarity scores
- Clustering analysis
- Statistical significance testing

## Technical Implementation

### 1. Image Description Standardization
- Selected 20 diverse images for initial validation
- Images chosen to represent varied characteristics:
  - Textures
  - Objects
  - Emotional content
  - Color schemes
- Created standardized descriptions:
  - 20 descriptors per image
  - Consistent complexity level across all images
  - Descriptions cover multiple aspects (visual, emotional, contextual)

### 2. NLP Analysis Implementation
#### SBERT Encoding Approaches

##### Approach Comparison
Two potential methods were considered for encoding image descriptions:

1. **Individual Descriptor Encoding**
   - *Process*: Encode each of the 20 descriptors separately
   - *Advantages*:
     - More granular representation of each descriptor
     - Maintains full semantic meaning of individual descriptors
     - Allows for descriptor-level similarity analysis
     - Better handles descriptors that might conflict or contradict
   - *Disadvantages*:
     - Results in 20 separate embedding vectors per image
     - More computationally intensive
     - Requires additional aggregation strategy
     - May lose contextual relationships between descriptors

2. **Combined Descriptor Encoding**
   - *Process*: Encode all 20 descriptors as one comma-separated text
   - *Advantages*:
     - Single embedding vector per image
     - Captures potential relationships between descriptors
     - More efficient computation
     - Simpler similarity comparison between images
   - *Disadvantages*:
     - May dilute the importance of individual descriptors
     - Could hit token length limits for transformer models
     - Risk of losing fine-grained semantic details
     - Potential for descriptor order to affect encoding

##### Implementation Decision
The combined descriptor encoding approach was selected for implementation:

- **Implementation Details**:
  - Uses all-MiniLM-L6-v2 SBERT model
  - Processes each image's 20 descriptors as a single text input
  - Generates one feature vector per image
  - Stores descriptors in combined_descriptors.txt

- **Key Implementation Benefits**:
  - Simplified similarity computation between images
  - More efficient processing pipeline
  - Maintains contextual relationships between descriptors
  - Single embedding vector per image enables straightforward clustering

- **Processing Flow**:
  1. Load combined descriptors from text file
  2. Encode each combined description using SBERT
  3. Generate similarity matrix using correlation distance
  4. Create hierarchical clustering using Ward linkage
  5. Visualize results through dendrograms and heatmaps

### 3. CNN Analysis Implementation
- Computer vision analysis of image features
- Direct comparison of visual similarities between images
- Clustering-based validation of image relationships

## Future Directions
- Expansion of image dataset
- Refinement of similarity metrics
- Integration of additional analysis methods
- Development of real-time analysis capabilities

## Impact
This methodology represents a significant advancement in remote viewing research by:
- Establishing objective evaluation standards
- Enabling larger-scale studies
- Providing reproducible results
- Creating a foundation for more rigorous scientific investigation

### Visualization Methodology

#### Distance Matrix Generation Approaches

1. **Correlation-based Approach** (Original sbert-analysis.py)
   - Uses correlation distance metric
   - Focuses on pattern similarity
   - May emphasize relative relationships between features
   ```python
   condensed_dist_matrix_cnn = pdist(sbert_feature_vectors, metric="correlation")
   ```

2. **Cosine Similarity Approach** (Improved working-dendrogram.py)
   - First computes cosine similarity
   - Better suited for semantic text embeddings
   - More interpretable for text-based comparisons
   ```python
   similarity_matrix = cosine_similarity(embeddings_whole)
   distance_matrix = 1 - similarity_matrix
   ```

#### Implementation Decision
The cosine similarity approach was chosen as optimal for our experiment because:
- More appropriate for semantic text comparisons
- Standard metric in NLP tasks
- Better interpretation of similarity scores
- Directly comparable to human intuition about text similarity

#### Additional Analysis Features
The improved implementation includes:
- Top-N most related pairs identification
- Detailed similarity score output
- Multiple clustering method options
- Enhanced validation capabilities for remote viewing matches

### Hierarchical Clustering Implementation

#### Linkage Methods Analysis
Four different linkage methods were implemented and compared for semantic clustering:

1. **Ward Linkage**
   - *Threshold*: 0.3
   - *Characteristics*:
     - Minimizes variance within clusters
     - Creates compact, balanced clusters
     - Best for finding groups of descriptions with similar semantic content
     - Useful when descriptions should form distinct, well-separated groups
   - *Application*: Ideal for identifying major thematic groups in image descriptions

2. **Complete Linkage**
   - *Threshold*: 0.5
   - *Characteristics*:
     - Uses maximum distances between points
     - Creates evenly sized clusters
     - Conservative in forming clusters
     - Sensitive to outliers in descriptions
   - *Application*: Helpful for finding highly distinct semantic groups

3. **Average Linkage**
   - *Threshold*: 0.4
   - *Characteristics*:
     - Uses mean distances between all pairs
     - Balances cluster characteristics
     - More robust to description variations
     - Provides middle-ground between single and complete linkage
   - *Application*: Good for general-purpose semantic clustering

4. **Single Linkage**
   - *Threshold*: 0.2
   - *Characteristics*:
     - Uses minimum distances between pairs
     - Can find elongated clusters
     - Sensitive to semantic bridges between clusters
     - May create chain-like clusters
   - *Application*: Useful for finding gradual semantic transitions

#### Visualization Enhancements
- **Color Coding**:
  - Distinct colors for each cluster
  - Grey for connections above threshold
  - Helps visualize cluster boundaries
  - Makes relationship patterns more apparent

- **Threshold Visualization**:
  - Horizontal red line showing cut-off point
  - Helps interpret cluster formation
  - Allows for consistent cluster identification

#### Cluster Analysis Output
For each linkage method:
- Detailed cluster assignments
- Color-coded groupings
- Similarity scores within clusters
- Inter-cluster relationships

#### Selection Criteria for Remote Viewing Analysis
The optimal linkage method should:
1. Create semantically meaningful clusters
2. Maintain distinction between different description types
3. Be robust to variations in description style
4. Provide consistent clustering across similar descriptions

Based on these criteria, [preferred method] linkage was selected as the primary clustering method because [reasoning].
