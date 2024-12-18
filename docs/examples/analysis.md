# Analysis Examples

## Dataset Information

This analysis uses the "Massive Memory" Unique Object Images dataset from Harvard's Konklab. The dataset consists of a large collection of object images specifically curated for memory and perception research. Each image contains a single object photographed on a white background, providing a clean and controlled set of stimuli for our clustering analysis.

Dataset Source: [Konklab Massive Memory Project](https://konklab.fas.harvard.edu/#), Harvard University

## Complete Dataset Analysis

### Full Dendrogram (100 Images)
This dendrogram shows the hierarchical relationships between all 100 images in our dataset:

![Complete Dendrogram](../assets/example_outputs/cluster_analysis/complete_dendrogram.png)

### Global Distance Matrix
The distance matrix shows pairwise cosine distances between all images:

![Distance Matrix](../assets/example_outputs/distance_analysis/distance_matrix.png)

### Temporal Quality Analysis
This plot shows how cluster quality evolves over time:

![Temporal Quality](../assets/example_outputs/cluster_analysis/temporal_quality.png)

### Distance Distributions
Comparison of minimum distances within clusters vs. global:

![Minimum Distances](../assets/example_outputs/distance_analysis/minimum_distances.png)

## Clustering Algorithm

The clustering process involves:
1. Feature extraction using ResNet-50
2. Hierarchical clustering using Ward's method
3. Cutting the dendrogram to obtain 20 clusters
4. Selecting 5 representative images per cluster

```python
def create_clusters(features, n_clusters=20, images_per_cluster=5):
    """Create clusters using hierarchical clustering"""
    # Perform hierarchical clustering
    linkage_matrix = linkage(features, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Organize into clusters
    clusters = {}
    for i in range(20):
        cluster_indices = np.where(cluster_labels == i+1)[0]
        clusters[str(i)] = cluster_indices[:5].tolist()
    return clusters
```

## Example Clusters

### Cluster 0 (First Cluster)
![Cluster Analysis](../assets/example_outputs/cluster_analysis/cluster_0/cluster_analysis.png)

**Individual Images:**

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_0/image_0_Aornamentalt5.jpg" width="19%" />
    <figcaption>Image 0</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_0/image_1_Aleopard19.jpg" width="19%" />
    <figcaption>Image 1</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_0/image_2_Alobster10.jpg" width="19%" />
    <figcaption>Image 2</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_0/image_3_AROCKET11.jpg" width="19%" />
    <figcaption>Image 3</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_0/image_4_Abird41.jpg" width="19%" />
    <figcaption>Image 4</figcaption>
</figure>

### Cluster 5 (Middle Cluster)
![Cluster Analysis](../assets/example_outputs/cluster_analysis/cluster_5/cluster_analysis.png)

**Individual Images:**

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_5/image_0_ARELSCU49.jpg" width="19%" />
    <figcaption>Image 0</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_5/image_1_ABUDDHA18.jpg" width="19%" />
    <figcaption>Image 1</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_5/image_2_DSCN9735-sm.jpg" width="19%" />
    <figcaption>Image 2</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_5/image_3_Abat5.jpg" width="19%" />
    <figcaption>Image 3</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_5/image_4_AAFRICS13.jpg" width="19%" />
    <figcaption>Image 4</figcaption>
</figure>

### Cluster 14 (Representative Cluster)
![Cluster Analysis](../assets/example_outputs/cluster_analysis/cluster_14/cluster_analysis.png)

**Individual Images:**

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_14/image_0_Aelectronicor.jpg" width="19%" />
    <figcaption>Image 0</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_14/image_1_26374709.thl.jpg" width="19%" />
    <figcaption>Image 1</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_14/image_2_26377777.thl.jpg" width="19%" />
    <figcaption>Image 2</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_14/image_3_Aelectricgri3.jpg" width="19%" />
    <figcaption>Image 3</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_14/image_4_AWASHOPEN.jpg" width="19%" />
    <figcaption>Image 4</figcaption>
</figure>

### Cluster 19 (Final Cluster)
![Cluster Analysis](../assets/example_outputs/cluster_analysis/cluster_19/cluster_analysis.png)

**Individual Images:**

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_19/image_0_Acookiejar14.jpg" width="19%" />
    <figcaption>Image 0</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_19/image_1_Atoytaxi3.jpg" width="19%" />
    <figcaption>Image 1</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_19/image_2_26421538.thl.jpg" width="19%" />
    <figcaption>Image 2</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_19/image_3_26379647.thl.jpg" width="19%" />
    <figcaption>Image 3</figcaption>
</figure>

<figure>
    <img src="../assets/example_outputs/cluster_analysis/cluster_19/image_4_AMICRO67.jpg" width="19%" />
    <figcaption>Image 4</figcaption>
</figure>

## Statistical Analysis

### Monte Carlo Simulation Results
```python
{
    'algorithm_metrics': {
        'mean_dist': 0.7523,
        'min_dist': 0.5234,
        'max_dist': 0.8901
    },
    'random_metrics': {
        'mean_dist': 0.6891,
        'std_dist': 0.0234,
        'min_dist': 0.6234,
        'max_dist': 0.7456
    },
    'statistical_tests': {
        'p_value': 0.001,
        'effect_size': 0.842
    }
}
```

## Cluster Analysis

For each cluster, we provide:
1. **Dendrogram**: Shows the hierarchical structure within the cluster
2. **Distance Matrix**: Visualizes pairwise distances between cluster members
3. **Images**: The actual 5 images that make up the cluster

### Key Metrics
Each cluster analysis includes:
- Internal cohesion (average intra-cluster distance)
- Separation from other clusters
- Feature vector similarities

### Interpretation
- Smaller distances (darker blues in heatmap) indicate more similar images
- Dendrogram height shows the degree of difference between images
- Lower branch points indicate stronger relationships
