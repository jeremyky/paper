# Analysis Output Interpretation Guide

## Visualization Outputs

### 1. Temporal Quality Plot (`temporal_quality.png`)
Located in: `cluster_analysis/temporal_quality.png`

**What it shows:**
- X-axis: Cluster IDs in temporal order (0-19)
- Y-axis: Mean internal distance within each cluster
- Red dashed line: Overall average distance
- Shaded area: Standard deviation range

**How to interpret:**
- Higher values = More diverse images within cluster
- Consistent values = Stable algorithm performance
- Downward trend = Later clusters less diverse
- Upward trend = Later clusters more diverse
- Wide shaded area = High variability within clusters

**Key metrics to look for:**
- Clusters significantly above/below average
- Sudden changes in diversity
- Overall trend direction

### 2. Minimum Distances Plot (`minimum_distances.png`)
Located in: `distance_analysis/minimum_distances.png`

**What it shows:**
- Histogram comparing two distributions:
  1. Global: Closest pairs across all images
  2. Within Clusters: Closest pairs within each cluster
- X-axis: Distance values (0-1, cosine distance)
- Y-axis: Frequency count

**How to interpret:**
- Left-shifted peaks = More similar images
- Right-shifted peaks = More diverse images
- Overlap between distributions = Cluster effectiveness
- Separate peaks = Clear distinction between global and cluster distances

**Key patterns to look for:**
- Bimodal distributions
- Outlier distances
- Distribution overlap

## Metrics Summary

### 1. Algorithm Performance
Found in `metrics_summary.txt`

**Key metrics:**
- Algorithm Mean Distance: Overall diversity measure
  - Higher = More diverse selection
  - Target: > Random mean
- Minimum Distance: Closest pair of images
  - Higher = Better separation
  - Compare with random minimum

### 2. Statistical Validation
**P-value interpretation:**
- < 0.05: Algorithm significantly better than random
- < 0.01: Strong evidence of effectiveness
- > 0.05: Not significantly better than random

**Effect size (Cohen's d):**
- 0.2: Small effect
- 0.5: Medium effect
- 0.8: Large effect
- > 1.0: Very large effect

### 3. Cluster Quality Metrics
**Per-cluster statistics:**
- Mean Internal Distance: Average diversity within cluster
- Min/Max Distance: Range of similarities
- Standard Deviation: Consistency measure

**What to look for:**
- Consistent mean distances across clusters
- Reasonable min/max ranges
- Small standard deviations

## Common Patterns

### Good Performance Indicators:
1. Temporal plot shows consistent diversity
2. Minimum distances higher than random
3. Clear separation in distance distributions
4. P-value < 0.05 with large effect size
5. Consistent cluster metrics

### Warning Signs:
1. Declining temporal quality
2. Overlapping distance distributions
3. High p-value (> 0.05)
4. Large variations between clusters
5. Very small minimum distances

## Using the Results

1. **Algorithm Validation:**
   - Compare algorithm vs random metrics
   - Check statistical significance
   - Evaluate effect size

2. **Cluster Quality:**
   - Review temporal patterns
   - Check cluster consistency
   - Identify problematic clusters

3. **Image Pair Analysis:**
   - Examine closest pairs
   - Verify diversity within clusters
   - Identify potential improvements

## Troubleshooting

If results show:
1. **Poor diversity:**
   - Adjust selection algorithm parameters
   - Increase minimum distance thresholds
   - Review feature extraction

2. **Inconsistent clusters:**
   - Adjust clustering parameters
   - Review cluster size settings
   - Check feature normalization

3. **Statistical issues:**
   - Increase number of Monte Carlo simulations
   - Review random sampling method
   - Check for data anomalies 