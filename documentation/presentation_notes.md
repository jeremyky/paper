# Remote Viewing Experiment - Changes and Updates

## Major Changes Implemented

1. **Documentation Website Setup**
   - Created MkDocs-based documentation
   - Added comprehensive analysis examples
   - Organized technical documentation

2. **Analysis Pipeline Improvements**
   - Added cluster visualization improvements
   - Enhanced statistical validation
   - Added temporal quality analysis
   - Improved minimum distance comparisons

3. **Visualization Enhancements**
   - Added dendrograms for each cluster
   - Created distance matrix heatmaps
   - Added temporal quality plots
   - Improved cluster visualization layout

4. **Dataset Documentation**
   - Added Harvard Konklab dataset attribution
   - Documented image preprocessing steps
   - Added cluster analysis examples

## How to Run the Analysis

1. **Complete Analysis Pipeline**
```bash
# Run full analysis from features
python scripts/run_analysis.py --start_from features
```

2. **View Documentation**
```bash
# Start documentation server
mkdocs serve
# View at http://127.0.0.1:8000
```

## Key Results to Show

1. **Cluster Analysis**
   - Show complete dendrogram (100 images)
   - Highlight specific clusters (0, 5, 14, 19)
   - Demonstrate diversity within clusters

2. **Statistical Validation**
   - Monte Carlo simulation results
   - P-value significance (0.001)
   - Effect size (0.842)

3. **Visualization Examples**
   - Distance matrix heatmap
   - Temporal quality plot
   - Individual cluster dendrograms

## Directory Structure
```
remote-viewing-experiment/
├── docs/                      # Documentation
│   ├── examples/             # Analysis examples
│   ├── technical/            # Technical docs
│   └── assets/              # Images and outputs
├── src/                      # Source code
└── scripts/                  # Analysis scripts
```

## Key Files to Show

1. **Analysis Examples** (`docs/examples/analysis.md`)
   - Shows complete analysis workflow
   - Includes all visualizations
   - Documents statistical results

2. **Technical Documentation**
   - Architecture overview
   - API reference
   - Configuration options

3. **Output Examples**
   - Cluster visualizations
   - Statistical results
   - Distance matrices

## Running a Demo

1. **Start with Dataset**
   - Show Harvard Konklab dataset
   - Explain preprocessing steps
   - Demonstrate feature extraction

2. **Show Clustering**
   - Run clustering algorithm
   - Show dendrogram formation
   - Explain cluster selection

3. **Present Results**
   - Show statistical validation
   - Demonstrate cluster quality
   - Compare with random baseline

## Future Improvements

1. **Planned Enhancements**
   - SBERT integration
   - Additional visualization options
   - Enhanced cluster metrics

2. **Potential Extensions**
   - Interactive visualizations
   - Additional statistical tests
   - More cluster analysis tools

## Questions to Address

1. **Methodology**
   - Why ResNet-50 features?
   - How are clusters formed?
   - Why these statistical tests?

2. **Results**
   - What do p-values mean?
   - How to interpret dendrograms?
   - Why these specific clusters?

3. **Implementation**
   - How to modify parameters?
   - How to add new features?
   - How to extend analysis? 