# Quick Start Guide

## Basic Usage

1. **Prepare Your Images**
   ```bash
   # Place images in the correct directory
   cp your_images/* data/raw/images/
   ```

2. **Run Feature Extraction**
   ```bash
   python scripts/run_analysis.py --start_from features
   ```

3. **View Results**
   Results will be saved in `experiments/experiment_N_TIMESTAMP/`

## Example Analysis

```python
# Basic analysis script
from src import unique_objects_analysis_global as analysis

# Load and analyze images
output_dir = "experiments/my_experiment"
results = analysis.run_analysis_from_baseline("baseline.json", output_dir)

# Print key metrics
print(f"Algorithm mean distance: {results['algorithm_metrics']['mean_dist']:.4f}")
print(f"P-value: {results['statistical_tests']['p_value']:.4f}")
```

## Understanding Outputs

The analysis generates several key files:
- Distance matrix visualizations
- Cluster analysis plots
- Statistical validation results

See [Understanding Outputs](../user-guide/outputs.md) for detailed explanations.
