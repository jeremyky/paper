import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.config.experiment_config import PIPELINE_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_next_experiment_number(base_dir):
    """Get the next experiment number by checking existing directories"""
    existing_dirs = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d)) 
                    and d.startswith('experiment_')]
    if not existing_dirs:
        return 1
    
    numbers = [int(d.split('_')[1]) for d in existing_dirs]
    return max(numbers) + 1

def create_experiment_dir():
    """Create a unique experiment directory with timestamp"""
    # Base directory for all experiments
    experiments_dir = os.path.join(project_root, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Get next experiment number
    exp_num = get_next_experiment_number(experiments_dir)
    
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(experiments_dir, f"experiment_{exp_num}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for different analyses
    os.makedirs(os.path.join(exp_dir, "original_analysis"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "global_analysis"), exist_ok=True)
    
    return exp_dir

def save_analysis_summary(exp_dir, metrics):
    """Save analysis metrics and create visualizations"""
    summary_dir = os.path.join(exp_dir, "analysis_summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. Save metrics as text file
    with open(os.path.join(summary_dir, "metrics_summary.txt"), "w") as f:
        f.write("=== Analysis Metrics Summary ===\n\n")
        
        # Monte Carlo Results
        f.write("Monte Carlo Simulation Results:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Number of Simulations: {metrics['monte_carlo']['n_simulations']}\n")
        f.write(f"Algorithm Mean Distance: {metrics['monte_carlo']['algorithm_mean_dist']:.4f}\n")
        f.write(f"Random Selection Stats:\n")
        f.write(f"  Mean: {metrics['monte_carlo']['random_mean']:.4f}\n")
        f.write(f"  Std: {metrics['monte_carlo']['random_std']:.4f}\n")
        f.write(f"  Min: {metrics['monte_carlo']['random_min']:.4f}\n")
        f.write(f"  Max: {metrics['monte_carlo']['random_max']:.4f}\n")
        f.write(f"P-value: {metrics['monte_carlo']['p_value']:.6f}\n\n")
        
        # Clustering Results
        f.write("Clustering Results:\n")
        f.write("-" * 50 + "\n")
        f.write("Per-Cluster Statistics:\n")
        for cluster_id, stats in metrics['clustering'].items():
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Mean Internal Distance: {stats['mean_dist']:.4f}\n")
            f.write(f"  Min Internal Distance: {stats['min_dist']:.4f}\n")
            f.write(f"  Max Internal Distance: {stats['max_dist']:.4f}\n")
    
    # 2. Create visualizations
    
    # Monte Carlo Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(metrics['monte_carlo']['random_distances'], bins=50)
    plt.axvline(metrics['monte_carlo']['algorithm_mean_dist'], 
                color='r', linestyle='--', label='Algorithm Performance')
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Mean Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(summary_dir, 'monte_carlo_distribution.png'))
    plt.close()
    
    # Cluster Performance Plot
    cluster_means = [stats['mean_dist'] for stats in metrics['clustering'].values()]
    cluster_ids = list(metrics['clustering'].keys())
    
    plt.figure(figsize=(12, 6))
    plt.plot(cluster_ids, cluster_means, marker='o')
    plt.title('Cluster Performance')
    plt.xlabel('Cluster ID')
    plt.ylabel('Mean Internal Distance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(summary_dir, 'cluster_performance.png'))
    plt.close()
    
    # Save raw metrics as JSON for future reference
    with open(os.path.join(summary_dir, "raw_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def main():
    """Run the complete analysis pipeline"""
    try:
        # Setup paths
        input_dir = os.path.join(project_root, "data", "raw", "mm_unique_objects")
        feature_dir = os.path.join(project_root, "src", "unique_objects_features")
        baseline_dir = os.path.join(project_root, PIPELINE_CONFIG['baseline_dir'])
        
        # Create experiment directory
        exp_dir = create_experiment_dir()
        logger.info(f"Created experiment directory: {exp_dir}")
        os.environ['EXPERIMENT_DIR'] = exp_dir
        
        metrics = {
            'monte_carlo': {},
            'clustering': {}
        }
        
        # Check pipeline starting point
        start_from = PIPELINE_CONFIG['start_from']
        logger.info(f"Starting pipeline from: {start_from}")
        
        if start_from == 'features':
            # Full run - start with feature extraction
            logger.info("Starting from feature extraction...")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(feature_dir, exist_ok=True)
            
            # Run feature extraction
            import src.unique_objects_analysis
            
        if start_from in ['features', 'clusters']:
            # Run clustering if not starting from analysis
            logger.info("Running clustering analysis...")
            from src import unique_objects_analysis_global
            analysis_results = unique_objects_analysis_global.run_analysis(
                os.path.join(exp_dir, 'global_analysis')
            )
            
            if PIPELINE_CONFIG['save_baseline']:
                # Save current clustering as baseline
                os.makedirs(baseline_dir, exist_ok=True)
                baseline_path = os.path.join(baseline_dir, PIPELINE_CONFIG['baseline_name'])
                logger.info(f"Saving baseline clustering to {baseline_path}")
                unique_objects_analysis_global.save_clustering(baseline_path)
        
        else:  # start_from == 'analysis'
            if PIPELINE_CONFIG['use_baseline']:
                # Use baseline clustering
                baseline_path = os.path.join(baseline_dir, PIPELINE_CONFIG['baseline_name'])
                if not os.path.exists(baseline_path + '.json'):
                    raise FileNotFoundError(f"No baseline clustering found at {baseline_path}")
                
                logger.info(f"Loading baseline clustering from {baseline_path}")
                from src import unique_objects_analysis_global
                analysis_results = unique_objects_analysis_global.run_analysis_from_baseline(
                    baseline_path,
                    os.path.join(exp_dir, 'global_analysis')
                )
            else:
                raise ValueError("Must set use_baseline=True when starting from analysis")
        
        # Update metrics from analysis results
        metrics['monte_carlo'] = {
            'n_simulations': 1000,
            'algorithm_mean_dist': analysis_results['algorithm_metrics'],
            'random_mean': np.mean(analysis_results['random_metrics']),
            'random_std': np.std(analysis_results['random_metrics']),
            'random_min': np.min(analysis_results['random_metrics']),
            'random_max': np.max(analysis_results['random_metrics']),
            'p_value': analysis_results['p_value'],
            'random_distances': analysis_results['random_metrics']
        }
        
        metrics['clustering'] = analysis_results['cluster_metrics']
        
        # Save analysis summary
        save_analysis_summary(exp_dir, metrics)
        
        # Save experiment info
        with open(os.path.join(exp_dir, "experiment_info.txt"), "w") as f:
            f.write(f"Experiment run at: {datetime.now()}\n")
            f.write(f"Input directory: {input_dir}\n")
            f.write(f"Feature directory: {feature_dir}\n")
            f.write(f"Pipeline configuration:\n")
            for key, value in PIPELINE_CONFIG.items():
                f.write(f"  {key}: {value}\n")
        
        logger.info(f"Analysis complete! Results saved in {exp_dir}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 