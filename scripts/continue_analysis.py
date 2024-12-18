import os
import sys
import numpy as np
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_cluster_with_images(features, image_paths, indices, cluster_id, output_dir):
    """Plot cluster dendrogram, heatmap, and images"""
    cluster_dir = os.path.join(output_dir, 'cluster_analysis', f'cluster_{cluster_id}')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Get cluster features and images
    cluster_features = features[indices]
    cluster_images = [image_paths[i] for i in indices]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 1. Dendrogram
    ax1 = plt.subplot(gs[0, 0])
    cluster_linkage = linkage(cluster_features, method='ward')
    dendrogram(cluster_linkage)
    ax1.set_title(f'Cluster {cluster_id} Dendrogram')
    
    # 2. Distance Heatmap
    ax2 = plt.subplot(gs[0, 1])
    distances = 1 - cosine_similarity(cluster_features)
    sns.heatmap(distances, ax=ax2, cmap='viridis', annot=True, fmt='.2f')
    ax2.set_title('Distance Matrix')
    
    # 3. Images
    ax3 = plt.subplot(gs[1, :])
    image_grid = np.zeros((224, 224*5, 3))
    
    # Save individual images and create grid
    for idx, img_path in enumerate(cluster_images):
        try:
            # Load and save individual image
            img = Image.open(os.path.join(project_root, "ObjectsAll", "OBJECTSALL", img_path))
            img_save_path = os.path.join(cluster_dir, f"image_{idx}_{os.path.basename(img_path)}")
            img.save(img_save_path)
            
            # Add to grid
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized)
            image_grid[:, idx*224:(idx+1)*224] = img_array
            
        except Exception as e:
            logger.warning(f"Error processing image {img_path}: {e}")
    
    ax3.imshow(image_grid)
    ax3.axis('off')
    ax3.set_title('Cluster Images')
    
    plt.tight_layout()
    plt.savefig(os.path.join(cluster_dir, 'cluster_analysis.png'), dpi=300)
    plt.close()
    
    # Save cluster information with mappings
    cluster_info = {
        'images': [
            {
                'filename': img_path,
                'index': int(idx),
                'feature_vector': cluster_features[i].tolist()
            }
            for i, (idx, img_path) in enumerate(zip(indices, cluster_images))
        ],
        'distances': distances.tolist(),
        'indices': indices.tolist()
    }
    
    with open(os.path.join(cluster_dir, 'cluster_info.json'), 'w') as f:
        json.dump(cluster_info, f, indent=2)

def plot_complete_dendrogram(features, output_dir):
    """Plot dendrogram for all images"""
    logger.info("Generating complete dendrogram...")
    
    plt.figure(figsize=(20, 10))
    linkage_matrix = linkage(features, method='ward')
    dendrogram(linkage_matrix)
    plt.title('Complete Hierarchical Clustering Dendrogram (100 Images)')
    plt.xlabel('Image Index')
    plt.ylabel('Distance')
    
    # Save dendrogram
    plt.savefig(os.path.join(output_dir, 'cluster_analysis', 'complete_dendrogram.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return linkage_matrix

def continue_analysis():
    """Continue analysis from existing baseline"""
    output_dir = os.path.join(project_root, "docs", "assets", "example_outputs")
    baseline_path = os.path.join(output_dir, "baseline.json")
    
    logger.info(f"Loading baseline from: {baseline_path}")
    
    try:
        # Load baseline data
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        # Extract features and image names
        features = np.array(baseline_data['features'])
        image_names = baseline_data['image_names']
        
        # Create cluster analysis directory
        cluster_dir = os.path.join(output_dir, 'cluster_analysis')
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Generate complete dendrogram
        linkage_matrix = plot_complete_dendrogram(features, output_dir)
        
        # Create clusters if not already present
        if 'clusters' not in baseline_data:
            cluster_labels = fcluster(linkage_matrix, 20, criterion='maxclust')
            clusters = {}
            for i in range(20):
                cluster_indices = np.where(cluster_labels == i+1)[0]
                clusters[str(i)] = cluster_indices[:5].tolist()
            baseline_data['clusters'] = clusters
            
            # Save updated baseline
            with open(baseline_path, 'w') as f:
                json.dump(baseline_data, f)
        else:
            clusters = baseline_data['clusters']
        
        # Generate visualizations for each cluster
        logger.info("Generating cluster visualizations...")
        for cluster_id, indices in clusters.items():
            logger.info(f"Processing cluster {cluster_id}")
            plot_cluster_with_images(features, image_names, np.array(indices), 
                                   cluster_id, output_dir)
        
        # Create markdown with image paths
        with open(os.path.join(output_dir, 'cluster_examples.md'), 'w') as f:
            f.write("# Cluster Examples\n\n")
            for cluster_id in clusters:
                f.write(f"## Cluster {cluster_id}\n\n")
                f.write(f"### Analysis\n")
                f.write(f"![Cluster Analysis](cluster_analysis/cluster_{cluster_id}/cluster_analysis.png)\n\n")
                f.write(f"### Individual Images\n")
                for i in range(5):
                    f.write(f"![Image {i}](cluster_analysis/cluster_{cluster_id}/image_{i}_*.jpg)\n")
                f.write("\n")
        
        logger.info(f"Analysis complete! Results saved in: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        output_dir = continue_analysis()
        print(f"\nResults saved in: {output_dir}")
    except Exception as e:
        logger.error(f"Error continuing analysis: {str(e)}", exc_info=True) 