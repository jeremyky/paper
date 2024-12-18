import os
import sys
import numpy as np
import logging
from datetime import datetime
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import unique_objects_analysis_global as analysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_images(image_dir):
    """Load and preprocess images"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    image_names = []
    
    logger.info("Loading images...")
    for filename in tqdm(os.listdir(image_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                processed_image = transform(image)
                images.append(processed_image)
                image_names.append(filename)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
    
    return torch.stack(images), image_names

def extract_features(images):
    """Extract features using ResNet-50"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
            batch = images[i:i+batch_size].to(device)
            batch_features = model(batch).squeeze()
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

def create_clusters(features, n_clusters=20, images_per_cluster=5):
    """Create clusters using hierarchical clustering"""
    logger.info("Creating clusters...")
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(features, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Organize into clusters
    clusters = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i+1)[0]
        clusters[str(i)] = cluster_indices[:images_per_cluster].tolist()
    
    return clusters

def run_analysis_pipeline():
    """Run complete analysis and save outputs for documentation"""
    
    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "docs", "assets", "example_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process images
    image_dir = os.path.join(project_root, "ObjectsAll", "OBJECTSALL")
    logger.info(f"Processing images from: {image_dir}")
    
    # Load and process images
    images, image_names = load_and_process_images(image_dir)
    
    # Extract features
    features = extract_features(images)
    
    # Create clusters
    clusters = create_clusters(features)
    
    # Save baseline data
    baseline_data = {
        'features': features.tolist(),
        'clusters': clusters,
        'image_names': image_names
    }
    
    baseline_path = os.path.join(output_dir, "baseline")
    with open(baseline_path + '.json', 'w') as f:
        import json
        json.dump(baseline_data, f)
    
    # Run analysis
    results = analysis.run_analysis_from_baseline(baseline_path, output_dir)
    
    # Save results summary
    with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
        f.write("Analysis Summary\n")
        f.write("================\n\n")
        f.write(f"Algorithm mean distance: {results['algorithm_metrics']['mean_dist']:.4f}\n")
        f.write(f"Random mean distance: {results['random_metrics']['mean_dist']:.4f}\n")
        f.write(f"P-value: {results['statistical_tests']['p_value']:.4f}\n")
        f.write(f"Effect size: {results['statistical_tests']['effect_size']:.4f}\n")
    
    logger.info(f"Analysis complete! Results saved in: {output_dir}")
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = run_analysis_pipeline()
        print(f"\nResults saved in: {output_dir}")
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True) 