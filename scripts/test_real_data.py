import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import logging
from datetime import datetime
from tqdm import tqdm
import gc
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src import unique_objects_analysis_global as analysis

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_images(image_dir, batch_size=100):
    """Load and preprocess images from directory in batches"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.thl'}
    image_files = [f for f in os.listdir(image_dir) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    logger.info(f"Found {len(image_files)} valid image files")
    
    images = []
    image_names = []
    
    # Process images in batches with progress bar
    for i in tqdm(range(0, len(image_files), batch_size), desc="Loading images"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        
        for filename in batch_files:
            try:
                image_path = os.path.join(image_dir, filename)
                image = Image.open(image_path).convert('RGB')
                processed_image = transform(image)
                batch_images.append(processed_image)
                image_names.append(filename)
            except Exception as e:
                logger.warning(f"Error loading image {filename}: {str(e)}")
        
        if batch_images:
            images.append(torch.stack(batch_images))
            
        # Clear memory
        gc.collect()
    
    # Combine all batches
    all_images = torch.cat(images)
    logger.info(f"Successfully loaded {len(image_names)} images")
    return all_images, image_names

def extract_features(images, batch_size=16):
    """Extract features using ResNet-50 with batching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    features = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    
    logger.info("Extracting features...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), total=n_batches, desc="Extracting features"):
            # Process batch
            batch = images[i:i + batch_size].to(device)
            batch_features = model(batch).squeeze()
            
            # Handle single-image batch case
            if len(batch) == 1:
                batch_features = batch_features.unsqueeze(0)
            
            # Move to CPU and convert to numpy
            features.append(batch_features.cpu().numpy())
            
            # Clear GPU memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Clear CPU memory
            gc.collect()
    
    # Combine all features
    all_features = np.vstack(features)
    logger.info(f"Extracted features shape: {all_features.shape}")
    return all_features

def create_test_clusters(features, n_clusters=20, images_per_cluster=5):
    """Create initial test clusters"""
    n_images = len(features)
    indices = np.random.permutation(n_images)
    
    clusters = {}
    for i in range(n_clusters):
        start_idx = i * images_per_cluster
        end_idx = start_idx + images_per_cluster
        if end_idx <= len(indices):
            clusters[str(i)] = indices[start_idx:end_idx].tolist()
    
    return clusters

def test_with_real_data():
    """Run analysis pipeline on real image data"""
    try:
        # Load and process images
        image_dir = os.path.join(project_root, "ObjectsAll", "OBJECTSALL")
        images, image_names = load_images(image_dir)
        
        # Extract features
        features = extract_features(images)
        
        # Free memory
        del images
        gc.collect()
        
        # Create test clusters
        logger.info("Creating test clusters...")
        clusters = create_test_clusters(features)
        
        # Create test output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir = os.path.join(project_root, "test_outputs", f"real_data_{timestamp}")
        os.makedirs(test_dir, exist_ok=True)
        
        # Save baseline data
        logger.info("Saving baseline data...")
        baseline_data = {
            'features': features.tolist(),
            'clusters': clusters,
            'image_names': image_names
        }
        
        baseline_path = os.path.join(test_dir, "baseline")
        with open(baseline_path + '.json', 'w') as f:
            json.dump(baseline_data, f)
        
        # Free memory
        del baseline_data
        gc.collect()
        
        # Run analysis
        logger.info("Running analysis...")
        results = analysis.run_analysis_from_baseline(baseline_path, test_dir)
        
        # Save results
        logger.info("Saving results...")
        results_path = os.path.join(test_dir, "analysis_results.json")
        with open(results_path, 'w') as f:
            serializable_results = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results.items()
            }
            json.dump(serializable_results, f, indent=2)
        
        # Log summary
        logger.info("\nAnalysis Summary:")
        logger.info(f"Number of images: {len(image_names)}")
        logger.info(f"Number of clusters: {len(clusters)}")
        logger.info(f"Algorithm mean distance: {results['algorithm_metrics']['mean_dist']:.4f}")
        logger.info(f"Random mean distance: {results['random_metrics']['mean_dist']:.4f}")
        logger.info(f"P-value: {results['statistical_tests']['p_value']:.4f}")
        
        return test_dir
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting real data analysis...")
        output_dir = test_with_real_data()
        logger.info(f"Analysis complete! Results saved in: {output_dir}")
    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
    except Exception as e:
        logger.error("Analysis failed", exc_info=True)