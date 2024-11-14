import torch
from torchvision import models
import os
import numpy as np
from load_cifar100 import load_cifar100, get_class_samples
from sentence_transformers import SentenceTransformer
import json
import torch.utils.data


class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        # Register hook on last maxpool layer
        self.hook = model.features[12].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove_hook(self):
        self.hook.remove()

def generate_semantic_descriptions(class_names):
    """
    Generate semantic descriptions for each class
    Returns a dictionary mapping class index to description
    """
    descriptions = {}
    for idx, class_name in enumerate(class_names):
        # Create a richer description by:
        # 1. Replacing underscores with spaces
        # 2. Adding "a photograph of" prefix
        # 3. Adding common attributes
        clean_name = class_name.replace('_', ' ')
        description = f"a photograph of a {clean_name}. "
        description += f"This image shows a clear view of a {clean_name}. "
        description += f"The main subject is a {clean_name}."
        descriptions[idx] = description
    return descriptions

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up directories
    output_dir = 'cifar_features'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CIFAR-100 dataset with proper class mapping
    dataset, class_names, class_to_idx, idx_to_class = load_cifar100()
    subset_dataset, selected_indices, selected_labels = get_class_samples(dataset)
    
    # Initialize models
    alexnet = models.alexnet(pretrained=True)
    alexnet.to(device)
    alexnet.eval()
    feature_extractor = FeatureExtractor(alexnet)
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate semantic descriptions
    semantic_descriptions = generate_semantic_descriptions(class_names)
    
    # Extract features and semantic embeddings
    cnn_features = []
    semantic_features = []
    metadata = []
    
    dataloader = torch.utils.data.DataLoader(subset_dataset, 
                                           batch_size=32,
                                           shuffle=False)
    
    print("Extracting features...")
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move images to device
        images = images.to(device)
        
        # Extract CNN features
        with torch.no_grad():
            _ = alexnet(images)
            batch_features = feature_extractor.features
            
            # Process each image in batch
            for idx, (feature, label) in enumerate(zip(batch_features, labels)):
                global_idx = batch_idx * 32 + idx
                label_idx = label.item()
                
                # Save CNN features
                feature_filename = f"{global_idx}_{label_idx}.pt"
                feature_path = os.path.join(output_dir, feature_filename)
                torch.save(feature.cpu(), feature_path)
                cnn_features.append(feature.cpu().numpy())
                
                # Get semantic embedding
                description = semantic_descriptions[label_idx]
                semantic_embedding = sbert_model.encode(description)
                semantic_features.append(semantic_embedding)
                
                # Store metadata with correct class name
                metadata.append({
                    'index': selected_indices[global_idx],  # Store original dataset index
                    'class_id': label_idx,
                    'class_name': idx_to_class[label_idx],
                    'description': description,
                    'feature_file': feature_filename
                })
        
        if batch_idx % 10 == 0:
            print(f"Processed batch {batch_idx}")
    
    # Save metadata and semantic features
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    semantic_features = np.array(semantic_features)
    np.save(os.path.join(output_dir, 'semantic_features.npy'), semantic_features)
    
    print("Feature extraction complete!")
    feature_extractor.remove_hook()

if __name__ == "__main__":
    main() 