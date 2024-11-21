import torch
from torchvision import models, transforms
from PIL import Image
import os
import json

# Get the current script's directory
script_dir = os.path.dirname(__file__)

# Set up AlexNet pre-trained model
alexnet = models.alexnet(pretrained=True)

# Set AlexNet to evaluation mode
alexnet.eval()

# Transformation for input images (resize, normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a hook to capture the output of the last maxpool layer
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        # Register the hook on the last maxpool layer (before the classifier)
        self.hook = model.features[12].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output  # Capture the features

    def remove_hook(self):
        self.hook.remove()

# Function to extract features from a single image using the hook
def extract_features(image_path, feature_extractor):
    img = Image.open(image_path).convert('RGB')
    img_preprocessed = preprocess(img)
    img_batch = img_preprocessed.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No gradient calculation needed
        _ = alexnet(img_batch)  # Perform forward pass

    features = feature_extractor.features
    return features  # Return the raw feature tensor

# Extract and save features from a directory of images, one per file
def extract_and_save_features(image_directory, feature_extractor, output_directory, max_images=100):
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.jpeg', '.png', '.thl'))]
    feature_mapping = {}  # Dictionary to store feature file to image file mapping
    
    # Limit to max_images images or fewer if the directory contains fewer than max_images
    for img_file in image_files[:max_images]:
        img_path = os.path.join(image_directory, img_file)
        
        try:
            # Extract features
            features = extract_features(img_path, feature_extractor)
            
            # Get base name and create filenames
            base_name = os.path.splitext(img_file)[0]
            feature_filename = base_name + ".pt"
            output_path = os.path.join(output_directory, feature_filename)
            
            # Save the feature vector
            torch.save(features, output_path)
            
            # Store the mapping
            feature_mapping[base_name] = {
                'image_file': img_file,
                'feature_file': feature_filename
            }
            
            print(f"Saved features for {img_file} to {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # Save the mapping to a JSON file
    mapping_file = os.path.join(output_directory, 'feature_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(feature_mapping, f, indent=4)
    
    print(f"Saved feature to image mapping in {mapping_file}")
    return feature_mapping

# Initialize feature extractor for the last maxpool layer
feature_extractor = FeatureExtractor(alexnet)

# Update the image directory path to point to ObjectsAll/OBJECTSALL
image_directory = os.path.join(os.path.dirname(script_dir), 'ObjectsAll', 'OBJECTSALL')
output_directory = os.path.join(script_dir, 'unique_objects_features')

# Add directory checks and better error messages
if not os.path.isdir(image_directory):
    raise FileNotFoundError(f"Image directory not found at: {image_directory}\n"
                          f"Please ensure the ObjectsAll folder is in the correct location.")

print(f"Using image directory: {image_directory}")
print(f"Using output directory: {output_directory}")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Extract features from the images in the directory and save each as a separate .pt file
feature_mapping = extract_and_save_features(image_directory, feature_extractor, output_directory, max_images=2400)

# Remove the hook when done
feature_extractor.remove_hook() 