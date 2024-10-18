import torch
from torchvision import models, transforms
from PIL import Image
import os

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
def extract_and_save_features(image_directory, feature_extractor, output_directory, max_images=50):
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png'))]
    
    # Limit to 50 images or fewer if the directory contains fewer than 50 images
    for img_file in image_files[:max_images]:
        img_path = os.path.join(image_directory, img_file)
        features = extract_features(img_path, feature_extractor)
        
        # Create a name for the output .pt file (e.g., "1.pt", "2.pt", based on the image filename)
        feature_filename = os.path.splitext(img_file)[0] + ".pt"
        output_path = os.path.join(output_directory, feature_filename)
        
        # Save the feature vector to a separate .pt file
        torch.save(features, output_path)
        print(f"Saved features for {img_file} to {output_path}")

# Initialize feature extractor for the last maxpool layer
feature_extractor = FeatureExtractor(alexnet)

# Specify your image directory and output directory (use your paths)
image_directory = os.path.join(script_dir, 'targets')
output_directory = os.path.join(script_dir, 'features')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Extract features from the images in the directory and save each as a separate .pt file
extract_and_save_features(image_directory, feature_extractor, output_directory, max_images=22)

# Remove the hook when done
feature_extractor.remove_hook()
