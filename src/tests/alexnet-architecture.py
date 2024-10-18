import torch
from torchvision import models, transforms
from PIL import Image

# Set up AlexNet pre-trained model
alexnet = models.alexnet(pretrained=True)

# Set AlexNet to evaluation mode
alexnet.eval()
print(alexnet.features)
# Transformation for input images (resize, normalize)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Hook to capture the output from a specific layer (index 5, which is the second maxpooling layer)
def hook_fn(module, input, output):
    print(f"Output shape from hooked layer (index 5): {output.shape}")

# Register the hook on the layer at index 5
hook = alexnet.features[5].register_forward_hook(hook_fn)

# Load an example image (Make sure the image is available in the current directory)
img = Image.open(r'C:\Users\kyjer\Documents\david\dops-rv.exp\images\00ecc67a6c3a048c.jpg').convert('RGB')

# Preprocess the image
img_preprocessed = preprocess(img)
img_batch = img_preprocessed.unsqueeze(0)  # Add batch dimension

# Perform a forward pass through the network
with torch.no_grad():
    alexnet(img_batch)

# Remove the hook after testing
hook.remove()
