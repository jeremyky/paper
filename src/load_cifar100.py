import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np

def load_cifar100(root='./data'):
    """
    Load CIFAR-100 dataset and return the test set with class names
    """
    # Define the same transforms as used in CNN feature extraction
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-100 test set
    testset = torchvision.datasets.CIFAR100(root=root, 
                                          train=False,
                                          download=True, 
                                          transform=transform)
    
    # Get class names and create a mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(testset.classes)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    return testset, testset.classes, class_to_idx, idx_to_class

def get_class_samples(dataset, num_classes=100, samples_per_class=6):
    """
    Get a balanced subset of images from each class
    Returns:
        subset_dataset: Subset of the original dataset
        selected_indices: List of selected indices
        selected_labels: List of labels for selected indices
    """
    # Get indices for each class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Sample equal number from each class
    selected_indices = []
    selected_labels = []
    for class_idx in class_indices:
        indices = np.random.choice(class_indices[class_idx], 
                                 size=samples_per_class, 
                                 replace=False)
        selected_indices.extend(indices)
        selected_labels.extend([class_idx] * samples_per_class)
    
    return Subset(dataset, selected_indices), selected_indices, selected_labels 