import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def get_mnist_loader(augment=True, train = False):
    """
    Get MNIST data loader with appropriate augmentations for digit recognition
    Args:
        augment (bool): If True, apply data augmentation
    """
    if augment:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=5),      # Slight rotation (reduced from 15 to 10 degrees)
            transforms.RandomAffine(
                degrees=0,                             # No rotation in affine transform
                translate=(0.05, 0.05),                  # Random translation up to 10%
                scale=(0.95, 1.05)                       # Random scaling between 90% and 110%
            ),
            transforms.GaussianBlur(
                kernel_size=3,                         # Blur kernel size
                sigma=(0.01, 0.02)                       # Random sigma for blur
            ),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        # Original transform without augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset = datasets.MNIST('./data', train=train, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    return data_loader 


def get_model_accuracy(model, train = False, augment = False, device="cpu"):
    model.eval()
    
    dataloader = get_mnist_loader(augment=augment, train = False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
    
    return 100 * correct / total

def count_parameters(model):
    # print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters')
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

