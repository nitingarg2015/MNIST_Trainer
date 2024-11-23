from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def show_augmented_samples(num_samples=5):
    """
    Display original and augmented versions of sample images
    Args:
        num_samples (int): Number of samples to display
    """
    # Define transforms
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    augment_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 0.2)
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load dataset
    dataset = datasets.MNIST('./data', train=True, download=True, transform=None)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    fig.suptitle('MNIST Augmentation Samples', fontsize=16)
    
    for idx in range(num_samples):
        # Get a random image
        image, label = dataset[np.random.randint(len(dataset))]
        
        # Original image
        axes[idx, 0].imshow(image, cmap='gray')
        axes[idx, 0].set_title(f'Original (Label: {label})')
        axes[idx, 0].axis('off')
        
        # Apply basic transform (just normalization)
        basic_img = basic_transform(image)
        basic_img = basic_img.squeeze().numpy()
        axes[idx, 1].imshow(basic_img, cmap='gray')
        axes[idx, 1].set_title('Basic Transform')
        axes[idx, 1].axis('off')
        
        # Apply augmentation
        aug_img = augment_transform(image)
        aug_img = aug_img.squeeze().numpy()
        axes[idx, 2].imshow(aug_img, cmap='gray')
        axes[idx, 2].set_title('Augmented')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Show sample augmentations when run directly
    show_augmented_samples() 