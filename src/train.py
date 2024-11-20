import torch
import torch.nn as nn
import torch.optim as optim
from model import MNISTModel
from data_loader import get_mnist_loader
from datetime import datetime
import os

def train():
    # Set device
    device = torch.device("cpu")
    
    # Load MNIST dataset using the new function
    train_loader = get_mnist_loader()
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model with timestamp in models directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join('models', f'model_mnist_{timestamp}.pth')
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train() 