from  utils import get_model_accuracy
from model import MNISTModel
import os
import torch

def model_accuracy():
    model = MNISTModel()
    
    # Look in the models directory
    model_path = 'models'
        
    model_files = [f for f in os.listdir(model_path) if f.startswith('model_mnist_') and f.endswith('.pth')]
    if not model_files:
        print("No trained model found in models directory")
    
    # Load the latest model
    latest_model = max(model_files)
    model_file_path = os.path.join(model_path, latest_model)
    print(f"Loading model from: {model_file_path}")  # Debug print
    
    try:
        model.load_state_dict(torch.load(model_file_path))
        accuracy = get_model_accuracy(model, train = True, augment=True)
        print(f'accuracy:{accuracy}')
        assert accuracy > 95, f"Model accuracy is {accuracy}%, should be > 95%"
    except Exception as e:
        print(f'exception: {e}')

if __name__ == "__main__":
    model_accuracy()