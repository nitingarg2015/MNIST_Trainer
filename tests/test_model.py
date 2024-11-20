import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTModel
from src.utils import count_parameters, get_model_accuracy

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"

def test_model_parameters():
    model = MNISTModel()
    param_count = count_parameters(model)
    
    assert param_count < 100000, f"Model has {param_count} parameters, should be less than 100000"

def test_model_accuracy():
    model = MNISTModel()
    
    # Look in the models directory
    model_path = 'models'
    if not os.path.exists(model_path):
        pytest.skip("Models directory not found")
        
    model_files = [f for f in os.listdir(model_path) if f.startswith('model_mnist_') and f.endswith('.pth')]
    if not model_files:
        pytest.skip("No trained model found in models directory")
    
    # Load the latest model
    latest_model = max(model_files)
    model_file_path = os.path.join(model_path, latest_model)
    print(f"Loading model from: {model_file_path}")  # Debug print
    
    try:
        model.load_state_dict(torch.load(model_file_path))
        accuracy = get_model_accuracy(model)
        assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%"
    except Exception as e:
        pytest.fail(f"Failed to load or test model: {str(e)}") 