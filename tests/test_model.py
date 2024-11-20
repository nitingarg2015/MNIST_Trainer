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
    # Load the latest trained model
    model_files = [f for f in os.listdir('.') if f.startswith('model_mnist_') and f.endswith('.pth')]
    if not model_files:
        pytest.skip("No trained model found")
    
    latest_model = max(model_files)
    model.load_state_dict(torch.load(latest_model))
    
    accuracy = get_model_accuracy(model)
    assert accuracy > 80, f"Model accuracy is {accuracy}%, should be > 80%" 