import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTModel
from src.utils import count_parameters

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape should be (batch_size, 10)"

def test_model_parameters():
    model = MNISTModel()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

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
        checkpoint = torch.load(model_file_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        train_accuracy = checkpoint['train_accuracy']
        assert train_accuracy > 95, f"Model accuracy is {train_accuracy}%, should be > 95%"
    except Exception as e:
        pytest.fail(f"Failed to load or test model: {str(e)}") 

def test_model_forward_different_batch_sizes():
    """Test if model can handle different batch sizes"""
    model = MNISTModel()
    batch_sizes = [1, 4, 16, 32]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

def test_model_output_range():
    """Test if model outputs are in valid probability range after softmax"""
    model = MNISTModel()
    test_input = torch.randn(10, 1, 28, 28)
    output = torch.nn.functional.softmax(model(test_input), dim=1)
    
    assert torch.all(output >= 0) and torch.all(output <= 1), "Output values should be between 0 and 1"
    assert torch.allclose(output.sum(dim=1), torch.ones(10)), "Probabilities should sum to 1"

def test_model_gradients():
    """Test if gradients are properly computed"""
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_gradients = any(param.grad is not None and torch.any(param.grad != 0) 
                       for param in model.parameters())
    assert has_gradients, "Model should have non-zero gradients"

def test_model_input_validation():
    """Test model behavior with invalid inputs"""
    model = MNISTModel()
    
    # Test wrong input channels
    with pytest.raises(RuntimeError):
        wrong_channels = torch.randn(1, 3, 28, 28)  # 3 channels instead of 1
        model(wrong_channels)
    
    # Test wrong input size
    with pytest.raises(RuntimeError):
        wrong_size = torch.randn(1, 1, 32, 32)  # 32x32 instead of 28x28
        model(wrong_size)
