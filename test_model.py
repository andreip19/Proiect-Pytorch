import torch
from model import SimpleCNN

def test_tensor_shapes():
    model = SimpleCNN()
    dummy_input = torch.randn(2, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (2, 10)