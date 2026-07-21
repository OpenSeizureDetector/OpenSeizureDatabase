#!/usr/bin/env python3
"""
Test script for PyTorch model visualization.

This script demonstrates the new model visualization functionality
that was added to match the TensorFlow version's keras.utils.plot_model() feature.
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2 import deepEpiCnnModel_torch
from user_tools.nnTraining2.nnTrainer import visualize_pytorch_model


def test_model_visualization():
    """Test the model visualization with a sample model."""
    print("="*80)
    print("Testing PyTorch Model Visualization")
    print("="*80)
    
    # Create a sample model
    config = {
        'sampleFreq': 25,
        'freqCutoff': 12,
        'sdThresh': 75,
        'psdThresh': 0,
        'movingAvLen': 600,
        'nMin': 3,
        'nMax': 3,
        'warnTime': 10,
        'alarmFreqMin': 3,
        'alarmFreqMax': 8,
        'alarmThresh': 100,
        'alarmRatioThresh': 57,
        'rawDataInputFname': 'input_file.csv',
        'window': 750,
        'framework': 'pytorch'
    }
    
    print("\nCreating DeepEpiCnn model...")
    m = deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch(configObj=config, debug=True)
    model = m.makeModel(input_shape=(750, 1), num_classes=2)
    
    print(f"\nModel created successfully!")
    print(f"Device: {m.device}")
    
    # Test the visualization function
    print("\n" + "="*80)
    print("Calling visualize_pytorch_model()...")
    print("="*80)
    visualize_pytorch_model(model, input_shape=(750, 1), model_name="test_model")
    
    # Test forward pass
    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)
    batch_size = 2
    test_input = torch.randn(batch_size, 750, 1).to(m.device)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output (logits): {output}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    
    # Check if visualization files were created
    if os.path.exists("test_model_architecture.png"):
        print("\n✓ Visual diagram created: test_model_architecture.png")
    else:
        print("\nNote: Visual diagram not created (install torchviz for this feature)")
    
    return True


if __name__ == '__main__':
    success = test_model_visualization()
    sys.exit(0 if success else 1)
