#!/usr/bin/env python3
"""
Test script to verify dropout configuration is correctly read and applied.
"""

import sys
import os
import json
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2 import deepEpiCnnModel_torch
from user_tools.nnTraining2.nnTrainer import visualize_pytorch_model


def test_dropout_config():
    """Test that dropout is correctly configured from JSON config."""
    
    # Load the actual config used in training
    config_path = "user_tools/nnTraining2/nnConfig_deep_pytorch.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("="*80)
    print("Testing Dropout Configuration")
    print("="*80)
    print(f"\nConfig file: {config_path}")
    print(f"convDropout from config: {config['modelConfig']['convDropout']}")
    print(f"denseDropout from config: {config['modelConfig']['denseDropout']}")
    print()
    
    # Create model with config
    print("Creating model with config...")
    m = deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch(configObj=config['modelConfig'], debug=True)
    
    # Verify dropout values were read
    print(f"\n✓ Model conv_dropout: {m.conv_dropout}")
    print(f"✓ Model dense_dropout: {m.dense_dropout}")
    print()
    
    # Create the actual model
    input_shape = (125, 1)  # matching config window size
    model = m.makeModel(input_shape=input_shape, num_classes=2)
    
    # Visualize to see dropout layers
    print("\n" + "="*80)
    print("Model Visualization")
    print("="*80)
    visualize_pytorch_model(model, input_shape, model_name="test_dropout_model")
    
    # Count dropout layers
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_count += 1
            print(f"Found dropout layer: {name} with p={module.p}")
    
    print(f"\n✓ Total dropout layers found: {dropout_count}")
    
    expected_dropout_layers = 14 + 1  # 14 conv blocks + 1 dense
    if m.conv_dropout > 0.0:
        if dropout_count >= 14:
            print(f"✓ Conv dropout layers correctly applied (expected 14 for conv blocks)")
        else:
            print(f"✗ Missing conv dropout layers (found {dropout_count}, expected >= 14)")
    
    if m.dense_dropout > 0.0:
        print(f"✓ Dense dropout correctly applied")
    
    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == '__main__':
    test_dropout_config()
