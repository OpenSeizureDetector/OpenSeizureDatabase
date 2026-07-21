#!/usr/bin/env python3
"""
Script to visualize a saved PyTorch model from training output.

Usage:
    python visualize_saved_model.py <model_path> <config_path>

Example:
    python visualize_saved_model.py \
        output/deepEpiCnnModel_pytorch/13/outerfold0/fold2/deepEpiCnnModel_pytorch.pt \
        output/deepEpiCnnModel_pytorch/13/nnConfig_deep_pytorch.json
"""

import sys
import os
import json
import torch

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2 import deepEpiCnnModel_torch
from user_tools.nnTraining2.nnTrainer import visualize_pytorch_model


def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def visualize_model(model_path, config_path, output_name=None):
    """
    Load and visualize a saved PyTorch model.
    
    Args:
        model_path: Path to the saved .pt model file
        config_path: Path to the configuration JSON file
        output_name: Optional name for the output diagram
    """
    print("="*80)
    print(f"Visualizing PyTorch Model")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    print()
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract model configuration
    model_config = config.get('modelConfig', {})
    data_processing = config.get('dataProcessing', {})
    
    # Get input shape from config
    window = data_processing.get('window', 750)
    input_shape = (window, 1)
    
    # Get number of classes
    num_classes = model_config.get('num_classes', 2)
    
    print(f"Creating model architecture...")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    print()
    
    # Create model architecture
    m = deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch(configObj=config, debug=True)
    model = m.makeModel(input_shape=input_shape, num_classes=num_classes)
    
    # Load saved weights
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=m.device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        print(f"✓ Loaded model from epoch {epoch}")
        if val_loss != 'unknown':
            print(f"  Validation loss: {val_loss:.6f}")
    else:
        # Older format or direct state dict
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")
    
    print()
    
    # Determine output name
    if output_name is None:
        output_name = os.path.splitext(model_path)[0]
    
    # Visualize the model
    visualize_pytorch_model(model, input_shape, model_name=output_name)
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)
    
    # List output files
    diagram_png = f"{output_name}_architecture.png"
    if os.path.exists(diagram_png):
        print(f"\n✓ Diagram saved: {diagram_png}")
    
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_saved_model.py <model_path> <config_path> [output_name]")
        print()
        print("Example:")
        print("  python visualize_saved_model.py \\")
        print("    output/deepEpiCnnModel_pytorch/13/outerfold0/fold2/deepEpiCnnModel_pytorch.pt \\")
        print("    output/deepEpiCnnModel_pytorch/13/nnConfig_deep_pytorch.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    success = visualize_model(model_path, config_path, output_name)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
