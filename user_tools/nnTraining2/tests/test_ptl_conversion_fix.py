#!/usr/bin/env python3
"""
Test script to verify .pt to .ptl conversion works with new dropout parameters.
"""

import torch
import sys
import os

# Add to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepEpiCnnModel_torch import DeepEpiCnn
from convertPt2Ptl import convert_pt_to_ptl

def test_conversion():
    """Test that conversion works with new dropout parameters."""
    
    print("="*80)
    print("Testing .pt to .ptl conversion with new dropout parameters")
    print("="*80)
    
    # Create a test model with dropout values
    print("\n1. Creating test model...")
    model = DeepEpiCnn(
        input_length=750,
        num_classes=2,
        conv_dropout=0.1,
        dense_dropout=0.05
    )
    model.eval()
    print(f"   Model created with conv_dropout=0.1, dense_dropout=0.05")
    
    # Save as checkpoint (simulating training checkpoint format)
    test_pt_path = '/tmp/test_model.pt'
    print(f"\n2. Saving test checkpoint to {test_pt_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'conv_dropout': 0.1,
        'dense_dropout': 0.05,
        'epoch': 100
    }, test_pt_path)
    print("   ✓ Checkpoint saved")
    
    # Try conversion
    test_ptl_path = '/tmp/test_model.ptl'
    print(f"\n3. Converting to .ptl format...")
    success = convert_pt_to_ptl(
        input_path=test_pt_path,
        output_path=test_ptl_path,
        input_shape=(1, 1, 750),
        num_classes=2,
        verbose=True
    )
    
    if success:
        print("\n" + "="*80)
        print("✓ SUCCESS: Conversion completed without errors!")
        print("="*80)
        print("\nThe fix is working correctly. The conversion script can now:")
        print("  1. Load checkpoints with conv_dropout and dense_dropout")
        print("  2. Create models with the correct parameter names")
        print("  3. Convert successfully to .ptl format")
        
        # Cleanup
        os.remove(test_pt_path)
        os.remove(test_ptl_path)
        print("\nTest files cleaned up.")
        return True
    else:
        print("\n" + "="*80)
        print("✗ FAILED: Conversion encountered errors")
        print("="*80)
        return False

if __name__ == '__main__':
    success = test_conversion()
    sys.exit(0 if success else 1)
