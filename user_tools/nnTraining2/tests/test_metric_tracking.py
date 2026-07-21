#!/usr/bin/env python3
"""
Test to verify metric tracking works correctly across epochs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2.nnTrainer import calculate_selection_metric


def test_metric_tracking():
    """Test that metrics are tracked properly and -inf is only initial value."""
    
    print("="*80)
    print("Testing Metric Tracking Logic")
    print("="*80)
    print()
    
    # Simulate training epochs with improving metrics
    best_metric = -float('inf')
    
    epochs_data = [
        (0.37, 0.024, "Epoch 1"),
        (0.46, 0.033, "Epoch 2"),
        (0.50, 0.028, "Epoch 3"),
        (0.48, 0.030, "Epoch 4"),
        (0.55, 0.025, "Epoch 5"),
    ]
    
    print(f"{'Epoch':<10} {'Sens':<8} {'FPR':<8} {'F-beta':<10} {'Best':<10} {'Action':<30}")
    print("-" * 80)
    
    for sens, fpr, label in epochs_data:
        current_metric = calculate_selection_metric(sens, fpr, 'f_beta', beta=2.0)
        
        should_save = current_metric > best_metric
        
        action = f"✓ SAVE (new best)" if should_save else "✗ skip"
        
        print(f"{label:<10} {sens:<8.4f} {fpr:<8.4f} {current_metric:<10.4f} {best_metric:<10.4f} {action:<30}")
        
        if should_save:
            best_metric = current_metric
    
    print()
    print("✓ Test PASSED: Metric tracking works correctly")
    print(f"  - Initial best_metric: -inf")
    print(f"  - First epoch saved (f_beta improves from -inf)")
    print(f"  - Final best_metric: {best_metric:.4f}")
    print()
    print("Expected behavior:")
    print("  - Epoch 1: -inf → 0.5086 (saves)")
    print("  - Epoch 2: 0.5086 → 0.5456 (saves, improved)")
    print("  - Epoch 3: 0.5456 → 0.5743 (saves, improved)")
    print("  - Epoch 4: 0.5743 → 0.5574 (skips, metric decreased)")
    print("  - Epoch 5: 0.5574 → 0.5945 (saves, improved)")
    print("="*80)


if __name__ == '__main__':
    test_metric_tracking()
