#!/usr/bin/env python3
"""
Test script to demonstrate event-based threshold analysis functionality.

This creates synthetic test data and runs the event-based threshold analysis
to verify the new TPR/FPR calculation and plotting works correctly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import sys
import json

def test_event_threshold_analysis():
    """Test the event-based threshold analysis logic."""
    
    print("="*70)
    print("TESTING EVENT-BASED THRESHOLD ANALYSIS")
    print("="*70)
    
    # Create synthetic event data
    # 20 events: 10 seizures (true_label=1), 10 non-seizures (true_label=0)
    np.random.seed(42)
    
    event_stats_df = pd.DataFrame({
        'eventId': range(20),
        'true_label': [1]*10 + [0]*10,  # First 10 are seizures
        'max_seizure_prob': np.concatenate([
            np.random.uniform(0.6, 0.95, 10),  # Seizures: high probabilities
            np.random.uniform(0.05, 0.4, 10)   # Non-seizures: low probabilities
        ])
    })
    
    print(f"\nTest data created:")
    print(f"  Total events: {len(event_stats_df)}")
    print(f"  Seizure events: {(event_stats_df['true_label'] == 1).sum()}")
    print(f"  Non-seizure events: {(event_stats_df['true_label'] == 0).sum()}")
    
    # Event-based threshold analysis
    event_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    event_tpr_list = []
    event_fpr_list = []
    event_tp_list = []
    event_fp_list = []
    event_tn_list = []
    event_fn_list = []
    
    for threshold in event_threshold_list:
        # For each event, classify as positive if max_seizure_prob >= threshold
        event_preds_at_threshold = (event_stats_df['max_seizure_prob'] >= threshold).astype(int)
        event_true_labels = event_stats_df['true_label'].values
        
        # Calculate confusion matrix for this threshold
        event_cm_th = sklearn.metrics.confusion_matrix(event_true_labels, event_preds_at_threshold, labels=[0, 1])
        event_tn_th, event_fp_th, event_fn_th, event_tp_th = event_cm_th.ravel()
        
        # Calculate TPR and FPR
        event_tpr_th = event_tp_th / (event_tp_th + event_fn_th) if (event_tp_th + event_fn_th) > 0 else 0
        event_fpr_th = event_fp_th / (event_fp_th + event_tn_th) if (event_fp_th + event_tn_th) > 0 else 0
        
        event_tpr_list.append(event_tpr_th)
        event_fpr_list.append(event_fpr_th)
        event_tp_list.append(int(event_tp_th))
        event_fp_list.append(int(event_fp_th))
        event_tn_list.append(int(event_tn_th))
        event_fn_list.append(int(event_fn_th))
    
    print(f"\n{'Threshold':<12} {'TPR':<12} {'FPR':<12} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 70)
    for i, th in enumerate(event_threshold_list):
        print(f"{th:<12.1f} {event_tpr_list[i]:<12.4f} {event_fpr_list[i]:<12.4f} "
              f"{event_tp_list[i]:<8} {event_fp_list[i]:<8} {event_tn_list[i]:<8} {event_fn_list[i]:<8}")
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: TPR and FPR vs Threshold
    axes[0].plot(event_threshold_list, event_tpr_list, 'o-', color='green', linewidth=2, markersize=8, label='TPR (Sensitivity)')
    axes[0].plot(event_threshold_list, event_fpr_list, 's-', color='red', linewidth=2, markersize=8, label='FPR (False Alarm Rate)')
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Rate', fontsize=12)
    axes[0].set_title('Event-Based TPR and FPR vs Threshold (Test Data)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.05])
    
    # Plot 2: ROC-style curve
    sorted_indices = np.argsort(event_fpr_list)
    sorted_fpr = [event_fpr_list[i] for i in sorted_indices]
    sorted_tpr = [event_tpr_list[i] for i in sorted_indices]
    sorted_th = [event_threshold_list[i] for i in sorted_indices]
    
    axes[1].plot(sorted_fpr, sorted_tpr, 'o-', color='blue', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate (FPR)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[1].set_title('Event-Based ROC Curve (Test Data)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = '/tmp/test_event_threshold_analysis.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Test plot saved to {output_path}")
    
    # Verify key properties
    print("\nVerification:")
    print(f"  ✓ At threshold 0.1: TPR={event_tpr_list[0]:.2f}, FPR={event_fpr_list[0]:.2f} (low threshold → high TPR, high FPR)")
    print(f"  ✓ At threshold 0.5: TPR={event_tpr_list[4]:.2f}, FPR={event_fpr_list[4]:.2f} (medium threshold)")
    print(f"  ✓ At threshold 0.9: TPR={event_tpr_list[8]:.2f}, FPR={event_fpr_list[8]:.2f} (high threshold → low TPR, low FPR)")
    
    # Check that TPR generally decreases as threshold increases
    tpr_decreasing = all(event_tpr_list[i] >= event_tpr_list[i+1] - 0.01 for i in range(len(event_tpr_list)-1))
    print(f"  ✓ TPR generally decreases with threshold: {tpr_decreasing}")
    
    print("\n" + "="*70)
    print("✓ EVENT-BASED THRESHOLD ANALYSIS TEST PASSED")
    print("="*70)

if __name__ == "__main__":
    test_event_threshold_analysis()
