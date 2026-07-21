#!/usr/bin/env python3
"""
Test and demonstrate the new model selection criteria.

This script shows how different model selection metrics work and helps
choose appropriate settings for your training configuration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2.nnTrainer import calculate_selection_metric


def demo_model_selection():
    """Demonstrate different model selection criteria."""
    
    print("="*80)
    print("Model Selection Criteria Comparison")
    print("="*80)
    print()
    
    # Example scenarios: (sensitivity, FPR, description)
    scenarios = [
        (0.95, 0.30, "High sensitivity, high FPR (catches all seizures but many false alarms)"),
        (0.85, 0.10, "Good sensitivity, low FPR (good balance)"),
        (0.75, 0.05, "Moderate sensitivity, very low FPR (few false alarms, misses some seizures)"),
        (0.60, 0.02, "Low sensitivity, very low FPR (too conservative)"),
        (0.90, 0.20, "High sensitivity, moderate FPR"),
        (0.80, 0.15, "Balanced performance"),
    ]
    
    print("Scenario Analysis:")
    print("-" * 80)
    print(f"{'Sensitivity':<12} {'FPR':<8} {'Youden':<10} {'F1':<10} {'F-beta(2)':<12} {'Description':<35}")
    print("-" * 80)
    
    for sens, fpr, desc in scenarios:
        youden = calculate_selection_metric(sens, fpr, 'youden')
        f1 = calculate_selection_metric(sens, fpr, 'f1')
        f_beta = calculate_selection_metric(sens, fpr, 'f_beta', beta=2.0)
        
        print(f"{sens:<12.3f} {fpr:<8.3f} {youden:<10.3f} {f1:<10.3f} {f_beta:<12.3f} {desc:<35}")
    
    print()
    print("="*80)
    print("Metric Interpretations:")
    print("="*80)
    print()
    print("1. Youden's J Statistic (TPR - FPR)")
    print("   - Range: -1 to 1 (higher is better)")
    print("   - Optimal balance between sensitivity and specificity")
    print("   - Best for finding the 'sweet spot' on ROC curve")
    print()
    print("2. F1 Score")
    print("   - Range: 0 to 1 (higher is better)")
    print("   - Harmonic mean of sensitivity and specificity")
    print("   - Equal weighting of sensitivity and false alarm rate")
    print()
    print("3. F-beta Score (beta=2.0)")
    print("   - Range: 0 to 1 (higher is better)")
    print("   - Weighted harmonic mean favoring sensitivity when beta > 1")
    print("   - beta=2.0: sensitivity weighted 2x more than specificity")
    print("   - beta=0.5: specificity weighted 2x more than sensitivity")
    print("   - Recommended for medical applications where missing events is costly")
    print()
    print("="*80)
    print("Configuration Recommendations:")
    print("="*80)
    print()
    print("For Seizure Detection (prioritize catching seizures):")
    print("  modelSelectionMetric: 'f_beta'")
    print("  fBeta: 2.0  # Favors sensitivity over low FPR")
    print("  saveBestMaxFpr: 0.15  # Still limit max FPR to avoid too many alarms")
    print("  saveBestMinSensitivity: 0.70  # Ensure we catch most seizures")
    print()
    print("For Balanced Performance:")
    print("  modelSelectionMetric: 'youden'  # or 'f1'")
    print("  saveBestMaxFpr: 0.10")
    print("  saveBestMinSensitivity: 0.80")
    print()
    print("For Minimizing False Alarms:")
    print("  modelSelectionMetric: 'f_beta'")
    print("  fBeta: 0.5  # Favors specificity (low FPR)")
    print("  saveBestMaxFpr: 0.05")
    print("  saveBestMinSensitivity: 0.60")
    print()
    print("For Legacy Behavior (Spahr et al. 2025):")
    print("  modelSelectionMetric: 'dual_improvement'")
    print("  saveBestMaxFpr: null  # No hard limit")
    print("="*80)


def test_max_fpr_enforcement():
    """Demonstrate max FPR threshold enforcement."""
    
    print("\n" + "="*80)
    print("Maximum FPR Threshold Enforcement")
    print("="*80)
    print()
    print("When saveBestMaxFpr is set (e.g., 0.15), models with FPR > 0.15 will")
    print("NEVER be saved, regardless of how high their sensitivity is.")
    print()
    print("Example with saveBestMaxFpr=0.15:")
    print("-" * 80)
    print(f"{'Sensitivity':<12} {'FPR':<8} {'Would Save?':<15} {'Reason':<40}")
    print("-" * 80)
    
    max_fpr = 0.15
    min_sens = 0.25
    
    test_cases = [
        (0.95, 0.30, False, "FPR exceeds max threshold"),
        (0.85, 0.10, True, "Within limits, good balance"),
        (0.90, 0.15, True, "At max FPR threshold, high sensitivity"),
        (0.95, 0.16, False, "FPR slightly exceeds threshold"),
        (0.20, 0.05, False, "Low FPR but sensitivity below minimum"),
        (0.80, 0.14, True, "Good balance within all limits"),
    ]
    
    for sens, fpr, should_save, reason in test_cases:
        exceeds_fpr = fpr > max_fpr
        below_sens = sens < min_sens
        
        if exceeds_fpr or below_sens:
            would_save = "❌ NO"
        else:
            would_save = "✓ YES"
        
        print(f"{sens:<12.3f} {fpr:<8.3f} {would_save:<15} {reason:<40}")
    
    print()
    print("This prevents the common problem of saving models with high sensitivity")
    print("but unacceptably high false alarm rates.")
    print("="*80)


if __name__ == '__main__':
    demo_model_selection()
    test_max_fpr_enforcement()
