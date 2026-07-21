#!/usr/bin/env python3
"""
Compare your two models and understand F-beta scoring.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from user_tools.nnTraining2.nnTrainer import calculate_selection_metric


def compare_models():
    """Compare the two models mentioned by the user."""
    
    print("="*80)
    print("Comparing Your Two Models")
    print("="*80)
    print()
    
    model_1 = (0.47, 0.03, "Current model")
    model_2 = (0.42, 0.019, "Alternative model (preferred)")
    
    print(f"{'Model':<25} {'TPR':<8} {'FPR':<8} {'Specificity':<12}")
    print("-" * 60)
    print(f"{model_1[2]:<25} {model_1[0]:<8.3f} {model_1[1]:<8.3f} {1-model_1[1]:<12.4f}")
    print(f"{model_2[2]:<25} {model_2[0]:<8.3f} {model_2[1]:<8.3f} {1-model_2[1]:<12.4f}")
    print()
    
    # Calculate metrics for different beta values
    beta_values = [0.25, 0.5, 1.0, 2.0, 4.0]
    
    print("F-Beta Scores (higher is better):")
    print("-" * 80)
    print(f"{'Beta':<8} {'Metric':<15} {'Model 1 (0.47/0.03)':<20} {'Model 2 (0.42/0.019)':<20} {'Winner':<15}")
    print("-" * 80)
    
    for beta in beta_values:
        m1_score = calculate_selection_metric(model_1[0], model_1[1], 'f_beta', beta)
        m2_score = calculate_selection_metric(model_2[0], model_2[1], 'f_beta', beta)
        
        winner = "Model 1" if m1_score > m2_score else "Model 2" if m2_score > m1_score else "Tie"
        diff = abs(m1_score - m2_score)
        
        print(f"{beta:<8.2f} F-beta        {m1_score:<20.4f} {m2_score:<20.4f} {winner:<15} (Δ={diff:.4f})")
    
    print()
    print("Other Metrics:")
    print("-" * 80)
    
    youden_m1 = calculate_selection_metric(model_1[0], model_1[1], 'youden')
    youden_m2 = calculate_selection_metric(model_2[0], model_2[1], 'youden')
    
    f1_m1 = calculate_selection_metric(model_1[0], model_1[1], 'f1')
    f1_m2 = calculate_selection_metric(model_2[0], model_2[1], 'f1')
    
    print(f"{'Youden (TPR-FPR)':<30} Model 1: {youden_m1:.4f}  Model 2: {youden_m2:.4f}  Winner: {'Model 1' if youden_m1 > youden_m2 else 'Model 2'}")
    print(f"{'F1 Score':<30} Model 1: {f1_m1:.4f}  Model 2: {f1_m2:.4f}  Winner: {'Model 1' if f1_m1 > f1_m2 else 'Model 2'}")
    
    print()
    print("="*80)
    print("Analysis:")
    print("="*80)
    print()
    print("Model 1 (TPR=0.47, FPR=0.03) scores HIGHER on all metrics")
    print("Model 2 (TPR=0.42, FPR=0.019) scores LOWER on all metrics")
    print()
    print("However, you prefer Model 2 because:")
    print("  - 33% fewer false alarms (0.03 vs 0.019)")
    print("  - Slightly lower sensitivity (0.47 vs 0.42) is acceptable trade-off")
    print()
    print("="*80)
    print("Recommendations:")
    print("="*80)
    print()
    print("Option 1: Set a HARD FPR LIMIT (Recommended)")
    print("  - This is more intuitive and easier to control")
    print("  - Example: saveBestMaxFpr: 0.025")
    print("  - This means: 'Never save models with >2.5% false alarms'")
    print("  - Model 1 would be REJECTED (0.03 > 0.025)")
    print("  - Model 2 would be ACCEPTED (0.019 < 0.025)")
    print()
    print("Option 2: Reduce fBeta significantly")
    print("  - fBeta: 0.25 (heavily favors specificity/low FPR)")
    print("  - But this is less intuitive than a hard limit")
    print()
    print("Option 3: Switch to Youden's J")
    print("  - modelSelectionMetric: 'youden'")
    print("  - But Model 1 still wins: 0.44 > 0.401")
    print("  - Doesn't help in this case")
    print()
    print("="*80)
    print()
    print("BOTTOM LINE:")
    print("  Use saveBestMaxFpr: 0.025 (or lower)")
    print("  This is clearer and more directly expresses your preference")
    print("  than trying to tune fBeta.")
    print("="*80)


if __name__ == '__main__':
    compare_models()
