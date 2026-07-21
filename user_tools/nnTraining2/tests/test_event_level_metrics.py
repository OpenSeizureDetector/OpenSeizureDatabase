#!/usr/bin/env python3
"""
Test script: Analyze validation data with event-level metrics.

Shows how to:
1. Load validation data
2. Get model predictions
3. Calculate event-level metrics
4. Compare to datapoint-level metrics
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse

# Add to path
sys.path.insert(0, os.path.dirname(__file__))

from eventLevelMetrics import calculate_event_level_metrics, compare_metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze validation data with event-level metrics")
    parser.add_argument('val_csv', nargs='?', default=None, help='Path to validation CSV file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Probability threshold for positive class (default: 0.5)')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with synthetic data')
    args = parser.parse_args()
    
    if args.demo or args.val_csv is None:
        run_demo()
    else:
        analyze_file(args.val_csv, args.threshold)


def run_demo():
    """Run demo with synthetic validation data."""
    print("\n" + "="*80)
    print("DEMO: Event-Level Metrics Calculation")
    print("="*80 + "\n")
    
    # Create synthetic validation data
    print("Creating synthetic validation data...")
    
    # Scenario: 10 seizure events, 10 non-seizure events
    # Each event has 100-150 datapoints
    
    np.random.seed(42)
    
    data = []
    event_id = 0
    
    # Seizure events: model correctly detects 40% of datapoints
    for i in range(10):
        event_id += 1
        n_points = np.random.randint(100, 150)
        
        # 40% of datapoints detected as seizure (TPR)
        n_detected = int(n_points * 0.40)
        predictions = np.concatenate([
            np.ones(n_detected),
            np.zeros(n_points - n_detected)
        ])
        np.random.shuffle(predictions)
        
        for j, pred in enumerate(predictions):
            data.append({
                'eventId': f'seizure_{event_id}',
                'label': 1,
                'prediction': pred if pred == 1 else np.random.uniform(0, 0.3)
            })
    
    # Non-seizure events: model incorrectly detects 2% (FPR)
    for i in range(10):
        event_id += 1
        n_points = np.random.randint(100, 150)
        
        # 2% false alarm rate
        n_false_alarms = max(1, int(n_points * 0.02))
        predictions = np.concatenate([
            np.ones(n_false_alarms),
            np.zeros(n_points - n_false_alarms)
        ])
        np.random.shuffle(predictions)
        
        for j, pred in enumerate(predictions):
            data.append({
                'eventId': f'non_seizure_{event_id}',
                'label': 0,
                'prediction': pred if pred == 1 else np.random.uniform(0, 0.3)
            })
    
    val_df = pd.DataFrame(data)
    val_predictions = val_df['prediction'].values
    val_df = val_df[['eventId', 'label']]
    
    print(f"Created {len(val_df)} datapoints across {val_df['eventId'].nunique()} events")
    print(f"  - {np.sum(val_df['label'] == 1)} seizure datapoints")
    print(f"  - {np.sum(val_df['label'] == 0)} non-seizure datapoints\n")
    
    # Calculate metrics
    event_metrics = calculate_event_level_metrics(val_df, val_predictions)
    
    print("RESULTS")
    print("-" * 80)
    print(f"Seizure events: {event_metrics['n_seizure_events']}")
    print(f"  - Correctly detected (TP): {event_metrics['event_tp']}")
    print(f"  - Missed (FN): {event_metrics['event_fn']}")
    print(f"Non-seizure events: {event_metrics['n_non_seizure_events']}")
    print(f"  - Correctly identified (TN): {event_metrics['event_tn']}")
    print(f"  - False alarms (FP): {event_metrics['event_fp']}\n")
    
    compare_metrics(
        event_metrics['datapoint_tpr'],
        event_metrics['datapoint_fpr'],
        event_metrics['event_tpr'],
        event_metrics['event_fpr']
    )
    
    print("INTERPRETATION")
    print("-" * 80)
    print("Notice how:")
    print(f"  - Datapoint TPR {event_metrics['datapoint_tpr']*100:.1f}% ➔ Event TPR {event_metrics['event_tpr']*100:.1f}%")
    print(f"    (even though we're only detecting {event_metrics['datapoint_tpr']*100:.0f}% of datapoints,")
    print(f"     having 100+ chances per event means we catch all/most events!)")
    print(f"\n  - Datapoint FPR {event_metrics['datapoint_fpr']*100:.2f}% ≈ Event FPR {event_metrics['event_fpr']*100:.2f}%")
    print(f"    (false alarms are independent, so FPR stays similar)")
    print("\n" + "="*80 + "\n")


def analyze_file(csv_path, threshold):
    """Analyze actual validation CSV file."""
    
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\nLoading validation data from {csv_path}...")
    val_df = pd.read_csv(csv_path)
    
    # Check required columns
    if 'eventId' not in val_df.columns:
        print("ERROR: CSV must have 'eventId' column")
        print(f"Available columns: {list(val_df.columns)}")
        sys.exit(1)
    
    if 'label' not in val_df.columns and 'true_label' not in val_df.columns:
        print("ERROR: CSV must have 'label' or 'true_label' column")
        sys.exit(1)
    
    # Look for predictions column
    pred_col = None
    for col in ['seizure_prob', 'model_prob', 'prob', 'prediction', 'predicted_prob']:
        if col in val_df.columns:
            pred_col = col
            break
    
    if pred_col is None:
        print("ERROR: Could not find predictions column")
        print(f"Available columns: {list(val_df.columns)}")
        print("Expected one of: seizure_prob, model_prob, prob, prediction, predicted_prob")
        sys.exit(1)
    
    print(f"Found predictions in column: {pred_col}")
    
    # Rename label column if needed
    if 'label' not in val_df.columns:
        val_df = val_df.rename(columns={'true_label': 'label'})
    
    val_predictions = val_df[pred_col].values
    val_df = val_df[['eventId', 'label']]
    
    print(f"Loaded {len(val_df)} datapoints across {val_df['eventId'].nunique()} events\n")
    
    # Calculate metrics
    event_metrics = calculate_event_level_metrics(val_df, val_predictions)
    
    print("RESULTS")
    print("-" * 80)
    print(f"Total events: {event_metrics['n_events']}")
    print(f"  Seizure events: {event_metrics['n_seizure_events']}")
    print(f"    - Correctly detected (TP): {event_metrics['event_tp']}")
    print(f"    - Missed (FN): {event_metrics['event_fn']}")
    print(f"  Non-seizure events: {event_metrics['n_non_seizure_events']}")
    print(f"    - Correctly identified (TN): {event_metrics['event_tn']}")
    print(f"    - False alarms (FP): {event_metrics['event_fp']}\n")
    
    compare_metrics(
        event_metrics['datapoint_tpr'],
        event_metrics['datapoint_fpr'],
        event_metrics['event_tpr'],
        event_metrics['event_fpr']
    )


if __name__ == '__main__':
    main()
