"""
Utilities for calculating event-level metrics from datapoint-level predictions.

This module provides functions to convert datapoint-level predictions and targets
into event-level metrics, matching the approach used by nnTester.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def calculate_event_level_metrics(val_df: pd.DataFrame, val_predictions: np.ndarray) -> Dict[str, Any]:
    """
    Calculate event-level TPR/FPR from datapoint-level predictions.
    
    Uses the maximum seizure probability across all datapoints in each event
    to make the event-level prediction, matching nnTester's approach.
    
    Args:
        val_df: DataFrame with validation data (must have 'eventId' and 'label' columns)
        val_predictions: Array of predicted probabilities for class 1 (shape: N,)
                        or class predictions (shape: N,). If predictions are class 
                        labels (0/1), they're converted to probabilities.
    
    Returns:
        dict: {
            'event_tpr': float - Event-level True Positive Rate
            'event_fpr': float - Event-level False Positive Rate
            'event_tn': int - Number of correctly identified non-seizure events
            'event_tp': int - Number of correctly identified seizure events
            'event_fp': int - Number of false alarm events
            'event_fn': int - Number of missed seizure events
            'n_events': int - Total number of events
            'n_seizure_events': int - Number of seizure events
            'n_non_seizure_events': int - Number of non-seizure events
            'datapoint_tpr': float - Original datapoint-level TPR for reference
            'datapoint_fpr': float - Original datapoint-level FPR for reference
        }
    """
    # Add predictions to dataframe
    val_df_copy = val_df.copy()
    val_df_copy['predicted'] = val_predictions
    
    # Group by event and get max prediction and true label per event
    event_stats = val_df_copy.groupby('eventId').agg({
        'predicted': 'max',  # Use max prediction probability in the event
        'label': 'first'      # True label (same for all datapoints in event)
    }).reset_index()
    
    event_stats.columns = ['eventId', 'max_predicted', 'true_label']
    
    # Use probability threshold of 0.5 for event classification
    # (0.5 is standard; can be parameterized if needed)
    event_predictions = (event_stats['max_predicted'] >= 0.5).astype(int)
    
    # Calculate event-level confusion matrix
    event_tp = np.sum((event_predictions == 1) & (event_stats['true_label'] == 1))
    event_fp = np.sum((event_predictions == 1) & (event_stats['true_label'] == 0))
    event_tn = np.sum((event_predictions == 0) & (event_stats['true_label'] == 0))
    event_fn = np.sum((event_predictions == 0) & (event_stats['true_label'] == 1))
    
    # Calculate event-level TPR and FPR
    event_tpr = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0.0
    event_fpr = event_fp / (event_fp + event_tn) if (event_fp + event_tn) > 0 else 0.0
    
    # Calculate datapoint-level metrics for reference
    datapoint_tp = np.sum((val_predictions >= 0.5) & (val_df['label'].values == 1))
    datapoint_fp = np.sum((val_predictions >= 0.5) & (val_df['label'].values == 0))
    datapoint_tn = np.sum((val_predictions < 0.5) & (val_df['label'].values == 0))
    datapoint_fn = np.sum((val_predictions < 0.5) & (val_df['label'].values == 1))
    
    datapoint_tpr = datapoint_tp / (datapoint_tp + datapoint_fn) if (datapoint_tp + datapoint_fn) > 0 else 0.0
    datapoint_fpr = datapoint_fp / (datapoint_fp + datapoint_tn) if (datapoint_fp + datapoint_tn) > 0 else 0.0
    
    return {
        'event_tpr': event_tpr,
        'event_fpr': event_fpr,
        'event_tp': int(event_tp),
        'event_tn': int(event_tn),
        'event_fp': int(event_fp),
        'event_fn': int(event_fn),
        'n_events': len(event_stats),
        'n_seizure_events': int(np.sum(event_stats['true_label'] == 1)),
        'n_non_seizure_events': int(np.sum(event_stats['true_label'] == 0)),
        'datapoint_tpr': datapoint_tpr,
        'datapoint_fpr': datapoint_fpr,
    }


def compare_metrics(datapoint_tpr: float, datapoint_fpr: float, event_tpr: float, event_fpr: float) -> None:
    """
    Print a comparison between datapoint-level and event-level metrics.
    
    Args:
        datapoint_tpr: Datapoint-level True Positive Rate
        datapoint_fpr: Datapoint-level False Positive Rate
        event_tpr: Event-level True Positive Rate
        event_fpr: Event-level False Positive Rate
    """
    print("\n" + "="*80)
    print("DATAPOINT vs EVENT-LEVEL METRICS COMPARISON")
    print("="*80)
    print(f"\nDatapoint-level metrics:")
    print(f"  TPR (Sensitivity): {datapoint_tpr*100:6.2f}%")
    print(f"  FPR (False Alarm Rate): {datapoint_fpr*100:6.2f}%")
    print(f"\nEvent-level metrics:")
    print(f"  TPR (Sensitivity): {event_tpr*100:6.2f}%")
    print(f"  FPR (False Alarm Rate): {event_fpr*100:6.2f}%")
    print(f"\nGain from aggregation:")
    print(f"  TPR improvement: {(event_tpr - datapoint_tpr)*100:+6.2f}% absolute ({(event_tpr/datapoint_tpr if datapoint_tpr > 0 else 0)*100:.0f}% relative)")
    print(f"  FPR change: {(event_fpr - datapoint_fpr)*100:+6.2f}% absolute")
    print("="*80 + "\n")
