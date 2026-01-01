#!/usr/bin/env python3
"""Debug script to check OSD event aggregation logic - mimics exact nnTester pattern"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Create sample data that mimics filtered dataframe after dp2vector
np.random.seed(42)

# Simulate 10 events with varying numbers of datapoints
events_data = []
for event_id in range(10):
    true_label = 1 if event_id < 5 else 0  # First 5 are seizures
    n_points = np.random.randint(30, 60)
    
    for i in range(n_points):
        # For seizure events, OSD should detect SOME datapoints
        if true_label == 1:
            osd_alarm = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.3, 0.1])
        else:
            osd_alarm = np.random.choice([0, 1, 2, 3], p=[0.8, 0.15, 0.04, 0.01])
        
        events_data.append({
            'eventId': event_id,
            'type': true_label,
            'osdAlarmState': osd_alarm
        })

df = pd.DataFrame(events_data)
print(f"Total datapoints: {len(df)}")
print(f"Events: {df['eventId'].nunique()}")

# Simulate the exact code from nnTester.py line 445
df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

# Check datapoint-level OSD predictions
print(f"\n=== DATAPOINT-LEVEL OSD ===")
print(f"Positive predictions: {(df['osd_pred'] == 1).sum()}/{len(df)}")
print(f"In seizure events: {df[df['type']==1]['osd_pred'].sum()}/{len(df[df['type']==1])}")
print(f"In non-seizure events: {df[df['type']==0]['osd_pred'].sum()}/{len(df[df['type']==0])}")

# Datapoint-level confusion matrix (like line 449)
yTestOsd = df['type'].values
yPredOsd = df['osd_pred'].values
cmOsd = confusion_matrix(yTestOsd, yPredOsd, labels=[0, 1])
tnOsd, fpOsd, fnOsd, tpOsd = cmOsd.ravel()
print(f"\nDatapoint-level CM: TP={tpOsd}, FP={fpOsd}, FN={fnOsd}, TN={tnOsd}")

# Event-level statistics (like lines 453-468)
print(f"\n=== EVENT-LEVEL OSD (using exact nnTester logic) ===")
event_stats = []
for eventId, group in df.groupby('eventId'):
    true_label = group['type'].iloc[0]
    osd_event_pred = 1 if (group['osd_pred'] == 1).any() else 0
    
    # Debug: print details for each event
    n_osd_detections = (group['osd_pred'] == 1).sum()
    print(f"Event {eventId}: true={true_label}, osd_pred={osd_event_pred}, " +
          f"datapoints_with_alarm={n_osd_detections}/{len(group)}")
    
    event_stats.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })

event_stats_df = pd.DataFrame(event_stats)

# Event-level confusion matrix (like line 517)
print(f"\n=== EVENT-LEVEL CONFUSION MATRIX ===")
osd_event_cm = confusion_matrix(event_stats_df['true_label'], event_stats_df['osd_pred'], labels=[0, 1])
osd_event_tn, osd_event_fp, osd_event_fn, osd_event_tp = osd_event_cm.ravel()
print(f"Event-level CM: TP={osd_event_tp}, FP={osd_event_fp}, FN={osd_event_fn}, TN={osd_event_tn}")

osd_event_tpr = osd_event_tp / (osd_event_tp + osd_event_fn) if (osd_event_tp + osd_event_fn) > 0 else 0
osd_event_fpr = osd_event_fp / (osd_event_fp + osd_event_tn) if (osd_event_fp + osd_event_tn) > 0 else 0
print(f"TPR={osd_event_tpr:.3f}, FPR={osd_event_fpr:.3f}")

# Verify the event_stats_df structure
print(f"\n=== EVENT_STATS_DF ===")
print(event_stats_df)
print(f"\nColumn dtypes: {event_stats_df.dtypes.to_dict()}")
print(f"osd_pred values: {event_stats_df['osd_pred'].values}")
print(f"osd_pred sum: {event_stats_df['osd_pred'].sum()}")
