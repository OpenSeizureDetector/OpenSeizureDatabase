#!/usr/bin/env python3
"""Debug script to check OSD event statistics calculation"""

import pandas as pd
import numpy as np

# Create sample data mimicking the actual dataframe structure
np.random.seed(42)

# Simulate 20 events with 50 datapoints each
n_events = 20
n_datapoints_per_event = 50

data = []
for event_id in range(n_events):
    # Half seizures (type=1), half non-seizures (type=0)
    true_label = 1 if event_id < 10 else 0
    
    for dp in range(n_datapoints_per_event):
        # OSD alarm state: simulate some alarms
        # For seizures, let's make OSD detect some of them
        if true_label == 1:
            # OSD detects 60% of seizure events (some datapoints in the event have alarm >= 2)
            osd_alarm = np.random.choice([0, 1, 2, 3], p=[0.3, 0.1, 0.4, 0.2])
        else:
            # OSD has false alarms 20% of the time
            osd_alarm = np.random.choice([0, 1, 2, 3], p=[0.7, 0.1, 0.15, 0.05])
        
        data.append({
            'eventId': event_id,
            'type': true_label,
            'osdAlarmState': osd_alarm
        })

df = pd.DataFrame(data)
print(f"Created dataframe with {len(df)} rows, {df['eventId'].nunique()} events")
print(f"Seizure events: {df[df['type']==1]['eventId'].nunique()}")
print(f"Non-seizure events: {df[df['type']==0]['eventId'].nunique()}")

# Simulate the OSD prediction calculation
df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

print(f"\nDatapoint-level OSD predictions:")
print(f"  Total positive predictions: {(df['osd_pred'] == 1).sum()}")
print(f"  Total negative predictions: {(df['osd_pred'] == 0).sum()}")

# Calculate event-level OSD statistics
event_stats = []
for eventId, group in df.groupby('eventId'):
    true_label = group['type'].iloc[0]
    osd_event_pred = 1 if (group['osd_pred'] == 1).any() else 0
    
    event_stats.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })

event_stats_df = pd.DataFrame(event_stats)

print(f"\nEvent-level OSD predictions:")
print(f"  Events predicted positive: {(event_stats_df['osd_pred'] == 1).sum()}")
print(f"  Events predicted negative: {(event_stats_df['osd_pred'] == 0).sum()}")

# Calculate confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(event_stats_df['true_label'], event_stats_df['osd_pred'], labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print(f"\nOSD Event-level Confusion Matrix:")
print(f"  TN: {tn}, FP: {fp}")
print(f"  FN: {fn}, TP: {tp}")

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print(f"  TPR: {tpr:.3f}")
print(f"  FPR: {fpr:.3f}")

# Check which seizure events were detected
print(f"\nDetailed event breakdown:")
for _, row in event_stats_df.iterrows():
    if row['true_label'] == 1:  # Only show seizure events
        status = "DETECTED" if row['osd_pred'] == 1 else "MISSED"
        print(f"  Event {row['eventId']}: {status}")
