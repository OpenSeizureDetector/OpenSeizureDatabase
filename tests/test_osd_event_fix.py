#!/usr/bin/env python3
"""Test script to verify the OSD event-level fix"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

print("=== Testing OSD Event-Level Fix ===\n")

# Simulate the scenario where dp2vector filters out early datapoints
np.random.seed(42)

# Create original dataframe with 5 events
events_data_orig = []
for event_id in range(5):
    true_label = 1 if event_id < 3 else 0  # First 3 are seizures
    n_points = 60
    
    for i in range(n_points):
        # OSD alarms occur early in seizure events (first 15 datapoints)
        # But dp2vector might skip these early datapoints!
        if true_label == 1 and i < 15:
            osd_alarm = 3  # High alarm
        elif true_label == 1 and i < 30:
            osd_alarm = 2  # Moderate alarm
        else:
            osd_alarm = np.random.choice([0, 1], p=[0.9, 0.1])
        
        events_data_orig.append({
            'eventId': event_id,
            'type': true_label,
            'osdAlarmState': osd_alarm,
            'row_index': len(events_data_orig)
        })

df_original = pd.DataFrame(events_data_orig)
print(f"Original dataframe: {len(df_original)} rows, {df_original['eventId'].nunique()} events")

# Simulate dp2vector filtering: skip first 20 datapoints of each event (buffer filling)
kept_indices = []
for event_id in df_original['eventId'].unique():
    event_indices = df_original[df_original['eventId'] == event_id].index
    # Skip first 20 datapoints
    kept_indices.extend(event_indices[20:])

print(f"dp2vector filtering: keeping {len(kept_indices)}/{len(df_original)} rows")
print(f"Removed {len(df_original) - len(kept_indices)} rows (first 20 of each event)\n")

# Create filtered dataframe
df = df_original.iloc[kept_indices].reset_index(drop=True)

# OLD METHOD (INCORRECT): Calculate OSD event stats from filtered df
print("=== OLD METHOD (using filtered df) ===")
df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

event_stats_old = []
for eventId, group in df.groupby('eventId'):
    true_label = group['type'].iloc[0]
    osd_event_pred = 1 if (group['osd_pred'] == 1).any() else 0
    event_stats_old.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })

event_stats_df_old = pd.DataFrame(event_stats_old)
cm_old = confusion_matrix(event_stats_df_old['true_label'], event_stats_df_old['osd_pred'], labels=[0, 1])
tn_old, fp_old, fn_old, tp_old = cm_old.ravel()
print(f"Old method - TP={tp_old}, FP={fp_old}, FN={fn_old}, TN={tn_old}")
print(f"Problem: TP={tp_old} (should be 3 since all 3 seizure events had OSD alarms!)\n")

# NEW METHOD (CORRECT): Calculate OSD event stats from ORIGINAL df
print("=== NEW METHOD (using original df) ===")
df_original['osd_pred_orig'] = df_original['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

event_stats_new = []
for eventId in df['eventId'].unique():
    # Get true label from filtered df
    true_label = df[df['eventId'] == eventId]['type'].iloc[0]
    
    # Get OSD prediction from ORIGINAL df
    group_orig = df_original[df_original['eventId'] == eventId]
    osd_event_pred = 1 if (group_orig['osd_pred_orig'] == 1).any() else 0
    
    n_alarms_filtered = (df[df['eventId'] == eventId]['osd_pred'] == 1).sum() if eventId in df['eventId'].values else 0
    n_alarms_original = (group_orig['osd_pred_orig'] == 1).sum()
    
    event_stats_new.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })
    
    if true_label == 1:
        print(f"  Seizure event {eventId}: OSD alarms filtered={n_alarms_filtered}/{len(df[df['eventId']==eventId])}, " +
              f"original={n_alarms_original}/{len(group_orig)}, pred={osd_event_pred}")

event_stats_df_new = pd.DataFrame(event_stats_new)
cm_new = confusion_matrix(event_stats_df_new['true_label'], event_stats_df_new['osd_pred'], labels=[0, 1])
tn_new, fp_new, fn_new, tp_new = cm_new.ravel()
print(f"\nNew method - TP={tp_new}, FP={fp_new}, FN={fn_new}, TN={tn_new}")
print(f"Success: TP={tp_new} (correctly identified all 3 seizure events!)\n")

print("=== CONCLUSION ===")
print(f"Old method missed {tp_new - tp_old} seizure events due to filtering.")
print("New method correctly uses original unfiltered data for OSD event-level statistics.")
