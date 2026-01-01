#!/usr/bin/env python3
"""Test script with extreme filtering that removes ALL OSD alarms"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

print("=== Testing OSD Event-Level Fix (Extreme Case) ===\n")

# Simulate scenario where OSD alarms only occur in first few datapoints
# which get filtered out by dp2vector
np.random.seed(42)

events_data_orig = []
for event_id in range(5):
    true_label = 1 if event_id < 3 else 0  # First 3 are seizures
    n_points = 60
    
    for i in range(n_points):
        # OSD alarms ONLY in first 10 datapoints of seizure events
        # All other datapoints have no alarms
        if true_label == 1 and i < 10:
            osd_alarm = 3  # High alarm - BUT THESE GET FILTERED OUT!
        else:
            osd_alarm = 0  # No alarm
        
        events_data_orig.append({
            'eventId': event_id,
            'type': true_label,
            'osdAlarmState': osd_alarm,
        })

df_original = pd.DataFrame(events_data_orig)
print(f"Original dataframe: {len(df_original)} rows")
print(f"OSD alarms (>=2) in original: {(df_original['osdAlarmState'] >= 2).sum()}")

# Check OSD alarms by event in original data
print("\nOriginal data - OSD alarms by event:")
for eid, grp in df_original.groupby('eventId'):
    true_label = grp['type'].iloc[0]
    n_alarms = (grp['osdAlarmState'] >= 2).sum()
    if true_label == 1:
        print(f"  Event {eid} (seizure): {n_alarms}/{len(grp)} datapoints with alarms")

# Simulate dp2vector filtering: skip first 15 datapoints of each event
# This removes ALL the OSD alarms!
kept_indices = []
for event_id in df_original['eventId'].unique():
    event_indices = df_original[df_original['eventId'] == event_id].index
    kept_indices.extend(event_indices[15:])  # Skip first 15

print(f"\ndp2vector filtering: keeping {len(kept_indices)}/{len(df_original)} rows")
df = df_original.iloc[kept_indices].reset_index(drop=True)
print(f"OSD alarms (>=2) after filtering: {(df['osdAlarmState'] >= 2).sum()}")
print("*** ALL OSD ALARMS WERE FILTERED OUT! ***\n")

# OLD METHOD (INCORRECT): Calculate from filtered df
print("=== OLD METHOD (using filtered df) ===")
df['osd_pred'] = df['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

event_stats_old = []
for eventId, group in df.groupby('eventId'):
    true_label = group['type'].iloc[0]
    osd_event_pred = 1 if (group['osd_pred'] == 1).any() else 0
    if true_label == 1:
        print(f"  Event {eventId} (seizure): osd_pred={osd_event_pred}")
    event_stats_old.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })

event_stats_df_old = pd.DataFrame(event_stats_old)
cm_old = confusion_matrix(event_stats_df_old['true_label'], event_stats_df_old['osd_pred'], labels=[0, 1])
tn_old, fp_old, fn_old, tp_old = cm_old.ravel()
print(f"\n*** WRONG: TP={tp_old} (should be 3, not {tp_old}!) ***\n")

# NEW METHOD (CORRECT): Calculate from ORIGINAL df
print("=== NEW METHOD (using original unfiltered df) ===")
df_original['osd_pred_orig'] = df_original['osdAlarmState'].apply(lambda x: 1 if x >= 2 else 0)

event_stats_new = []
for eventId in df['eventId'].unique():
    true_label = df[df['eventId'] == eventId]['type'].iloc[0]
    group_orig = df_original[df_original['eventId'] == eventId]
    osd_event_pred = 1 if (group_orig['osd_pred_orig'] == 1).any() else 0
    
    if true_label == 1:
        n_alarms_orig = (group_orig['osd_pred_orig'] == 1).sum()
        print(f"  Event {eventId} (seizure): osd_pred={osd_event_pred} (found {n_alarms_orig} alarms in original data)")
    
    event_stats_new.append({
        'eventId': eventId,
        'true_label': true_label,
        'osd_pred': osd_event_pred
    })

event_stats_df_new = pd.DataFrame(event_stats_new)
cm_new = confusion_matrix(event_stats_df_new['true_label'], event_stats_df_new['osd_pred'], labels=[0, 1])
tn_new, fp_new, fn_new, tp_new = cm_new.ravel()
print(f"\n*** CORRECT: TP={tp_new} (all 3 seizure events detected!) ***\n")

print("="*60)
print(f"OLD METHOD: TP={tp_old}, FP={fp_old}, FN={fn_old}, TN={tn_old}")
print(f"NEW METHOD: TP={tp_new}, FP={fp_new}, FN={fn_new}, TN={tn_new}")
print(f"\nThe fix recovered {tp_new - tp_old} missed seizure detections!")
print("="*60)
