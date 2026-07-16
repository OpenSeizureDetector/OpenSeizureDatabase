#!/usr/bin/env python3
"""
Debug script to trace merge of events 115 and 119
"""

import json
import sys
sys.path.insert(0, '/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/src')

from event_grouping import merge_grouped_events, concatenate_datapoints, select_best_event_from_group

# Load the downloaded events
with open('/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures_08jun26.json', 'r') as f:
    events = json.load(f)

# Find events 115 and 119
event_115 = None
event_119 = None

for event in events:
    if event.get('id') == 115:
        event_115 = event.copy()
    elif event.get('id') == 119:
        event_119 = event.copy()

if not event_115 or not event_119:
    print("ERROR: Could not find events 115 and 119!")
    sys.exit(1)

print("=== Original Events ===")
print(f"Event 115: {len(event_115.get('datapoints', []))} datapoints")
print(f"Event 119: {len(event_119.get('datapoints', []))} datapoints")
print(f"Total expected: {len(event_115.get('datapoints', []))} + {len(event_119.get('datapoints', []))} = {len(event_115.get('datapoints', [])) + len(event_119.get('datapoints', []))}")

# Create a group
group = [event_119, event_115]  # Time order: 119 is earlier

# Test concatenate_datapoints directly
print("\n=== Testing concatenate_datapoints ===")
concatenated = concatenate_datapoints(group, remove_duplicates=True)
print(f"Concatenated datapoints (with dedup): {len(concatenated)}")

concatenated_no_dedup = concatenate_datapoints(group, remove_duplicates=False)
print(f"Concatenated datapoints (no dedup): {len(concatenated_no_dedup)}")

# Test select_best_event
print("\n=== Testing select_best_event_from_group ===")
selected = select_best_event_from_group(group)
print(f"Selected event: {selected.get('id')}")
print(f"Selected has {len(selected.get('datapoints', []))} datapoints")

# Test merge_grouped_events
print("\n=== Testing merge_grouped_events ===")
merged = merge_grouped_events(group, selected, concatenate_datapoints_flag=True)
print(f"Merged event id: {merged.get('id')}")
print(f"Merged datapoints: {len(merged.get('datapoints', []))}")
print(f"_merged_from_event_ids: {merged.get('_merged_from_event_ids', [])}")
print(f"_merged_event_count: {merged.get('_merged_event_count', 0)}")
print(f"_merged_datapoint_count: {merged.get('_merged_datapoint_count', 0)}")

# Compare
print("\n=== Summary ===")
print(f"Input: {len(event_115.get('datapoints', []))} + {len(event_119.get('datapoints', []))} = {len(event_115.get('datapoints', [])) + len(event_119.get('datapoints', []))}")
print(f"Concatenated (no dedup): {len(concatenated_no_dedup)}")
print(f"Concatenated (with dedup): {len(concatenated)}")
print(f"Final merged: {len(merged.get('datapoints', []))}")
print(f"Missing: {len(concatenated) - len(merged.get('datapoints', []))}")
