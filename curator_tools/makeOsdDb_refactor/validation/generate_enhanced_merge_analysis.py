#!/usr/bin/env python3
"""
Enhanced merge analysis showing datapoint increases and time ranges
"""

import json
import csv
from pathlib import Path
from dateutil import parser as dateutil_parser

def parse_datatime(dt_str):
    """Parse datetime string with dayfirst=True, return timezone-naive."""
    if not dt_str or dt_str == 'N/A':
        return None
    try:
        dt = dateutil_parser.parse(str(dt_str), dayfirst=True)
        return dt.replace(tzinfo=None)
    except:
        return None

def get_datapoint_time_range(event):
    """Get first and last datapoint times."""
    datapoints = event.get('datapoints', [])
    if not datapoints:
        return None, None, 0
    
    times = []
    for dp in datapoints:
        if 'dataTime' in dp:
            try:
                times.append(dateutil_parser.parse(dp['dataTime']))
            except:
                pass
    
    if not times:
        return None, None, 0
    
    times.sort()
    duration = (times[-1] - times[0]).total_seconds()
    return times[0], times[-1], duration

# Load data
original_path = Path('/home/graham/osd/osdb_test_original/osdb_3min_allSeizures.json')
refactored_path = Path('/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures.json')

with open(original_path) as f:
    original_events = {e['id']: e for e in json.load(f)}

with open(refactored_path) as f:
    refactored_events = {e['id']: e for e in json.load(f)}

# Build mapping of merged events
merged_into_map = {}
for ref_event in refactored_events.values():
    merged_from = ref_event.get('_merged_from_event_ids', [])
    # Normalize to list (handle legacy formats)
    if merged_from is None:
        merged_from = []
    elif not isinstance(merged_from, list):
        merged_from = [merged_from]
    if merged_from and len(merged_from) > 1:
        target_id = ref_event['id']
        for source_id in merged_from:
            if source_id != target_id:
                merged_into_map[source_id] = target_id

# Generate enhanced report (now in validation/comparison_results/)
output_path = Path(__file__).parent / 'comparison_results' / 'merge_analysis_enhanced.csv'

with open(output_path, 'w', newline='') as csvfile:
    fieldnames = [
        'REMOVED_ID', 'REMOVED_DATAPOINTS', 'REMOVED_DURATION_sec',
        'MERGED_INTO_ID', 'TARGET_DATAPOINTS_BEFORE', 'TARGET_DURATION_BEFORE_sec',
        'MERGED_DATAPOINTS_AFTER', 'MERGED_DURATION_AFTER_sec',
        'DATAPOINTS_INCREASE', 'DURATION_INCREASE_sec',
        'EXPECTED_SUM', 'DUPLICATES_REMOVED', 'DUPLICATE_PCT',
        'TIME_DIFF_min', 'WITHIN_3MIN'
    ]
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    total_merges = 0
    total_increase = 0
    total_expected = 0
    total_duplicates = 0
    
    for removed_id in sorted(merged_into_map.keys()):
        merged_into_id = merged_into_map[removed_id]
        
        # Get original event data
        removed_event = original_events.get(removed_id)
        target_event_original = original_events.get(merged_into_id)
        merged_event = refactored_events.get(merged_into_id)
        
        if not removed_event or not target_event_original or not merged_event:
            continue
        
        removed_dps = len(removed_event.get('datapoints', []))
        target_dps_before = len(target_event_original.get('datapoints', []))
        merged_dps_after = len(merged_event.get('datapoints', []))
        
        # Get time ranges
        _, _, removed_duration = get_datapoint_time_range(removed_event)
        _, _, target_duration_before = get_datapoint_time_range(target_event_original)
        merged_first, merged_last, merged_duration_after = get_datapoint_time_range(merged_event)
        
        # Calculate statistics
        datapoints_increase = merged_dps_after - target_dps_before
        duration_increase = merged_duration_after - target_duration_before
        expected_sum = removed_dps + target_dps_before
        duplicates_removed = expected_sum - merged_dps_after
        duplicate_pct = (duplicates_removed / expected_sum * 100) if expected_sum > 0 else 0
        
        # Time difference between events
        removed_time = parse_datatime(removed_event.get('dataTime'))
        target_time = parse_datatime(target_event_original.get('dataTime'))
        
        if removed_time and target_time:
            time_diff = abs((removed_time - target_time).total_seconds() / 60.0)
            within_threshold = "YES" if time_diff <= 3.0 else "NO"
        else:
            time_diff = None
            within_threshold = "N/A"
        
        row = {
            'REMOVED_ID': removed_id,
            'REMOVED_DATAPOINTS': removed_dps,
            'REMOVED_DURATION_sec': f"{removed_duration:.0f}" if removed_duration else "N/A",
            'MERGED_INTO_ID': merged_into_id,
            'TARGET_DATAPOINTS_BEFORE': target_dps_before,
            'TARGET_DURATION_BEFORE_sec': f"{target_duration_before:.0f}" if target_duration_before else "N/A",
            'MERGED_DATAPOINTS_AFTER': merged_dps_after,
            'MERGED_DURATION_AFTER_sec': f"{merged_duration_after:.0f}" if merged_duration_after else "N/A",
            'DATAPOINTS_INCREASE': datapoints_increase,
            'DURATION_INCREASE_sec': f"{duration_increase:.0f}" if duration_increase else "N/A",
            'EXPECTED_SUM': expected_sum,
            'DUPLICATES_REMOVED': duplicates_removed,
            'DUPLICATE_PCT': f"{duplicate_pct:.1f}",
            'TIME_DIFF_min': f"{time_diff:.2f}" if time_diff is not None else "N/A",
            'WITHIN_3MIN': within_threshold
        }
        
        writer.writerow(row)
        
        total_merges += 1
        total_increase += datapoints_increase
        total_expected += expected_sum
        total_duplicates += duplicates_removed

print(f"✅ Enhanced merge analysis complete!")
print(f"")
print(f"Summary:")
print(f"  Total merges: {total_merges}")
print(f"  Average datapoint increase: {total_increase / total_merges:.1f} per merge")
print(f"  Total duplicates removed: {total_duplicates}")
print(f"  Average duplicate rate: {(total_duplicates / total_expected * 100):.1f}%")
print(f"")
print(f"Generated: {output_path}")
