#!/usr/bin/env python3
"""
analyze_merged_events.py

Creates a detailed spreadsheet showing which events were merged/removed
by the refactored version compared to the original, with full event details
to allow manual verification of grouping decisions.

Graham Jones, July 2026
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Paths
ORIGINAL_DB = "/home/graham/osd/osdb_test_original/osdb_3min_allSeizures.json"
REFACTORED_DB = "/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures.json"
OUTPUT_CSV = "/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/comparison_results/merged_events_detail.csv"

def load_events(filepath):
    """Load events from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_event_map(events):
    """Create a map of event ID to event details."""
    return {int(e['id']): e for e in events}

def parse_datatime(dt_str):
    """Parse dataTime string which can be in multiple formats. Returns timezone-naive datetime."""
    # Try ISO format first
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        # Convert to timezone-naive
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        pass
    
    # Try DD-MM-YYYY HH:MM:SS format
    try:
        return datetime.strptime(dt_str, '%d-%m-%Y %H:%M:%S')
    except:
        pass
    
    # Try other common formats
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
        try:
            return datetime.strptime(dt_str, fmt)
        except:
            continue
    
    raise ValueError(f"Cannot parse datetime: {dt_str}")

def find_time_neighbors(event, all_events, time_window_minutes=3):
    """Find events that could have been merged with this event (within 3 minutes)."""
    from datetime import datetime, timedelta
    
    event_time = parse_datatime(event['dataTime'])
    user_id = event.get('userId')
    event_type = event.get('type')
    
    neighbors = []
    for other in all_events:
        if int(other['id']) == int(event['id']):
            continue
        if other.get('userId') != user_id or other.get('type') != event_type:
            continue
            
        other_time = parse_datatime(other['dataTime'])
        time_diff = abs((event_time - other_time).total_seconds() / 60)
        
        if time_diff <= time_window_minutes:
            neighbors.append({
                'id': other['id'],
                'dataTime': other['dataTime'],
                'time_diff_minutes': round(time_diff, 2)
            })
    
    return sorted(neighbors, key=lambda x: x['time_diff_minutes'])

def main():
    print("="*70)
    print("Merged Events Analysis")
    print("="*70)
    
    # Load databases
    print(f"\nLoading original database: {ORIGINAL_DB}")
    original_events = load_events(ORIGINAL_DB)
    print(f"  Loaded {len(original_events)} events")
    
    print(f"\nLoading refactored database: {REFACTORED_DB}")
    refactored_events = load_events(REFACTORED_DB)
    print(f"  Loaded {len(refactored_events)} events")
    
    # Create maps
    original_map = create_event_map(original_events)
    refactored_map = create_event_map(refactored_events)
    
    # Find events that exist in original but not in refactored
    original_ids = set(original_map.keys())
    refactored_ids = set(refactored_map.keys())
    
    removed_ids = original_ids - refactored_ids
    
    print(f"\nEvents in original but not in refactored: {len(removed_ids)}")
    
    # Analyze each removed event
    analysis_rows = []
    
    for event_id in sorted(removed_ids):
        event = original_map[event_id]
        
        # Find potential merge targets (events that are still in refactored)
        neighbors = find_time_neighbors(event, refactored_events)
        
        # Get event details
        row = {
            'removed_event_id': event_id,
            'user_id': event.get('userId', ''),
            'dataTime': event.get('dataTime', ''),
            'type': event.get('type', ''),
            'subType': event.get('subType', ''),
            'desc': event.get('desc', ''),
            'osdAlarmState': event.get('osdAlarmState', ''),
            'datapoint_count': len(event.get('datapoints', [])) if isinstance(event.get('datapoints'), list) else event.get('datapoints', 0),
            'likely_merged_into_id': neighbors[0]['id'] if neighbors else 'UNKNOWN',
            'likely_merged_into_time': neighbors[0]['dataTime'] if neighbors else '',
            'time_difference_minutes': neighbors[0]['time_diff_minutes'] if neighbors else '',
            'num_nearby_events': len(neighbors)
        }
        
        analysis_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(analysis_rows)
    
    # Sort by user_id, then dataTime
    df = df.sort_values(['user_id', 'dataTime'])
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Saved detailed analysis to: {OUTPUT_CSV}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"\nTotal events removed/merged: {len(removed_ids)}")
    print(f"\nBreakdown by type:")
    print(df['type'].value_counts())
    
    print(f"\nBreakdown by user:")
    print(df['user_id'].value_counts().head(10))
    
    unknown_merges = len(df[df['likely_merged_into_id'] == 'UNKNOWN'])
    if unknown_merges > 0:
        print(f"\n⚠ Warning: {unknown_merges} events have no clear merge target within 3 minutes")
    
    # Calculate average time difference for events with known merge targets
    known_times = df[df['time_difference_minutes'] != '']['time_difference_minutes']
    if len(known_times) > 0:
        print(f"\nAverage time difference to merge target: {known_times.mean():.2f} minutes")
        print(f"Maximum time difference: {known_times.max():.2f} minutes")
    
    print("\n" + "="*70)
    print("CSV file contains columns:")
    print("="*70)
    for col in df.columns:
        print(f"  - {col}")
    
    print("\n✓ Analysis complete!")
    print(f"\nYou can open the CSV in a spreadsheet application:")
    print(f"  libreoffice {OUTPUT_CSV}")
    print(f"  or")
    print(f"  xdg-open {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
