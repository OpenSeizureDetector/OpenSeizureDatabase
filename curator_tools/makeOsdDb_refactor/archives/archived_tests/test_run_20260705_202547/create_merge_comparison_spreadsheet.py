#!/usr/bin/env python3
"""
create_merge_comparison_spreadsheet.py

Creates an Excel/LibreOffice-friendly spreadsheet showing removed events
and the events they were likely merged into, side-by-side for easy comparison.

Graham Jones, July 2026
"""

import json
import pandas as pd
from datetime import datetime

# Paths
ORIGINAL_DB = "/home/graham/osd/osdb_test_original/osdb_3min_allSeizures.json"
REFACTORED_DB = "/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures.json"
OUTPUT_CSV = "/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/comparison_results/merge_comparison_detailed.csv"

def parse_datatime(dt_str):
    """Parse dataTime string which can be in multiple formats."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        pass
    
    try:
        return datetime.strptime(dt_str, '%d-%m-%Y %H:%M:%S')
    except:
        pass
    
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
        try:
            return datetime.strptime(dt_str, fmt)
        except:
            continue
    
    return None

def load_events(filepath):
    """Load events from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_event_map(events):
    """Create a map of event ID to event details."""
    return {int(e['id']): e for e in events}

def find_time_neighbors(event, all_events_map, time_window_minutes=3):
    """Find events that could have been merged with this event."""
    event_time = parse_datatime(event['dataTime'])
    if not event_time:
        return []
    
    user_id = event.get('userId')
    event_type = event.get('type')
    
    neighbors = []
    for other_id, other in all_events_map.items():
        if other_id == int(event['id']):
            continue
        if other.get('userId') != user_id or other.get('type') != event_type:
            continue
            
        other_time = parse_datatime(other['dataTime'])
        if not other_time:
            continue
            
        time_diff = abs((event_time - other_time).total_seconds() / 60)
        
        if time_diff <= time_window_minutes:
            neighbors.append({
                'id': other['id'],
                'event': other,
                'time_diff_minutes': round(time_diff, 2)
            })
    
    return sorted(neighbors, key=lambda x: x['time_diff_minutes'])

def main():
    print("="*70)
    print("Creating Detailed Merge Comparison Spreadsheet")
    print("="*70)
    
    # Load databases
    print(f"\nLoading databases...")
    original_events = load_events(ORIGINAL_DB)
    refactored_events = load_events(REFACTORED_DB)
    
    original_map = create_event_map(original_events)
    refactored_map = create_event_map(refactored_events)
    
    # Find removed events
    original_ids = set(original_map.keys())
    refactored_ids = set(refactored_map.keys())
    removed_ids = original_ids - refactored_ids
    
    print(f"  Analyzing {len(removed_ids)} removed events...")
    
    # Create detailed comparison rows
    rows = []
    
    for event_id in sorted(removed_ids):
        removed_event = original_map[event_id]
        
        # Find the most likely merge target
        neighbors = find_time_neighbors(removed_event, refactored_map)
        
        if neighbors:
            merged_into = neighbors[0]['event']
            time_diff = neighbors[0]['time_diff_minutes']
            
            row = {
                # Removed event details
                'REMOVED_id': event_id,
                'REMOVED_userId': removed_event.get('userId'),
                'REMOVED_dataTime': removed_event.get('dataTime'),
                'REMOVED_type': removed_event.get('type'),
                'REMOVED_subType': removed_event.get('subType'),
                'REMOVED_desc': removed_event.get('desc', ''),
                'REMOVED_alarmState': removed_event.get('osdAlarmState'),
                'REMOVED_datapoints': len(removed_event.get('datapoints', [])) if isinstance(removed_event.get('datapoints'), list) else removed_event.get('datapoints', 0),
                
                # Time difference
                'TIME_DIFF_minutes': time_diff,
                
                # Merged into event details
                'MERGED_INTO_id': merged_into['id'],
                'MERGED_INTO_userId': merged_into.get('userId'),
                'MERGED_INTO_dataTime': merged_into.get('dataTime'),
                'MERGED_INTO_type': merged_into.get('type'),
                'MERGED_INTO_subType': merged_into.get('subType'),
                'MERGED_INTO_desc': merged_into.get('desc', ''),
                'MERGED_INTO_alarmState': merged_into.get('osdAlarmState'),
                'MERGED_INTO_datapoints': len(merged_into.get('datapoints', [])) if isinstance(merged_into.get('datapoints'), list) else merged_into.get('datapoints', 0),
                
                # Review status
                'REVIEW_STATUS': 'PENDING',
                'REVIEW_NOTES': ''
            }
        else:
            row = {
                # Removed event details
                'REMOVED_id': event_id,
                'REMOVED_userId': removed_event.get('userId'),
                'REMOVED_dataTime': removed_event.get('dataTime'),
                'REMOVED_type': removed_event.get('type'),
                'REMOVED_subType': removed_event.get('subType'),
                'REMOVED_desc': removed_event.get('desc', ''),
                'REMOVED_alarmState': removed_event.get('osdAlarmState'),
                'REMOVED_datapoints': len(removed_event.get('datapoints', [])) if isinstance(removed_event.get('datapoints'), list) else removed_event.get('datapoints', 0),
                
                # No merge target found
                'TIME_DIFF_minutes': '',
                'MERGED_INTO_id': 'NO_MATCH',
                'MERGED_INTO_userId': '',
                'MERGED_INTO_dataTime': '',
                'MERGED_INTO_type': '',
                'MERGED_INTO_subType': '',
                'MERGED_INTO_desc': '⚠ No nearby event found - may have been filtered out',
                'MERGED_INTO_alarmState': '',
                'MERGED_INTO_datapoints': '',
                
                'REVIEW_STATUS': 'NEEDS_INVESTIGATION',
                'REVIEW_NOTES': ''
            }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by user and time
    df = df.sort_values(['REMOVED_userId', 'REMOVED_dataTime'])
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Saved detailed comparison to: {OUTPUT_CSV}")
    print(f"\nSpreadsheet contains {len(df)} rows")
    print(f"\nColumn groups:")
    print(f"  - REMOVED_*: Details of event removed by refactored version")
    print(f"  - TIME_DIFF_minutes: Time between removed and merged event")
    print(f"  - MERGED_INTO_*: Details of event it was likely merged into")
    print(f"  - REVIEW_STATUS: For your manual review (PENDING/APPROVED/REJECTED)")
    print(f"  - REVIEW_NOTES: Add your notes here")
    
    no_match = len(df[df['MERGED_INTO_id'] == 'NO_MATCH'])
    if no_match > 0:
        print(f"\n⚠ {no_match} events have no clear merge target - need investigation")
    
    print("\n" + "="*70)
    print("To open in LibreOffice Calc:")
    print("="*70)
    print(f"  libreoffice --calc {OUTPUT_CSV}")
    print("\n  or simply:")
    print(f"  xdg-open {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
