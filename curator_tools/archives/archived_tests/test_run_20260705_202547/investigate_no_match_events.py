#!/usr/bin/env python3
"""
investigate_no_match_events.py

Investigates why 13 events have NO_MATCH in merge comparison.
These events were in original but not in refactored, with no nearby event found.

Graham Jones, July 2026
"""

import json
import csv
import sys
from pathlib import Path

# Paths
MERGE_CSV = "/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/comparison_results/merge_comparison_detailed.csv"
ORIGINAL_DB = "/home/graham/osd/osdb_test_original/osdb_3min_allSeizures.json"
REFACTORED_DB = "/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures.json"
INVALID_EVENTS_FILE = "/home/graham/osd/OpenSeizureDatabase/curator_tools/invalidEvents.txt"
CONFIG_FILE = "/home/graham/osd/OpenSeizureDatabase/curator_tools/osdb.cfg"

def load_invalid_events():
    """Load invalid event IDs."""
    invalid_ids = set()
    try:
        with open(INVALID_EVENTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        invalid_ids.add(int(line))
                    except:
                        pass
    except FileNotFoundError:
        print(f"Warning: {INVALID_EVENTS_FILE} not found")
    return invalid_ids

def load_config():
    """Load osdb.cfg configuration."""
    config = {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Warning: {CONFIG_FILE} not found")
    return config

def get_no_match_events():
    """Get event IDs that have NO_MATCH."""
    no_match_ids = []
    with open(MERGE_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['MERGED_INTO_id'] == 'NO_MATCH':
                no_match_ids.append(int(row['REMOVED_id']))
    return no_match_ids

def load_events(filepath):
    """Load events from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_event_map(events):
    """Create map of event ID to event."""
    return {int(e['id']): e for e in events}

def main():
    print("="*70)
    print("Investigating NO_MATCH Events")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    no_match_ids = get_no_match_events()
    print(f"  Found {len(no_match_ids)} NO_MATCH events")
    
    invalid_events = load_invalid_events()
    print(f"  Loaded {len(invalid_events)} invalid event IDs")
    
    config = load_config()
    exclude_sources = eval(config.get('excludeDataSources', '[]'))
    print(f"  Excluded data sources: {exclude_sources}")
    
    original_events = load_events(ORIGINAL_DB)
    original_map = create_event_map(original_events)
    print(f"  Loaded {len(original_events)} original events")
    
    refactored_events = load_events(REFACTORED_DB)
    refactored_map = create_event_map(refactored_events)
    print(f"  Loaded {len(refactored_events)} refactored events")
    
    print("\n" + "="*70)
    print("Analysis of Each NO_MATCH Event")
    print("="*70)
    
    reasons = {
        'invalid_event_list': [],
        'data_source_filter': [],
        'no_datapoints': [],
        'unknown': []
    }
    
    for event_id in sorted(no_match_ids):
        event = original_map.get(event_id)
        if not event:
            print(f"\n⚠ Event {event_id}: NOT FOUND in original database (shouldn't happen)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Event ID: {event_id}")
        print(f"  User: {event.get('userId')}")
        print(f"  Time: {event.get('dataTime')}")
        print(f"  Type: {event.get('type')} / {event.get('subType')}")
        print(f"  Desc: {event.get('desc', '')[:60]}")
        print(f"  Alarm State: {event.get('osdAlarmState')}")
        
        # Get datapoints info
        datapoints = event.get('datapoints', [])
        if isinstance(datapoints, list):
            dp_count = len(datapoints)
        else:
            dp_count = datapoints
        print(f"  Datapoints: {dp_count}")
        
        # Get data source
        data_source = event.get('dataSourceName', 'UNKNOWN')
        print(f"  Data Source: {data_source}")
        
        # Check why removed
        reason_found = False
        
        # 1. Check if in invalid events list
        if event_id in invalid_events:
            print(f"  ❌ REASON: In invalidEvents.txt list")
            reasons['invalid_event_list'].append(event_id)
            reason_found = True
        
        # 2. Check if data source filtered
        elif data_source in exclude_sources:
            print(f"  ❌ REASON: Data source '{data_source}' is in excludeDataSources")
            reasons['data_source_filter'].append(event_id)
            reason_found = True
        
        # 3. Check if no datapoints
        elif dp_count == 0:
            print(f"  ❌ REASON: Event has no datapoints (validation skip)")
            reasons['no_datapoints'].append(event_id)
            reason_found = True
        
        if not reason_found:
            print(f"  ⚠ REASON: UNKNOWN - should have been retained!")
            reasons['unknown'].append(event_id)
            
            # Additional investigation
            print(f"  🔍 Additional checks:")
            print(f"     - In refactored DB: {event_id in refactored_map}")
            print(f"     - In invalid list: {event_id in invalid_events}")
            print(f"     - Data source excluded: {data_source in exclude_sources}")
            print(f"     - Has datapoints: {dp_count > 0}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTotal NO_MATCH events: {len(no_match_ids)}")
    print(f"\nBreakdown by reason:")
    print(f"  - In invalidEvents.txt: {len(reasons['invalid_event_list'])} events")
    if reasons['invalid_event_list']:
        print(f"    IDs: {reasons['invalid_event_list']}")
    
    print(f"  - Data source filtered: {len(reasons['data_source_filter'])} events")
    if reasons['data_source_filter']:
        print(f"    IDs: {reasons['data_source_filter']}")
    
    print(f"  - No datapoints (validation): {len(reasons['no_datapoints'])} events")
    if reasons['no_datapoints']:
        print(f"    IDs: {reasons['no_datapoints']}")
    
    print(f"  - UNKNOWN (possible bug): {len(reasons['unknown'])} events")
    if reasons['unknown']:
        print(f"    IDs: {reasons['unknown']}")
        print(f"    ⚠ These events should have been retained - investigate!")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if len(reasons['unknown']) == 0:
        print("\n✓ All NO_MATCH events have valid reasons for removal:")
        print("  - They were either in invalidEvents.txt,")
        print("  - or filtered by data source,")
        print("  - or had no datapoints.")
        print("\n✓ These are NOT errors - they were correctly filtered out.")
        print("\n  The 'NO_MATCH' status means:")
        print("  - The event was legitimately removed (not merged)")
        print("  - There's no nearby event to compare it to")
        print("  - This is expected behavior")
    else:
        print(f"\n⚠ {len(reasons['unknown'])} events have UNKNOWN removal reason!")
        print("  These may have been deleted in error.")
        print("  Further investigation needed.")
    
    return len(reasons['unknown'])

if __name__ == "__main__":
    unknown_count = main()
    sys.exit(unknown_count)  # Exit with count of unknown events
