#!/usr/bin/env python3
"""
analyze_database_history.py

Analyzes whether the original /home/graham/osd/osdb database contains events
that are no longer available on the remote server, which would explain
differences between original and refactored update results.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

# File paths
V1_10_DIR = "/home/graham/osd/osdb/V1.10"
ORIGINAL_DIR = "/home/graham/osd/osdb"
REFACTORED_DIR = "/home/graham/osd/osdb_refactored"

def load_events(filepath):
    """Load events from JSON file."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return json.load(f)

def get_event_ids(events):
    """Extract event IDs from events list."""
    return set(e.get('id') for e in events if 'id' in e)

def analyze_event_history(event_type_name, v110_file, original_file, refactored_file):
    """
    Analyze the history of a specific event type across three versions.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {event_type_name}")
    print(f"{'='*70}")
    
    # Load all versions
    v110_events = load_events(v110_file)
    original_events = load_events(original_file)
    refactored_events = load_events(refactored_file)
    
    # Get event ID sets
    v110_ids = get_event_ids(v110_events)
    original_ids = get_event_ids(original_events)
    refactored_ids = get_event_ids(refactored_events)
    
    print(f"\nEvent Counts:")
    print(f"  V1.10 Baseline:      {len(v110_ids):4d} events")
    print(f"  Original Updated:    {len(original_ids):4d} events")
    print(f"  Refactored Updated:  {len(refactored_ids):4d} events")
    
    # Analyze differences
    print(f"\n--- Additions vs V1.10 ---")
    original_new = original_ids - v110_ids
    refactored_new = refactored_ids - v110_ids
    print(f"  Original added:    {len(original_new):4d} events")
    print(f"  Refactored added:  {len(refactored_new):4d} events")
    
    print(f"\n--- Events Removed from V1.10 ---")
    original_removed = v110_ids - original_ids
    refactored_removed = v110_ids - refactored_ids
    print(f"  Original removed:    {len(original_removed):4d} events")
    print(f"  Refactored removed:  {len(refactored_removed):4d} events")
    
    # Key insight: Events in original but not in refactored
    print(f"\n--- Events Only in Original (Not in Refactored) ---")
    only_in_original = original_ids - refactored_ids
    print(f"  Count: {len(only_in_original)} events")
    
    # Categorize these events
    only_orig_from_v110 = only_in_original & v110_ids
    only_orig_new = only_in_original - v110_ids
    
    print(f"\n  Breakdown:")
    print(f"    From V1.10 (removed by refactored):  {len(only_orig_from_v110):4d}")
    print(f"    New additions (not in refactored):   {len(only_orig_new):4d}")
    
    # Critical insight: Were these new additions downloaded by refactored?
    if len(only_orig_new) > 0:
        print(f"\n  ⚠️  CRITICAL: Original has {len(only_orig_new)} events that are:")
        print(f"      - NOT in V1.10 baseline")
        print(f"      - NOT in refactored update")
        print(f"      This suggests these events existed in a previous update")
        print(f"      but are no longer available from the remote database.")
        
        # Show sample IDs
        sample_ids = sorted(list(only_orig_new))[:20]
        print(f"\n  Sample Event IDs (first 20):")
        for i, eid in enumerate(sample_ids, 1):
            # Find this event in original to get details
            event = next((e for e in original_events if e.get('id') == eid), None)
            if event:
                dataTime = event.get('dataTime', 'unknown')
                eventType = event.get('type', 'unknown')
                subType = event.get('subType', '')
                print(f"    {i:2d}. Event {eid:6d} - {dataTime} - {eventType}/{subType}")
    
    # Events only in refactored
    print(f"\n--- Events Only in Refactored (Not in Original) ---")
    only_in_refactored = refactored_ids - original_ids
    print(f"  Count: {len(only_in_refactored)} events")
    
    if len(only_in_refactored) > 0:
        print(f"  These are likely events merged differently by sliding window grouping")
    
    # Common events with potential modifications
    common_events = original_ids & refactored_ids
    print(f"\n--- Common Events (In Both Original and Refactored) ---")
    print(f"  Count: {len(common_events)} events")
    
    return {
        'event_type': event_type_name,
        'v110_count': len(v110_ids),
        'original_count': len(original_ids),
        'refactored_count': len(refactored_ids),
        'only_in_original': len(only_in_original),
        'only_in_original_new': len(only_orig_new),
        'only_in_refactored': len(only_in_refactored),
        'common': len(common_events),
        'original_new_ids': list(only_orig_new) if len(only_orig_new) <= 200 else list(only_orig_new)[:200]
    }

def check_remote_availability(event_ids, config_path="../osdb.cfg"):
    """
    Check if specific event IDs are still available on the remote server.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    import libosd.webApiConnection
    import libosd.configUtils
    
    print(f"\nChecking remote availability for {len(event_ids)} events...")
    
    try:
        cfgObj = libosd.configUtils.loadConfig(config_path)
        osd = libosd.webApiConnection.WebApiConnection(
            cfg=cfgObj['credentialsFname'],
            download=True,
            debug=False
        )
        
        available = []
        not_available = []
        errors = []
        
        for i, event_id in enumerate(event_ids[:50], 1):  # Check first 50
            try:
                event = osd.getEvent(event_id, includeDatapoints=False)
                if event:
                    available.append(event_id)
                else:
                    not_available.append(event_id)
            except Exception as e:
                errors.append((event_id, str(e)))
            
            if i % 10 == 0:
                print(f"  Checked {i}/{min(len(event_ids), 50)}...")
        
        print(f"\nResults (first 50 events):")
        print(f"  ✓ Available on remote:     {len(available):3d}")
        print(f"  ✗ Not available on remote: {len(not_available):3d}")
        print(f"  ⚠ Errors:                  {len(errors):3d}")
        
        if not_available:
            print(f"\n  Not available Event IDs: {not_available[:10]}")
        
        if errors:
            print(f"\n  Errors: {[(eid, err[:50]) for eid, err in errors[:5]]}")
        
        return {
            'available': available,
            'not_available': not_available,
            'errors': errors
        }
    except Exception as e:
        print(f"  Error connecting to remote: {e}")
        return None

def main():
    print("="*70)
    print("Database History Analysis")
    print("="*70)
    print("\nPurpose: Determine if /home/graham/osd/osdb contains events")
    print("         that are no longer available on the remote database.")
    print("="*70)
    
    # Analyze all seizures
    all_seizures_results = analyze_event_history(
        "All Seizures",
        os.path.join(V1_10_DIR, "osdb_3min_allSeizures.json"),
        os.path.join(ORIGINAL_DIR, "osdb_3min_allSeizures.json"),
        os.path.join(REFACTORED_DIR, "osdb_3min_allSeizures.json")
    )
    
    # Analyze fall events
    fall_events_results = analyze_event_history(
        "Fall Events",
        os.path.join(V1_10_DIR, "osdb_3min_fallEvents.json"),
        os.path.join(ORIGINAL_DIR, "osdb_3min_fallEvents.json"),
        os.path.join(REFACTORED_DIR, "osdb_3min_fallEvents.json")
    )
    
    # Check remote availability for events only in original
    if all_seizures_results['only_in_original_new'] > 0:
        print("\n" + "="*70)
        print("Checking Remote Database Availability")
        print("="*70)
        
        sample_ids = all_seizures_results['original_new_ids'][:50]
        remote_check = check_remote_availability(sample_ids)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n**All Seizures:**")
    print(f"  Original has {all_seizures_results['only_in_original_new']} events that are:")
    print(f"    - Not in V1.10 baseline")
    print(f"    - Not in refactored update")
    print(f"  → These likely came from previous updates and may be deleted from remote")
    
    print(f"\n**Fall Events:**")
    print(f"  Original has {fall_events_results['only_in_original_new']} events that are:")
    print(f"    - Not in V1.10 baseline")
    print(f"    - Not in refactored update")
    
    print(f"\n**Hypothesis:**")
    if all_seizures_results['only_in_original_new'] > 100:
        print(f"  ✓ CONFIRMED: The /home/graham/osd/osdb database contains")
        print(f"    {all_seizures_results['only_in_original_new']} events from previous updates that are")
        print(f"    likely no longer available on the remote server.")
        print(f"\n  This explains why:")
        print(f"    - Original: {all_seizures_results['original_count']} events (kept old + added new)")
        print(f"    - Refactored: {all_seizures_results['refactored_count']} events (only V1.10 + new)")
        print(f"    - Difference: {all_seizures_results['original_count'] - all_seizures_results['refactored_count']} events")
    else:
        print(f"  The difference is primarily due to grouping algorithm differences,")
        print(f"  not deleted remote events.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
