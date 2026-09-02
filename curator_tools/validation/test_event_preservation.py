#!/usr/bin/env python3
"""
Test script to verify that existing event IDs are preserved during grouping.
Tests specifically for the 4 events that were completely lost (5486, 6590, 6668, 21569).
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from event_grouping import apply_sliding_window_grouping

def test_event_preservation():
    """
    Test that existing events are preserved when grouped.
    """
    print("=" * 80)
    print("Testing Event ID Preservation During Grouping")
    print("=" * 80)
    
    # Load the actual events from the test database
    source_file = "/home/graham/osd/osdb/osdb_3min_allSeizures.json"
    
    print(f"\nLoading events from {source_file}...")
    with open(source_file, 'r') as f:
        all_events = json.load(f)
    
    print(f"Loaded {len(all_events)} events")
    
    # The 4 events that were lost in previous test
    lost_event_ids = [5486, 6590, 6668, 21569]
    
    # Check if these events exist in source
    event_map = {int(e['id']): e for e in all_events}
    
    print(f"\nChecking for the 4 previously lost events:")
    for eid in lost_event_ids:
        if eid in event_map:
            event = event_map[eid]
            print(f"  ✓ Event {eid} found - User {event['userId']}, {event['dataTime']}")
        else:
            print(f"  ✗ Event {eid} NOT in source database")
    
    # Mark ALL events as existing (they're all from published database)
    for event in all_events:
        event['_is_existing_event'] = True
    
    print(f"\n{'-'*80}")
    print("Running grouping with event preservation...")
    print(f"{'-'*80}\n")
    
    # Run the grouping with the fix
    grouped_events, grouping_info = apply_sliding_window_grouping(
        all_events,
        time_threshold='3min',
        selection_strategy='alarm_first',
        concatenate_datapoints_flag=True,
        debug=False,
        show_progress=False
    )
    
    print(f"\n{'-'*80}")
    print("Grouping Results")
    print(f"{'-'*80}")
    print(f"Input events: {grouping_info['total_input_events']}")
    print(f"Output events: {grouping_info['total_output_events']}")
    print(f"Events merged: {grouping_info['total_input_events'] - grouping_info['total_output_events']}")
    print(f"Existing events tracked: {grouping_info['existing_events_tracked']}")
    print(f"Existing events preserved: {grouping_info['existing_events_preserved']}")
    
    lost_existing = grouping_info.get('lost_existing_events', [])
    if lost_existing:
        print(f"\n⚠️  LOST existing events: {len(lost_existing)}")
        print(f"  IDs: {lost_existing[:20]}")
    else:
        print(f"\n✓ All existing events preserved!")
    
    # Specifically check if our 4 previously lost events are preserved
    print(f"\n{'-'*80}")
    print("Checking Previously Lost Events")
    print(f"{'-'*80}")
    
    # Build map of grouped events
    grouped_map = {int(e['id']): e for e in grouped_events}
    
    # Also build map of all preserved IDs (including merged_from)
    all_preserved_ids = set()
    for event in grouped_events:
        all_preserved_ids.add(int(event['id']))
        merged_from = event.get('_merged_from_event_ids', [])
        # Normalize to list (handle legacy formats)
        if merged_from is None:
            merged_from = []
        elif not isinstance(merged_from, list):
            merged_from = [merged_from]
        all_preserved_ids.update(int(mid) for mid in merged_from)
    
    for eid in lost_event_ids:
        if eid in grouped_map:
            event = grouped_map[eid]
            merged_count = event.get('_merged_event_count', 1)
            print(f"  ✓ Event {eid} PRESERVED as primary event")
            if merged_count > 1:
                print(f"    - Merged {merged_count} events together")
                merged_ids = event.get('_merged_from_event_ids', [])
                # Normalize to list
                if merged_ids is None:
                    merged_ids = []
                elif not isinstance(merged_ids, list):
                    merged_ids = [merged_ids]
                print(f"    - Merged from: {merged_ids[:10]}")
        elif eid in all_preserved_ids:
            # Find which event it was merged into
            for event in grouped_events:
                merged_from = event.get('_merged_from_event_ids', [])
                # Normalize to list
                if merged_from is None:
                    merged_from = []
                elif not isinstance(merged_from, list):
                    merged_from = [merged_from]
                if eid in merged_from:
                    print(f"  ✓ Event {eid} PRESERVED (merged into Event {event['id']})")
                    break
        else:
            print(f"  ✗ Event {eid} LOST - NOT preserved!")
    
    print(f"\n{'='*80}")
    
    # Return status
    if lost_existing:
        print("❌ TEST FAILED - Some existing events were lost")
        return False
    else:
        print("✅ TEST PASSED - All existing events preserved")
        return True


if __name__ == '__main__':
    success = test_event_preservation()
    sys.exit(0 if success else 1)
