#!/usr/bin/env python3
"""
event_deduplication.py

Event deduplication module for Phase 2.
Detects and removes duplicate events (same event downloaded multiple times).

"""

import hashlib
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def compute_event_hash(event: Dict[str, Any],
                       fields: List[str] = None) -> str:
    """
    Compute a hash for an event based on specific fields.
    
    Args:
        event: Event dictionary
        fields: List of fields to include in hash (default: id, userId, dataTime, type)
        
    Returns:
        Hash string
    """
    if fields is None:
        fields = ['id', 'userId', 'dataTime', 'type']
    
    # Create a consistent representation
    hash_data = {}
    for field in fields:
        if field in event:
            hash_data[field] = event[field]
    
    # Sort keys for consistency
    hash_str = json.dumps(hash_data, sort_keys=True)
    return hashlib.md5(hash_str.encode()).hexdigest()


def find_duplicate_events(events: List[Dict[str, Any]],
                          method: str = 'hash',
                          hash_fields: List[str] = None) -> Dict[str, List[Dict]]:
    """
    Find duplicate events in a list.
    
    Args:
        events: List of event dictionaries
        method: Deduplication method ('hash' or 'id')
        hash_fields: Fields to use for hash-based deduplication
        
    Returns:
        Dictionary mapping duplicate keys to lists of duplicate events
    """
    duplicates = defaultdict(list)
    
    if method == 'hash':
        for event in events:
            hash_key = compute_event_hash(event, hash_fields)
            duplicates[hash_key].append(event)
        
        # Keep only actual duplicates (more than one event)
        duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    
    elif method == 'id':
        # Simple ID-based deduplication
        id_map = defaultdict(list)
        for event in events:
            event_id = event.get('id')
            if event_id is not None:
                id_map[event_id].append(event)
        
        # Keep only actual duplicates
        duplicates = {str(k): v for k, v in id_map.items() if len(v) > 1}
    
    return duplicates


def remove_duplicate_events(events: List[Dict[str, Any]],
                            method: str = 'hash',
                            hash_fields: List[str] = None,
                            keep: str = 'first') -> Tuple[List[Dict], Dict]:
    """
    Remove duplicate events from a list.
    
    Args:
        events: List of event dictionaries
        method: Deduplication method ('hash' or 'id')
        hash_fields: Fields to use for hash-based deduplication
        keep: Which duplicate to keep ('first', 'last', 'most_datapoints')
        
    Returns:
        Tuple of (deduplicated_events, dedup_info)
        - deduplicated_events: List with duplicates removed
        - dedup_info: Dictionary with deduplication statistics
    """
    if not events:
        return [], {
            'total_input': 0,
            'total_output': 0,
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'removed_event_ids': [],
            'duplicate_groups': 0
        }
    
    # Find duplicates
    duplicates = find_duplicate_events(events, method, hash_fields)
    
    if not duplicates:
        return events, {
            'total_input': len(events),
            'total_output': len(events),
            'duplicates_found': 0,
            'duplicates_removed': 0,
            'removed_event_ids': [],
            'duplicate_groups': 0
        }
    
    # Track which events to keep
    seen = set()
    deduplicated = []
    removed_ids = []
    
    for event in events:
        # Compute key
        if method == 'hash':
            key = compute_event_hash(event, hash_fields)
        else:  # method == 'id'
            key = str(event.get('id'))
        
        if key in duplicates and key in seen:
            # This is a duplicate we've already seen
            removed_ids.append(event.get('id'))
            continue
        
        if key in duplicates:
            # This is the first occurrence of a duplicate
            dup_group = duplicates[key]
            
            # Select which one to keep
            if keep == 'first':
                selected = dup_group[0]
            elif keep == 'last':
                selected = dup_group[-1]
            elif keep == 'most_datapoints':
                selected = max(dup_group, key=lambda e: len(e.get('datapoints', [])))
            else:
                selected = dup_group[0]
            
            deduplicated.append(selected)
            seen.add(key)
            
            # Mark others as removed
            for dup_event in dup_group:
                if dup_event['id'] != selected['id']:
                    removed_ids.append(dup_event['id'])
        else:
            # Not a duplicate
            deduplicated.append(event)
            seen.add(key)
    
    dedup_info = {
        'total_input': len(events),
        'total_output': len(deduplicated),
        'duplicates_found': sum(len(v) for v in duplicates.values()),
        'duplicates_removed': len(removed_ids),
        'removed_event_ids': removed_ids,
        'duplicate_groups': len(duplicates)
    }
    
    return deduplicated, dedup_info


# Example usage and testing
if __name__ == '__main__':
    # Test with sample events
    sample_events = [
        {
            'id': 1,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'datapoints': [{'time': 0}, {'time': 1}]
        },
        {
            'id': 1,  # Duplicate ID
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'datapoints': [{'time': 0}, {'time': 1}]
        },
        {
            'id': 2,
            'userId': 101,
            'dataTime': '2022-01-01T13:00:00Z',
            'type': 'False Alarm',
            'datapoints': [{'time': 0}]
        }
    ]
    
    print("Testing event deduplication...")
    print(f"Input: {len(sample_events)} events")
    
    # Find duplicates
    dups = find_duplicate_events(sample_events, method='id')
    print(f"Duplicate groups found: {len(dups)}")
    
    # Remove duplicates
    deduplicated, info = remove_duplicate_events(sample_events, method='id')
    print(f"After deduplication: {len(deduplicated)} events")
    print(f"Removed: {info['duplicates_removed']} duplicates")
    print(f"Removed IDs: {info['removed_event_ids']}")
    
    if len(deduplicated) == 2 and info['duplicates_removed'] == 1:
        print("\n✓ Deduplication working correctly!")
    else:
        print("\n✗ Deduplication test failed")
