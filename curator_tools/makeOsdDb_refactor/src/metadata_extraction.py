#!/usr/bin/env python3
"""
metadata_extraction.py

Extracts metadata fields from remote server events that are encoded in dataJSON.

The remote server returns events with metadata fields encoded in a nested dataJSON
field. This module extracts those fields so they can be properly stored in the database
and exported in CSV index files.

Key Fields Extracted:
- dataSourceName: Data source (e.g., "Watch", "Phone", "AndroidWear")
- phoneAppVersion: Phone app version number
- watchSdVersion: Watch SD version
- watchFwVersion: Watch firmware version
- watchSdName: Watch SD name
- watchPartNo: Watch part number
- watchSerialNo: Watch serial number
"""

import json
from typing import Dict, List, Any, Tuple, Optional


def extract_json_value(row: Dict[str, Any], field_name: str, debug: bool = False) -> Optional[Any]:
    """
    Extract the value of a field from the JSON string in the 'dataJSON' element
    of an event dictionary. Returns the value or None if not found.
    
    This mirrors the extractJsonVal function from the original makeOsdDb.py.
    
    Args:
        row: Event dictionary containing 'dataJSON' field
        field_name: Name of field to extract from dataJSON
        debug: Print debug information
        
    Returns:
        The value of the field from dataJSON, or None if not found
    """
    if debug:
        print(f"extract_json_value(): row keys={list(row.keys())}")
    
    dataJSON = row.get('dataJSON')
    
    if dataJSON is None:
        if debug:
            print(f"extract_json_value(): dataJSON is None")
        return None
    
    if not dataJSON:
        if debug:
            print(f"extract_json_value(): dataJSON is empty")
        return None
    
    try:
        if isinstance(dataJSON, str):
            if debug:
                print(f"extract_json_value(): Parsing JSON string")
            dataObj = json.loads(dataJSON)
        else:
            # Already a dict
            if debug:
                print(f"extract_json_value(): dataJSON already a dict")
            dataObj = dataJSON
        
        if field_name in dataObj:
            elem_val = dataObj[field_name]
            if debug:
                print(f"extract_json_value(): Found {field_name}={elem_val}")
            return elem_val
        else:
            if debug:
                print(f"extract_json_value(): Field {field_name} not in dataJSON")
            return None
            
    except Exception as e:
        if debug:
            print(f"extract_json_value(): Error parsing dataJSON: {e}")
        return None


def extract_metadata_from_events(events: List[Dict[str, Any]], debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract metadata fields from events that are stored in dataJSON.
    
    For each event, extracts these fields from dataJSON and adds them to the
    event dictionary at the top level (if not already present):
    - dataSourceName
    - phoneAppVersion
    - watchSdVersion
    - watchFwVersion
    - watchSdName
    - watchPartNo
    - watchSerialNo
    
    Args:
        events: List of event dictionaries from remote server
        debug: Print debug information
        
    Returns:
        Tuple of (updated_events, statistics_dict) where:
        - updated_events: List of events with extracted metadata
        - statistics_dict: Dict with extraction statistics
    """
    
    # Fields to extract from dataJSON
    metadata_fields = [
        'dataSourceName',
        'phoneAppVersion',
        'watchSdVersion',
        'watchFwVersion',
        'watchSdName',
        'watchPartNo',
        'watchSerialNo'
    ]
    
    stats = {
        'events_processed': 0,
        'events_with_dataJSON': 0,
        'events_updated': 0,
        'fields_extracted': {field: 0 for field in metadata_fields},
        'missing_dataJSON': []
    }
    
    updated_events = []
    
    for event in events:
        stats['events_processed'] += 1
        
        if 'dataJSON' not in event:
            stats['missing_dataJSON'].append(event.get('id', 'unknown'))
            updated_events.append(event)
            continue
        
        stats['events_with_dataJSON'] += 1
        event_updated = False
        
        # Extract each metadata field from dataJSON
        for field in metadata_fields:
            # Only extract if not already present at top level
            if field not in event or event[field] is None:
                value = extract_json_value(event, field, debug=debug)
                if value is not None:
                    event[field] = value
                    stats['fields_extracted'][field] += 1
                    event_updated = True
        
        if event_updated:
            stats['events_updated'] += 1
        
        updated_events.append(event)
    
    return updated_events, stats


def extract_metadata_from_dict_list(events: List[Dict[str, Any]], debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Alias for extract_metadata_from_events for backwards compatibility.
    
    Args:
        events: List of event dictionaries
        debug: Print debug information
        
    Returns:
        Tuple of (updated_events, statistics_dict)
    """
    return extract_metadata_from_events(events, debug=debug)


def print_extraction_summary(stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary of metadata extraction statistics.
    
    Args:
        stats: Statistics dictionary from extract_metadata_from_events
    """
    print("\n" + "="*70)
    print("Metadata Extraction Summary")
    print("="*70)
    
    print(f"\nProcessing:")
    print(f"  Events processed: {stats['events_processed']}")
    print(f"  Events with dataJSON: {stats['events_with_dataJSON']}")
    print(f"  Events updated: {stats['events_updated']}")
    
    if stats['missing_dataJSON']:
        print(f"\n  Events without dataJSON: {len(stats['missing_dataJSON'])}")
        if len(stats['missing_dataJSON']) <= 10:
            for event_id in stats['missing_dataJSON']:
                print(f"    - Event {event_id}")
    
    print(f"\nFields Extracted:")
    for field, count in stats['fields_extracted'].items():
        print(f"  {field}: {count} events")
    
    print("="*70 + "\n")
