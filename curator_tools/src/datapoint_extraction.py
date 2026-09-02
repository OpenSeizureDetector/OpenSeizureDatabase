#!/usr/bin/env python3
"""
datapoint_extraction.py

Extract acceleration data from nested dataJSON structures in remote server datapoints.

The remote OSD API server returns datapoints with deeply nested JSON:
- datapoint.dataJSON is a JSON string
- Inside that JSON is another dataJSON field (string)
- That inner field contains the actual rawData/rawData3D arrays

This module extracts this nested data and flattens it for database storage.

Example nested structure:
{
  "dataJSON": "{ \"dataJSON\": \"{ \\\"rawData\\\": [1080.97, 700.04, ...], ... }\" }"
}

After extraction:
{
  "dataJSON": "...",  # Keep original for reference
  "rawData": [1080.97, 700.04, ...],  # Extracted to top level
  "rawData3D": [-838, 640, ...],
  "hr": 0,
  "o2Sat": 0
}
"""

import json
from typing import Dict, List, Any, Optional, Tuple


def extract_nested_datapoint_data(datapoint: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """
    Extract acceleration data from nested dataJSON structure in remote server datapoints.
    
    The remote OSD API returns datapoints with acceleration data buried in a 
    nested JSON structure (dataJSON.dataJSON). This function extracts that data
    and adds it to the datapoint for database storage.
    
    Parameters:
    -----------
    datapoint : Dict
        Datapoint dictionary from remote server
    debug : bool
        Enable debug output
        
    Returns:
    --------
    Dict : Datapoint with extracted acceleration data added
    
    Notes:
    ------
    - This function is idempotent (safe to call multiple times)
    - Handles both server datapoints (nested) and JSON file datapoints (already extracted)
    - Gracefully skips datapoints that don't have nested structure
    - Preserves original dataJSON field for reference
    """
    
    # Skip if no dataJSON field
    if 'dataJSON' not in datapoint or not datapoint['dataJSON']:
        if debug:
            print(f"  No dataJSON field to extract")
        return datapoint
    
    try:
        # Parse outer dataJSON
        outer_json_str = datapoint['dataJSON']
        if not isinstance(outer_json_str, str):
            if debug:
                print(f"  dataJSON is not string, skipping extraction")
            return datapoint
        
        outer = json.loads(outer_json_str)
        
        # Check for inner dataJSON (double-nested structure)
        if 'dataJSON' not in outer:
            if debug:
                print(f"  No nested dataJSON found, structure already extracted")
            return datapoint
        
        inner_json_raw = outer['dataJSON']
        if not isinstance(inner_json_raw, str):
            if debug:
                print(f"  Inner dataJSON is not string")
            return datapoint
        
        # Parse the inner JSON
        try:
            inner = json.loads(inner_json_raw)
        except json.JSONDecodeError as e:
            if debug:
                print(f"  Failed to parse inner dataJSON: {e}")
            return datapoint
        
        # Extract acceleration data fields
        extracted_fields = {}
        
        if 'rawData' in inner:
            extracted_fields['rawData'] = inner['rawData']
            if debug:
                print(f"  Extracted rawData: {len(inner['rawData'])} samples")
        
        if 'rawData3D' in inner:
            extracted_fields['rawData3D'] = inner['rawData3D']
            if debug:
                print(f"  Extracted rawData3D: {len(inner['rawData3D'])} samples")
        
        # Extract vital signs (only if non-zero/non-null)
        if 'hr' in inner and inner['hr'] is not None:
            # Skip default zero value that indicates no data
            if isinstance(inner['hr'], (int, float)) and inner['hr'] != 0:
                extracted_fields['hr'] = inner['hr']
                if debug:
                    print(f"  Extracted hr: {inner['hr']}")
        
        if 'o2Sat' in inner and inner['o2Sat'] is not None:
            if isinstance(inner['o2Sat'], (int, float)) and inner['o2Sat'] != 0:
                extracted_fields['o2Sat'] = inner['o2Sat']
                if debug:
                    print(f"  Extracted o2Sat: {inner['o2Sat']}")
        
        # Add extracted fields to datapoint
        if extracted_fields:
            for key, value in extracted_fields.items():
                datapoint[key] = value
            if debug:
                print(f"  Successfully extracted {len(extracted_fields)} fields")
        else:
            if debug:
                print(f"  No extractable fields found in nested structure")
        
        return datapoint
        
    except (json.JSONDecodeError, TypeError, AttributeError, KeyError) as e:
        # If parsing fails for any reason, continue with datapoint as-is
        if debug:
            print(f"  Error extracting nested data: {e}")
        return datapoint


def extract_nested_data_from_events(events: List[Dict[str, Any]], 
                                    debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract acceleration data from all datapoints in a batch of events.
    
    Processes all events and their datapoints, extracting nested acceleration data
    from remote server responses.
    
    Parameters:
    -----------
    events : List[Dict]
        List of event dictionaries
    debug : bool
        Enable debug output
        
    Returns:
    --------
    Tuple : (processed_events, extraction_stats)
    - processed_events: List of events with extracted datapoints
    - extraction_stats: Dictionary with extraction statistics
    """
    
    extraction_stats = {
        'events_processed': 0,
        'events_with_datapoints': 0,
        'datapoints_processed': 0,
        'datapoints_extracted': 0,
        'extraction_errors': 0,
    }
    
    for event in events:
        extraction_stats['events_processed'] += 1
        
        if 'datapoints' not in event or not isinstance(event['datapoints'], list):
            continue
        
        extraction_stats['events_with_datapoints'] += 1
        
        for dp in event['datapoints']:
            if not isinstance(dp, dict):
                continue
            
            extraction_stats['datapoints_processed'] += 1
            
            # Try to extract nested data
            before_keys = set(dp.keys())
            dp = extract_nested_datapoint_data(dp, debug=debug)
            after_keys = set(dp.keys())
            
            # Check if extraction added new fields
            if after_keys > before_keys:
                extraction_stats['datapoints_extracted'] += 1
    
    return events, extraction_stats


__all__ = [
    'extract_nested_datapoint_data',
    'extract_nested_data_from_events'
]
