#!/usr/bin/env python3
"""
datetime_normalization.py

Standardize datetime formats across OSDB events and datapoints.
Converts older "DD-MM-YYYY HH:MM:SS" format to ISO 8601 "YYYY-MM-DDTHH:MM:SSZ" format.
"""

from datetime import datetime
from typing import Dict, Any, List
from dateutil import parser as dateutil_parser


# Target format: ISO 8601 with UTC timezone
TARGET_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Known formats in OSDB data
KNOWN_FORMATS = [
    "%d-%m-%Y %H:%M:%S",       # Old format: "02-10-2022 13:44:56"
    "%Y-%m-%dT%H:%M:%SZ",      # New format: "2024-07-12T05:58:24Z"
    "%Y-%m-%d %H:%M:%S",       # Alternative: "2024-07-12 05:58:24"
]


def normalize_datetime_string(dt_str: str) -> str:
    """
    Convert any datetime string to ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
    
    Args:
        dt_str: Datetime string in various formats
        
    Returns:
        Normalized datetime string in ISO 8601 format
        
    Raises:
        ValueError: If datetime string cannot be parsed
    """
    if not dt_str or not isinstance(dt_str, str):
        return dt_str
    
    # If already in target format, return as-is
    if len(dt_str) == 20 and dt_str[10] == 'T' and dt_str[-1] == 'Z':
        try:
            # Validate it's actually a valid datetime
            datetime.strptime(dt_str, TARGET_FORMAT)
            return dt_str
        except:
            pass
    
    # Try known formats first (faster)
    for fmt in KNOWN_FORMATS:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.strftime(TARGET_FORMAT)
        except:
            continue
    
    # Fall back to dateutil parser (handles many formats)
    try:
        dt = dateutil_parser.parse(dt_str, dayfirst=True)
        return dt.strftime(TARGET_FORMAT)
    except Exception as e:
        raise ValueError(f"Could not parse datetime string: '{dt_str}': {e}")


def normalize_event_datetimes(event: Dict[str, Any], 
                               normalize_datapoints: bool = True,
                               in_place: bool = False) -> Dict[str, Any]:
    """
    Normalize all datetime fields in an event to ISO 8601 format.
    
    Args:
        event: Event dictionary
        normalize_datapoints: Whether to normalize datapoint times too
        in_place: Modify event in place or create a copy
        
    Returns:
        Event with normalized datetime fields
    """
    if not in_place:
        import copy
        event = copy.deepcopy(event)
    
    # Normalize event-level dataTime
    if 'dataTime' in event and isinstance(event['dataTime'], str):
        try:
            event['dataTime'] = normalize_datetime_string(event['dataTime'])
        except ValueError as e:
            # Log warning but don't fail
            print(f"Warning: Could not normalize event {event.get('id')} dataTime: {e}")
    
    # Normalize datapoint times
    if normalize_datapoints and 'datapoints' in event and event['datapoints']:
        for dp in event['datapoints']:
            if 'dataTime' in dp and isinstance(dp['dataTime'], str):
                try:
                    dp['dataTime'] = normalize_datetime_string(dp['dataTime'])
                except ValueError:
                    # Silently skip unparseable datapoint times
                    pass
            
            # Some datapoints might have 't' or 'time' fields instead
            if 'time' in dp and isinstance(dp['time'], str):
                try:
                    dp['time'] = normalize_datetime_string(dp['time'])
                except ValueError:
                    pass
    
    return event


def normalize_events_batch(events: List[Dict[str, Any]],
                           normalize_datapoints: bool = True,
                           show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Normalize datetime fields for a batch of events.
    
    Args:
        events: List of event dictionaries
        normalize_datapoints: Whether to normalize datapoint times
        show_progress: Show progress bar
        
    Returns:
        List of events with normalized datetime fields
    """
    try:
        from tqdm import tqdm
        progress_wrapper = tqdm if show_progress else lambda x, **kwargs: x
    except ImportError:
        progress_wrapper = lambda x, **kwargs: x
    
    normalized_events = []
    stats = {'events_normalized': 0, 'datapoints_normalized': 0, 'errors': 0}
    
    for event in progress_wrapper(events, desc="Normalizing datetimes", unit="event"):
        try:
            # Count datapoints before normalization
            original_datapoints = event.get('datapoints', [])
            
            normalized = normalize_event_datetimes(
                event, 
                normalize_datapoints=normalize_datapoints,
                in_place=False
            )
            
            stats['events_normalized'] += 1
            if normalize_datapoints:
                stats['datapoints_normalized'] += len(original_datapoints)
            
            normalized_events.append(normalized)
        except Exception as e:
            print(f"Error normalizing event {event.get('id')}: {e}")
            stats['errors'] += 1
            # Include original event on error
            normalized_events.append(event)
    
    return normalized_events, stats


def detect_datetime_formats(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Analyze events to detect which datetime formats are present.
    
    Args:
        events: List of event dictionaries
        
    Returns:
        Dictionary with format names and counts
    """
    format_counts = {
        'iso_8601': 0,        # YYYY-MM-DDTHH:MM:SSZ
        'old_format': 0,      # DD-MM-YYYY HH:MM:SS
        'other': 0,
        'missing': 0
    }
    
    for event in events:
        dt_str = event.get('dataTime')
        
        if not dt_str:
            format_counts['missing'] += 1
        elif isinstance(dt_str, str):
            if len(dt_str) == 20 and dt_str[10] == 'T' and dt_str[-1] == 'Z':
                format_counts['iso_8601'] += 1
            elif len(dt_str) == 19 and dt_str[2] == '-' and dt_str[5] == '-':
                format_counts['old_format'] += 1
            else:
                format_counts['other'] += 1
        else:
            format_counts['other'] += 1
    
    return format_counts


# Example usage and testing
if __name__ == '__main__':
    # Test cases
    test_cases = [
        "02-10-2022 13:44:56",      # Old format
        "2024-07-12T05:58:24Z",     # New format (should remain unchanged)
        "04-05-2022 15:33:56",      # Old format
        "2022-10-27T03:36:01Z",     # New format
    ]
    
    print("Testing datetime normalization:")
    print("-" * 60)
    
    for test_str in test_cases:
        try:
            normalized = normalize_datetime_string(test_str)
            print(f"'{test_str}'")
            print(f"  -> '{normalized}'")
            print()
        except Exception as e:
            print(f"'{test_str}'")
            print(f"  -> ERROR: {e}")
            print()
    
    # Test event normalization
    print("\nTesting event normalization:")
    print("-" * 60)
    
    test_event = {
        'id': 12345,
        'dataTime': '02-10-2022 13:44:56',
        'datapoints': [
            {'dataTime': '02-10-2022 13:43:44', 'hr': 63},
            {'dataTime': '02-10-2022 13:43:45', 'hr': 64},
        ]
    }
    
    print(f"Original event dataTime: {test_event['dataTime']}")
    print(f"Original datapoint times: {[dp['dataTime'] for dp in test_event['datapoints']]}")
    
    normalized = normalize_event_datetimes(test_event, normalize_datapoints=True)
    
    print(f"\nNormalized event dataTime: {normalized['dataTime']}")
    print(f"Normalized datapoint times: {[dp['dataTime'] for dp in normalized['datapoints']]}")
