#!/usr/bin/env python3
"""
event_validation.py

Event validation module with clean error reporting.
Provides early validation of events with summary statistics instead of noisy errors.

Key Features:
- EventValidationError exception for validation failures
- validate_event() function for individual event validation
- Batch validation with summary statistics
- Detailed skip reports saved to JSON files
- Clean terminal output

"""

import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple progress indicator
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc or ""
            self.total = total or (len(iterable) if iterable else 0)
            self.n = 0
        
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self.n += n


class EventValidationError(Exception):
    """
    Exception raised when an event fails validation.
    
    Attributes:
        event_id: ID of the invalid event
        reason: Human-readable reason for validation failure
        details: Additional details about the validation failure
    """
    
    def __init__(self, event_id: int, reason: str, details: Optional[Dict] = None):
        self.event_id = event_id
        self.reason = reason
        self.details = details or {}
        message = f"Event {event_id}: {reason}"
        if details:
            message += f" - {details}"
        super().__init__(message)


def validate_event(event: Dict[str, Any], 
                   min_datapoints: int = 1,
                   required_fields: Optional[List[str]] = None,
                   debug: bool = False) -> None:
    """
    Validate a single event and raise EventValidationError if invalid.
    
    Args:
        event: Event dictionary to validate
        min_datapoints: Minimum number of datapoints required
        required_fields: List of required field names
        debug: Print debug information
        
    Raises:
        EventValidationError: If event fails validation
    """
    if required_fields is None:
        required_fields = ['id', 'userId', 'dataTime', 'type', 'osdAlarmState']
    
    event_id = event.get('id', 'unknown')
    
    # Check required fields
    missing = [f for f in required_fields if f not in event]
    if missing:
        raise EventValidationError(
            event_id,
            "Missing required fields",
            {'missing_fields': missing}
        )
    
    # Check datapoints field exists
    if 'datapoints' not in event:
        raise EventValidationError(
            event_id,
            "Event has no datapoints field",
            {}
        )
    
    # Check datapoints is a list
    if not isinstance(event['datapoints'], list):
        raise EventValidationError(
            event_id,
            "Event datapoints is not a list",
            {'type': str(type(event['datapoints']))}
        )
    
    # Check minimum datapoints
    dp_count = len(event['datapoints'])
    if dp_count < min_datapoints:
        raise EventValidationError(
            event_id,
            "Event has insufficient datapoints",
            {'count': dp_count, 'minimum': min_datapoints}
        )
    
    if debug:
        print(f"✓ Event {event_id} validated successfully ({dp_count} datapoints)")


def validate_events_batch(events: List[Dict[str, Any]],
                          min_datapoints: int = 1,
                          required_fields: Optional[List[str]] = None,
                          invalid_event_ids: Optional[List[int]] = None,
                          debug: bool = False,
                          show_progress: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Validate a batch of events and return valid events with summary statistics.
    
    Args:
        events: List of event dictionaries to validate
        min_datapoints: Minimum number of datapoints required
        required_fields: List of required field names
        invalid_event_ids: List of event IDs marked invalid in config
        debug: Print debug information
        show_progress: Show progress bar during validation
        
    Returns:
        Tuple of (valid_events, validation_report)
        - valid_events: List of events that passed validation
        - validation_report: Dictionary with summary statistics
    """
    if invalid_event_ids is None:
        invalid_event_ids = []
    
    valid_events = []
    skipped_events = []
    skip_reasons = defaultdict(list)
    
    # Use progress bar if available and requested
    iterator = tqdm(events, desc="Validating events", unit="event", disable=not show_progress) if show_progress else events
    
    for event in iterator:
        event_id = event.get('id', 'unknown')
        
        # Check if marked invalid in config
        if event_id in invalid_event_ids:
            skip_reasons['marked_invalid_in_config'].append(event_id)
            skipped_events.append({
                'event': event,
                'reason': 'marked_invalid_in_config',
                'details': {}
            })
            continue
        
        # Validate event
        try:
            validate_event(event, min_datapoints, required_fields, debug)
            valid_events.append(event)
        except EventValidationError as e:
            skip_reasons[e.reason].append(event_id)
            skipped_events.append({
                'event': event,
                'reason': e.reason,
                'details': e.details
            })
            if debug:
                print(f"⚠️  Skipped event {event_id}: {e.reason}")
    
    # Build validation report
    validation_report = {
        'total_checked': len(events),
        'valid': len(valid_events),
        'skipped': len(skipped_events),
        'skip_reasons': {reason: len(ids) for reason, ids in skip_reasons.items()},
        'skip_reason_details': dict(skip_reasons),
        'skipped_events': skipped_events
    }
    
    return valid_events, validation_report


def print_validation_summary(validation_report: Dict,
                            show_details: bool = False) -> None:
    """
    Print a clean validation summary to the terminal.
    
    Args:
        validation_report: Validation report from validate_events_batch()
        show_details: Show detailed event IDs (for debug mode)
    """
    total = validation_report['total_checked']
    valid = validation_report['valid']
    skipped = validation_report['skipped']
    
    print(f"\n{'='*60}")
    print("Event Validation Summary")
    print(f"{'='*60}")
    print(f"✓ Successfully validated: {valid} events")
    
    if skipped > 0:
        print(f"⚠️  Skipped: {skipped} events")
        print("   Reasons:")
        for reason, count in validation_report['skip_reasons'].items():
            print(f"     - {reason}: {count}")
        
        if show_details:
            print("\n   Skipped Event IDs:")
            for reason, ids in validation_report['skip_reason_details'].items():
                print(f"     {reason}: {ids[:10]}")  # Show first 10
                if len(ids) > 10:
                    print(f"       ... and {len(ids) - 10} more")
    
    print(f"{'='*60}\n")


def save_validation_report(validation_report: Dict,
                           output_dir: str = '.') -> str:
    """
    Save detailed validation report to a JSON file.
    
    Args:
        validation_report: Validation report from validate_events_batch()
        output_dir: Directory to save report file
        
    Returns:
        Path to saved report file
    """
    import os
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'validation_report_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Create simplified report for saving (exclude full event objects)
    save_report = {
        'timestamp': datetime.now().isoformat(),
        'total_checked': validation_report['total_checked'],
        'valid': validation_report['valid'],
        'skipped': validation_report['skipped'],
        'skip_reasons': validation_report['skip_reasons'],
        'skipped_event_details': [
            {
                'event_id': skip['event'].get('id'),
                'userId': skip['event'].get('userId'),
                'dataTime': skip['event'].get('dataTime'),
                'type': skip['event'].get('type'),
                'reason': skip['reason'],
                'details': skip['details']
            }
            for skip in validation_report['skipped_events']
        ]
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(save_report, f, indent=2)
    
    print(f"   Detailed report saved to: {filename}")
    
    return filepath


def download_and_validate_event(osd, 
                                event_id: int,
                                min_datapoints: int = 1,
                                max_retries: int = 3,
                                debug: bool = False) -> Optional[Dict]:
    """
    Download and validate an event from the server with retry logic.
    
    Args:
        osd: OsdDbConnection or WebApiConnection instance
        event_id: ID of event to download
        min_datapoints: Minimum number of datapoints required
        max_retries: Maximum number of retry attempts for network errors
        debug: Print debug information
        
    Returns:
        Event dictionary if valid, None if invalid
        
    Raises:
        Exception: For network errors after all retries exhausted
    """
    import time
    
    for attempt in range(max_retries):
        try:
            # Download event
            event = osd.getEvent(event_id, includeDatapoints=True)
            
            # Validate
            validate_event(event, min_datapoints=min_datapoints, debug=debug)
            
            return event
            
        except EventValidationError as e:
            # Validation failed - don't retry
            if debug:
                print(f"⚠️  Event {event_id} validation failed: {e.reason}")
            return None
            
        except Exception as e:
            # Network or other error - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if debug:
                    print(f"⚠️  Error downloading event {event_id} (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # All retries exhausted
                raise Exception(f"Failed to download event {event_id} after {max_retries} attempts: {e}")
    
    return None


# Example usage and testing
if __name__ == '__main__':
    import sys
    
    # Test with sample events
    sample_events = [
        {
            'id': 1,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2,
            'datapoints': [{'time': 0}, {'time': 1}]
        },
        {
            'id': 2,
            'userId': 100,
            'dataTime': '2022-01-01T12:05:00Z',
            'type': 'Seizure',
            'osdAlarmState': 1,
            'datapoints': []  # No datapoints - should fail
        },
        {
            'id': 3,
            'userId': 101,
            'dataTime': '2022-01-01T13:00:00Z',
            'type': 'False Alarm',
            # Missing 'osdAlarmState' - should fail
            'datapoints': [{'time': 0}]
        }
    ]
    
    print("Testing event validation...")
    valid, report = validate_events_batch(sample_events, min_datapoints=1, debug=True)
    
    print(f"\nResults:")
    print(f"  Valid events: {len(valid)}")
    print(f"  Skipped events: {report['skipped']}")
    print(f"  Skip reasons: {report['skip_reasons']}")
    
    print_validation_summary(report, show_details=True)
