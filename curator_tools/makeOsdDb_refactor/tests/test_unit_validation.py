#!/usr/bin/env python3
"""
test_unit_validation.py

Unit tests for event_validation.py module.

Run with: pytest test_unit_validation.py -v
Or: python3 test_unit_validation.py
"""

import sys
import os
import pytest

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_validation import (
    EventValidationError,
    validate_event,
    validate_events_batch
)


class TestEventValidationError:
    """Test EventValidationError exception."""
    
    def test_exception_creation(self):
        """Test creating EventValidationError."""
        err = EventValidationError(123, "Test reason", {'detail': 'value'})
        assert err.event_id == 123
        assert err.reason == "Test reason"
        assert err.details == {'detail': 'value'}
        assert "Event 123" in str(err)
        assert "Test reason" in str(err)


class TestValidateEvent:
    """Test validate_event function."""
    
    def test_valid_event(self):
        """Test validating a valid event."""
        event = {
            'id': 1,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2,
            'datapoints': [{'time': 0}, {'time': 1}]
        }
        # Should not raise exception
        validate_event(event, min_datapoints=1)
    
    def test_missing_required_field(self):
        """Test event missing required field."""
        event = {
            'id': 1,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            # Missing osdAlarmState
            'datapoints': [{'time': 0}]
        }
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert exc_info.value.reason == "Missing required fields"
        assert 'osdAlarmState' in exc_info.value.details['missing_fields']
    
    def test_no_datapoints_field(self):
        """Test event without datapoints field."""
        event = {
            'id': 2,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2
            # No datapoints field
        }
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert exc_info.value.reason == "Event has no datapoints field"
    
    def test_datapoints_not_list(self):
        """Test event with datapoints not a list."""
        event = {
            'id': 3,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2,
            'datapoints': "not a list"
        }
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event)
        assert exc_info.value.reason == "Event datapoints is not a list"
    
    def test_insufficient_datapoints(self):
        """Test event with insufficient datapoints."""
        event = {
            'id': 4,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2,
            'datapoints': []  # Empty
        }
        with pytest.raises(EventValidationError) as exc_info:
            validate_event(event, min_datapoints=1)
        assert exc_info.value.reason == "Event has insufficient datapoints"
        assert exc_info.value.details['count'] == 0
        assert exc_info.value.details['minimum'] == 1
    
    def test_custom_required_fields(self):
        """Test validation with custom required fields."""
        event = {
            'id': 5,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'datapoints': [{'time': 0}]
        }
        # Should fail because osdAlarmState is missing
        with pytest.raises(EventValidationError):
            validate_event(event, required_fields=['id', 'userId', 'osdAlarmState', 'datapoints'])
    
    def test_minimum_datapoints_threshold(self):
        """Test different minimum datapoints thresholds."""
        event = {
            'id': 6,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure',
            'osdAlarmState': 2,
            'datapoints': [{'time': 0}, {'time': 1}, {'time': 2}]  # 3 datapoints
        }
        # Should pass with min=1
        validate_event(event, min_datapoints=1)
        # Should pass with min=3
        validate_event(event, min_datapoints=3)
        # Should fail with min=4
        with pytest.raises(EventValidationError):
            validate_event(event, min_datapoints=4)


class TestValidateEventsBatch:
    """Test validate_events_batch function."""
    
    def test_all_valid_events(self):
        """Test batch with all valid events."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2022-01-01T12:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [{'time': 0}]
            },
            {
                'id': 2,
                'userId': 101,
                'dataTime': '2022-01-01T13:00:00Z',
                'type': 'False Alarm',
                'osdAlarmState': 1,
                'datapoints': [{'time': 0}, {'time': 1}]
            }
        ]
        valid, report = validate_events_batch(events, show_progress=False)
        
        assert len(valid) == 2
        assert report['total_checked'] == 2
        assert report['valid'] == 2
        assert report['skipped'] == 0
    
    def test_mixed_valid_invalid(self):
        """Test batch with mix of valid and invalid events."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2022-01-01T12:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [{'time': 0}]
            },
            {
                'id': 2,
                'userId': 100,
                'dataTime': '2022-01-01T12:05:00Z',
                'type': 'Seizure',
                'osdAlarmState': 1,
                'datapoints': []  # No datapoints - invalid
            },
            {
                'id': 3,
                'userId': 101,
                'dataTime': '2022-01-01T13:00:00Z',
                'type': 'False Alarm',
                # Missing osdAlarmState - invalid
                'datapoints': [{'time': 0}]
            }
        ]
        valid, report = validate_events_batch(events, show_progress=False)
        
        assert len(valid) == 1
        assert valid[0]['id'] == 1
        assert report['total_checked'] == 3
        assert report['valid'] == 1
        assert report['skipped'] == 2
        assert len(report['skip_reasons']) == 2
    
    def test_invalid_event_ids_config(self):
        """Test events marked invalid in config are skipped."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2022-01-01T12:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [{'time': 0}]
            },
            {
                'id': 2,
                'userId': 101,
                'dataTime': '2022-01-01T13:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [{'time': 0}]
            }
        ]
        # Mark event 2 as invalid
        valid, report = validate_events_batch(events, invalid_event_ids=[2], show_progress=False)
        
        assert len(valid) == 1
        assert valid[0]['id'] == 1
        assert report['skipped'] == 1
        assert 'marked_invalid_in_config' in report['skip_reasons']
        assert report['skip_reasons']['marked_invalid_in_config'] == 1
    
    def test_empty_batch(self):
        """Test validating empty batch."""
        valid, report = validate_events_batch([], show_progress=False)
        
        assert len(valid) == 0
        assert report['total_checked'] == 0
        assert report['valid'] == 0
        assert report['skipped'] == 0
    
    def test_skip_reason_categorization(self):
        """Test that skip reasons are correctly categorized."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2022-01-01T12:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': []  # Insufficient datapoints
            },
            {
                'id': 2,
                'userId': 100,
                'dataTime': '2022-01-01T12:05:00Z',
                'type': 'Seizure',
                # Missing osdAlarmState
                'datapoints': [{'time': 0}]
            },
            {
                'id': 3,
                'userId': 101,
                'dataTime': '2022-01-01T13:00:00Z',
                'type': 'False Alarm',
                'osdAlarmState': 1
                # Missing datapoints field
            }
        ]
        valid, report = validate_events_batch(events, show_progress=False)
        
        assert report['valid'] == 0
        assert report['skipped'] == 3
        assert 'Event has insufficient datapoints' in report['skip_reasons']
        assert 'Missing required fields' in report['skip_reasons']
        assert 'Event has no datapoints field' in report['skip_reasons']


# Allow running tests without pytest
if __name__ == '__main__':
    # Try to use pytest
    try:
        pytest.main([__file__, '-v'])
    except:
        # Fallback to manual testing
        print("Running manual tests (pytest not available)...")
        
        # Test EventValidationError
        print("\n1. Testing EventValidationError...")
        err = EventValidationError(123, "Test", {})
        print(f"   ✓ Created exception: {err}")
        
        # Test valid event
        print("\n2. Testing valid event...")
        valid_event = {
            'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure', 'osdAlarmState': 2, 'datapoints': [{'time': 0}]
        }
        try:
            validate_event(valid_event)
            print("   ✓ Valid event passed")
        except Exception as e:
            print(f"   ✗ Valid event failed: {e}")
        
        # Test invalid event
        print("\n3. Testing invalid event (no datapoints)...")
        invalid_event = {
            'id': 2, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure', 'osdAlarmState': 2, 'datapoints': []
        }
        try:
            validate_event(invalid_event)
            print("   ✗ Invalid event should have failed")
        except EventValidationError as e:
            print(f"   ✓ Invalid event correctly rejected: {e.reason}")
        
        # Test batch validation
        print("\n4. Testing batch validation...")
        events = [valid_event, invalid_event]
        valid, report = validate_events_batch(events, show_progress=False)
        print(f"   ✓ Batch: {report['valid']} valid, {report['skipped']} skipped")
        
        print("\n✅ All manual tests completed")
