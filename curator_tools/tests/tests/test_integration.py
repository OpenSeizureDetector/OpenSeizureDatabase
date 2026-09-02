#!/usr/bin/env python3
"""
test_integration.py

Integration tests that test the full pipeline from validation through grouping.

Run with: pytest test_integration.py -v
Or: python3 test_integration.py
"""

import sys
import os
import pytest

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_validation import validate_events_batch
from event_grouping import apply_sliding_window_grouping


class TestValidationAndGroupingPipeline:
    """Test the full pipeline of validation then grouping."""
    
    def test_full_pipeline_with_valid_events(self):
        """Test validation followed by grouping with all valid events."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'osdAlarmState': 2,
                'desc': 'Real seizure',
                'datapoints': [{'time': 0}]
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:02:00Z',  # 2 min later - should group
                'osdAlarmState': 1,
                'desc': '',
                'datapoints': [{'time': 0}]
            },
            {
                'id': 3,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:10:00Z',  # 8 min from first - new group
                'osdAlarmState': 1,
                'desc': '',
                'datapoints': [{'time': 0}]
            }
        ]
        
        # Step 1: Validation
        valid_events, val_report = validate_events_batch(events, show_progress=False)
        assert val_report['valid'] == 3
        assert val_report['skipped'] == 0
        
        # Step 2: Grouping
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            show_progress=False
        )
        
        # Should have 2 groups: (1,2) and (3)
        assert group_info['total_groups'] == 2
        assert len(unique_events) == 2
        assert group_info['total_input_events'] == 3
        assert len(group_info['discarded_events']) == 1
        assert 2 in group_info['discarded_events']
    
    def test_full_pipeline_with_invalid_events(self):
        """Test validation filters out invalid events before grouping."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'osdAlarmState': 2,
                'desc': '',
                'datapoints': [{'time': 0}]
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:01:00Z',
                'osdAlarmState': 1,
                'desc': '',
                'datapoints': []  # INVALID - no datapoints
            },
            {
                'id': 3,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:02:00Z',
                'osdAlarmState': 1,
                'desc': '',
                'datapoints': [{'time': 0}]
            }
        ]
        
        # Step 1: Validation
        valid_events, val_report = validate_events_batch(events, min_datapoints=1, show_progress=False)
        assert val_report['valid'] == 2  # Events 1 and 3
        assert val_report['skipped'] == 1  # Event 2
        
        # Step 2: Grouping
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            show_progress=False
        )
        
        # Events 1 and 3 should be grouped (both valid, same user/type, < 3 min)
        assert len(unique_events) == 1
        assert group_info['total_input_events'] == 2
    
    def test_full_pipeline_with_config_invalid_events(self):
        """Test that config-marked invalid events are excluded early."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'time': 0}]},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 0}]},
            {'id': 3, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:02:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 0}]},
        ]
        
        # Mark event 2 as invalid in config
        valid_events, val_report = validate_events_batch(
            events,
            invalid_event_ids=[2],
            show_progress=False
        )
        
        # Event 2 should be skipped
        assert val_report['valid'] == 2
        assert val_report['skipped'] == 1
        assert 'marked_invalid_in_config' in val_report['skip_reasons']
        
        # Grouping should only see events 1 and 3
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            show_progress=False
        )
        
        # Events 1 and 3 are 2 minutes apart, should group
        assert len(unique_events) == 1
        assert 3 in group_info['discarded_events'] or 1 in group_info['discarded_events']
    
    def test_177_second_bug_fix_integration(self):
        """Integration test for the critical 177-second bug fix."""
        events = [
            {
                'id': 100,
                'userId': 42,
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'dataTime': '2022-06-15T14:30:00Z',
                'osdAlarmState': 2,
                'desc': 'First event',
                'datapoints': [{'time': i} for i in range(20)]
            },
            {
                'id': 101,
                'userId': 42,
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'dataTime': '2022-06-15T14:32:57Z',  # 177 seconds later
                'osdAlarmState': 1,
                'desc': 'Second event - 177s later',
                'datapoints': [{'time': i} for i in range(15)]
            }
        ]
        
        # Validate
        valid_events, val_report = validate_events_batch(events, show_progress=False)
        assert val_report['valid'] == 2
        
        # Group
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            selection_strategy='alarm_first',
            show_progress=False
        )
        
        # CRITICAL: Events 177s apart MUST be grouped
        assert len(unique_events) == 1, "Events 177s apart should be grouped!"
        assert group_info['total_groups'] == 1
        assert unique_events[0]['id'] == 100  # First (alarm) event selected
        assert 101 in group_info['discarded_events']
    
    def test_different_users_not_grouped(self):
        """Test that events from different users are not grouped."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 0}]},
            {'id': 2, 'userId': 200, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:30Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 0}]},
        ]
        
        valid_events, val_report = validate_events_batch(events, show_progress=False)
        assert val_report['valid'] == 2
        
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            show_progress=False
        )
        
        # Different users - should NOT be grouped
        assert len(unique_events) == 2
        assert group_info['total_groups'] == 2
        assert len(group_info['discarded_events']) == 0
    
    def test_edge_case_empty_input(self):
        """Test pipeline with empty input."""
        events = []
        
        valid_events, val_report = validate_events_batch(events, show_progress=False)
        assert val_report['valid'] == 0
        
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            show_progress=False
        )
        
        assert len(unique_events) == 0
        assert group_info['total_groups'] == 0


class TestRealWorldScenarios:
    """Test realistic scenarios that might occur in production."""
    
    def test_multiple_users_multiple_types(self):
        """Test complex scenario with multiple users and event types."""
        events = [
            # User 100 - Seizures
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'t': 0}]},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'t': 0}]},
            # User 100 - False Alarms
            {'id': 3, 'userId': 100, 'type': 'False Alarm', 'dataTime': '2022-01-01T12:00:30Z',
             'osdAlarmState': 0, 'desc': '', 'datapoints': [{'t': 0}]},
            # User 200 - Seizures
            {'id': 4, 'userId': 200, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'t': 0}]},
            {'id': 5, 'userId': 200, 'type': 'Seizure', 'dataTime': '2022-01-01T12:02:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'t': 0}]},
        ]
        
        valid_events, val_report = validate_events_batch(events, show_progress=False)
        assert val_report['valid'] == 5
        
        unique_events, group_info = apply_sliding_window_grouping(
            valid_events,
            time_threshold='3min',
            show_progress=False
        )
        
        # Expected groups:
        # 1. User 100 Seizures (ids 1, 2) -> 1 event
        # 2. User 100 False Alarms (id 3) -> 1 event
        # 3. User 200 Seizures (ids 4, 5) -> 1 event
        # Total: 3 unique events
        assert len(unique_events) == 3
        assert group_info['total_input_events'] == 5
        assert len(group_info['discarded_events']) == 2


# Allow running tests without pytest
if __name__ == '__main__':
    try:
        pytest.main([__file__, '-v'])
    except:
        print("Running manual integration tests...")
        
        # Test 177s bug fix
        print("\n1. Testing 177-second bug fix (integration)...")
        events = [
            {'id': 100, 'userId': 42, 'type': 'Seizure', 'dataTime': '2022-06-15T14:30:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'time': 0}]},
            {'id': 101, 'userId': 42, 'type': 'Seizure', 'dataTime': '2022-06-15T14:32:57Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 0}]},
        ]
        valid, val_rep = validate_events_batch(events, show_progress=False)
        unique, grp_inf = apply_sliding_window_grouping(valid, time_threshold='3min', show_progress=False)
        
        if len(unique) == 1 and 101 in grp_inf['discarded_events']:
            print("   ✓ 177s bug fix working in full pipeline!")
        else:
            print("   ✗ 177s bug fix FAILED in pipeline")
        
        # Test validation filtering
        print("\n2. Testing validation filters invalid events...")
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'time': 0}]},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': []},  # Invalid
        ]
        valid, val_rep = validate_events_batch(events, show_progress=False)
        unique, grp_inf = apply_sliding_window_grouping(valid, show_progress=False)
        
        if val_rep['valid'] == 1 and len(unique) == 1:
            print("   ✓ Invalid events filtered before grouping")
        else:
            print("   ✗ Validation filtering failed")
        
        print("\n✅ Integration tests completed")
