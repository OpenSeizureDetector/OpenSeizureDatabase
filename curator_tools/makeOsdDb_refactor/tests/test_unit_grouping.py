#!/usr/bin/env python3
"""
test_unit_grouping.py

Unit tests for event_grouping.py module.

Run with: pytest test_unit_grouping.py -v
Or: python3 test_unit_grouping.py
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_grouping import (
    parse_time_delta,
    group_events_by_proximity,
    select_best_event_from_group,
    apply_sliding_window_grouping
)


class TestParseTimeDelta:
    """Test parse_time_delta function."""
    
    def test_parse_seconds(self):
        """Test parsing seconds."""
        delta = parse_time_delta('180s')
        assert delta.total_seconds() == 180
    
    def test_parse_minutes(self):
        """Test parsing minutes."""
        delta = parse_time_delta('3min')
        assert delta.total_seconds() == 180
    
    def test_parse_hours(self):
        """Test parsing hours."""
        delta = parse_time_delta('2h')
        assert delta.total_seconds() == 7200
    
    def test_parse_default_minutes(self):
        """Test parsing plain number defaults to minutes."""
        delta = parse_time_delta('5')
        assert delta.total_seconds() == 300


class TestGroupEventsByProximity:
    """Test group_events_by_proximity function."""
    
    def test_single_event(self):
        """Test grouping a single event."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'datapoints': [{'time': 0}]
            }
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 1
        assert len(groups[0]) == 1
    
    def test_events_within_threshold(self):
        """Test events within threshold are grouped."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'datapoints': []
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:02:57Z',  # 177s later - should group
                'datapoints': []
            }
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 1  # Should be 1 group
        assert len(groups[0]) == 2  # With 2 events
        assert groups[0][0]['id'] == 1
        assert groups[0][1]['id'] == 2
    
    def test_events_beyond_threshold(self):
        """Test events beyond threshold are not grouped."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'datapoints': []
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:03:05Z',  # 185s later - should NOT group
                'datapoints': []
            }
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 2  # Should be 2 separate groups
    
    def test_different_users_not_grouped(self):
        """Test events from different users are not grouped."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'datapoints': []
            },
            {
                'id': 2,
                'userId': 101,  # Different user
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:30Z',  # 30s later
                'datapoints': []
            }
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 2  # Should be 2 separate groups (different users)
    
    def test_different_types_not_grouped(self):
        """Test events of different types are not grouped."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'datapoints': []
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'False Alarm',  # Different type
                'dataTime': '2022-01-01T12:00:30Z',
                'datapoints': []
            }
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 2  # Should be 2 separate groups (different types)
    
    def test_multiple_groups(self):
        """Test forming multiple groups."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z', 'datapoints': []},  # Group 1
            {'id': 3, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:10:00Z', 'datapoints': []},  # Group 2
            {'id': 4, 'userId': 101, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},  # Group 3
        ]
        groups = group_events_by_proximity(events, time_threshold='3min')
        assert len(groups) == 3


class TestSelectBestEventFromGroup:
    """Test select_best_event_from_group function."""
    
    def test_single_event_group(self):
        """Test selecting from single event group."""
        group = [
            {'id': 1, 'osdAlarmState': 1, 'desc': 'Test', 'datapoints': []}
        ]
        selected = select_best_event_from_group(group)
        assert selected['id'] == 1
    
    def test_alarm_first_strategy(self):
        """Test alarm_first selection strategy."""
        group = [
            {'id': 1, 'osdAlarmState': 1, 'desc': '', 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},
            {'id': 2, 'osdAlarmState': 2, 'desc': '', 'dataTime': '2022-01-01T12:01:00Z', 'datapoints': []},  # ALARM
            {'id': 3, 'osdAlarmState': 0, 'desc': '', 'dataTime': '2022-01-01T12:02:00Z', 'datapoints': []},
        ]
        selected = select_best_event_from_group(group, strategy='alarm_first')
        assert selected['id'] == 2  # Should select alarm event
    
    def test_tagged_preference(self):
        """Test preference for tagged events when no alarm."""
        group = [
            {'id': 1, 'osdAlarmState': 1, 'desc': '', 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},
            {'id': 2, 'osdAlarmState': 1, 'desc': 'Real seizure', 'dataTime': '2022-01-01T12:01:00Z', 'datapoints': []},  # Tagged
            {'id': 3, 'osdAlarmState': 1, 'desc': '', 'dataTime': '2022-01-01T12:02:00Z', 'datapoints': []},
        ]
        selected = select_best_event_from_group(group, strategy='alarm_first')
        assert selected['id'] == 2  # Should select tagged event
    
    def test_most_datapoints_strategy(self):
        """Test most_datapoints selection strategy."""
        group = [
            {'id': 1, 'datapoints': [{'time': 0}]},
            {'id': 2, 'datapoints': [{'time': 0}, {'time': 1}, {'time': 2}]},  # Most
            {'id': 3, 'datapoints': [{'time': 0}, {'time': 1}]},
        ]
        selected = select_best_event_from_group(group, strategy='most_datapoints')
        assert selected['id'] == 2
    
    def test_first_strategy(self):
        """Test first selection strategy."""
        group = [
            {'id': 1, 'dataTime': '2022-01-01T12:02:00Z', 'datapoints': []},
            {'id': 2, 'dataTime': '2022-01-01T12:01:00Z', 'datapoints': []},
            {'id': 3, 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},  # Earliest
        ]
        selected = select_best_event_from_group(group, strategy='first')
        assert selected['id'] == 3  # Should select earliest
    
    def test_last_strategy(self):
        """Test last selection strategy."""
        group = [
            {'id': 1, 'dataTime': '2022-01-01T12:00:00Z', 'datapoints': []},
            {'id': 2, 'dataTime': '2022-01-01T12:01:00Z', 'datapoints': []},
            {'id': 3, 'dataTime': '2022-01-01T12:02:00Z', 'datapoints': []},  # Latest
        ]
        selected = select_best_event_from_group(group, strategy='last')
        assert selected['id'] == 3


class TestApplySlidingWindowGrouping:
    """Test apply_sliding_window_grouping function."""
    
    def test_empty_events(self):
        """Test grouping empty event list."""
        unique, info = apply_sliding_window_grouping([], show_progress=False)
        assert len(unique) == 0
        assert info['total_groups'] == 0
        assert info['total_input_events'] == 0
    
    def test_no_grouping_needed(self):
        """Test when no events need grouping."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},
            {'id': 2, 'userId': 101, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},
        ]
        unique, info = apply_sliding_window_grouping(events, show_progress=False)
        assert len(unique) == 2
        assert info['total_groups'] == 2
        assert len(info['discarded_events']) == 0
    
    def test_grouping_with_discard(self):
        """Test grouping that discards events."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 2, 'desc': '', 'datapoints': []},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},
        ]
        unique, info = apply_sliding_window_grouping(events, time_threshold='3min', show_progress=False)
        assert len(unique) == 1
        assert info['total_groups'] == 1
        assert info['total_input_events'] == 2
        assert info['total_output_events'] == 1
        assert 2 in info['discarded_events']
    
    def test_177_second_bug_fix(self):
        """Test that events 177s apart ARE grouped (the bug fix!)."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 2, 'desc': '', 'datapoints': []},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:02:57Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},  # 177s
        ]
        unique, info = apply_sliding_window_grouping(events, time_threshold='3min', show_progress=False)
        
        # Events should be grouped (177s < 180s)
        assert len(unique) == 1, "Events 177s apart should be grouped!"
        assert 2 in info['discarded_events'], "Event 2 should be discarded"
    
    def test_grouping_info_structure(self):
        """Test that grouping info has correct structure."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},
        ]
        unique, info = apply_sliding_window_grouping(events, show_progress=False)
        
        # Check all required fields present
        assert 'total_groups' in info
        assert 'total_input_events' in info
        assert 'total_output_events' in info
        assert 'discarded_events' in info
        assert 'events_per_group' in info
        assert 'time_threshold' in info
        assert 'selection_strategy' in info


# Allow running tests without pytest
if __name__ == '__main__':
    try:
        pytest.main([__file__, '-v'])
    except:
        print("Running manual tests (pytest not available)...")
        
        # Test time delta parsing
        print("\n1. Testing time delta parsing...")
        delta = parse_time_delta('3min')
        print(f"   ✓ 3min = {delta.total_seconds()}s")
        
        # Test 177s grouping (the bug fix!)
        print("\n2. Testing 177s grouping (critical bug fix)...")
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z', 'osdAlarmState': 2, 'desc': '', 'datapoints': []},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:02:57Z', 'osdAlarmState': 1, 'desc': '', 'datapoints': []},
        ]
        unique, info = apply_sliding_window_grouping(events, time_threshold='3min', show_progress=False)
        if len(unique) == 1 and 2 in info['discarded_events']:
            print("   ✓ Events 177s apart correctly grouped!")
        else:
            print("   ✗ Events 177s apart NOT grouped (BUG!)")
        
        # Test selection strategies
        print("\n3. Testing selection strategies...")
        group = [
            {'id': 1, 'osdAlarmState': 1, 'desc': '', 'datapoints': [{'t': 0}]},
            {'id': 2, 'osdAlarmState': 2, 'desc': '', 'datapoints': []},
        ]
        selected = select_best_event_from_group(group, 'alarm_first')
        print(f"   ✓ alarm_first strategy selected event {selected['id']}")
        
        selected = select_best_event_from_group(group, 'most_datapoints')
        print(f"   ✓ most_datapoints strategy selected event {selected['id']}")
        
        print("\n✅ All manual tests completed")
