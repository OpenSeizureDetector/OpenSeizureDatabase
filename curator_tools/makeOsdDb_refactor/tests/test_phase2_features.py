#!/usr/bin/env python3
"""
test_phase2_features.py

Unit tests for Phase 2 features:
- Datapoint concatenation
- Event deduplication

Run with: pytest test_phase2_features.py -v
Or: python3 test_phase2_features.py
"""

import sys
import os
import pytest

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_grouping import concatenate_datapoints, merge_grouped_events, apply_sliding_window_grouping
from event_deduplication import compute_event_hash, find_duplicate_events, remove_duplicate_events


class TestDatapointConcatenation:
    """Test datapoint concatenation functionality."""
    
    def test_concatenate_single_event(self):
        """Test concatenation with a single event."""
        events = [
            {
                'id': 1,
                'datapoints': [{'time': 0}, {'time': 100}, {'time': 200}]
            }
        ]
        
        result = concatenate_datapoints(events)
        assert len(result) == 3
        assert result[0]['time'] == 0
        assert result[2]['time'] == 200
    
    def test_concatenate_multiple_events(self):
        """Test concatenation with multiple events."""
        events = [
            {'id': 1, 'datapoints': [{'time': 0}, {'time': 100}]},
            {'id': 2, 'datapoints': [{'time': 200}, {'time': 300}]}
        ]
        
        result = concatenate_datapoints(events)
        assert len(result) == 4
        assert result[0]['time'] == 0
        assert result[3]['time'] == 300
    
    def test_concatenate_with_duplicates(self):
        """Test concatenation removes duplicate datapoints."""
        events = [
            {'id': 1, 'datapoints': [{'time': 0}, {'time': 100}]},
            {'id': 2, 'datapoints': [{'time': 100}, {'time': 200}]}  # 100 is duplicate
        ]
        
        result = concatenate_datapoints(events, remove_duplicates=True, time_tolerance_ms=50)
        assert len(result) == 3  # Should remove one duplicate
        times = [dp['time'] for dp in result]
        assert 0 in times
        assert 100 in times
        assert 200 in times
    
    def test_concatenate_empty_events(self):
        """Test concatenation with empty events list."""
        result = concatenate_datapoints([])
        assert len(result) == 0
    
    def test_concatenate_events_no_datapoints(self):
        """Test concatenation when events have no datapoints."""
        events = [
            {'id': 1, 'datapoints': []},
            {'id': 2, 'datapoints': []}
        ]
        
        result = concatenate_datapoints(events)
        assert len(result) == 0
    
    def test_concatenate_sorts_by_time(self):
        """Test that concatenation sorts datapoints by time."""
        events = [
            {'id': 1, 'datapoints': [{'time': 200}, {'time': 100}]},
            {'id': 2, 'datapoints': [{'time': 50}, {'time': 300}]}
        ]
        
        result = concatenate_datapoints(events, remove_duplicates=False)
        assert len(result) == 4
        assert result[0]['time'] == 50
        assert result[1]['time'] == 100
        assert result[2]['time'] == 200
        assert result[3]['time'] == 300


class TestMergeGroupedEvents:
    """Test merging of grouped events."""
    
    def test_merge_single_event(self):
        """Test merge with a single event (no merge needed)."""
        group = [
            {'id': 1, 'datapoints': [{'time': 0}], 'osdAlarmState': 2}
        ]
        
        result = merge_grouped_events(group, group[0], concatenate_datapoints_flag=True)
        assert result['id'] == 1
        assert len(result['datapoints']) == 1
        # Should not add merge metadata for single event
        assert '_merged_from_event_ids' not in result or len(result['_merged_from_event_ids']) == 1
    
    def test_merge_multiple_events(self):
        """Test merge with multiple events."""
        group = [
            {'id': 1, 'datapoints': [{'time': 0}, {'time': 100}], 'osdAlarmState': 2},
            {'id': 2, 'datapoints': [{'time': 200}, {'time': 300}], 'osdAlarmState': 1}
        ]
        
        result = merge_grouped_events(group, group[0], concatenate_datapoints_flag=True)
        assert result['id'] == 1
        assert len(result['datapoints']) == 4
        assert result['_merged_from_event_ids'] == [1, 2]
        assert result['_merged_event_count'] == 2
    
    def test_merge_disabled(self):
        """Test that merge can be disabled."""
        group = [
            {'id': 1, 'datapoints': [{'time': 0}], 'osdAlarmState': 2},
            {'id': 2, 'datapoints': [{'time': 100}], 'osdAlarmState': 1}
        ]
        
        result = merge_grouped_events(group, group[0], concatenate_datapoints_flag=False)
        assert result['id'] == 1
        assert len(result['datapoints']) == 1  # Only from first event


class TestEventDeduplication:
    """Test event deduplication functionality."""
    
    def test_compute_hash(self):
        """Test hash computation."""
        event = {
            'id': 1,
            'userId': 100,
            'dataTime': '2022-01-01T12:00:00Z',
            'type': 'Seizure'
        }
        
        hash1 = compute_event_hash(event)
        hash2 = compute_event_hash(event)
        assert hash1 == hash2  # Same event, same hash
    
    def test_different_events_different_hashes(self):
        """Test that different events produce different hashes."""
        event1 = {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'}
        event2 = {'id': 2, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'}
        
        hash1 = compute_event_hash(event1)
        hash2 = compute_event_hash(event2)
        assert hash1 != hash2
    
    def test_find_duplicates_by_id(self):
        """Test finding duplicates by ID."""
        events = [
            {'id': 1, 'userId': 100},
            {'id': 1, 'userId': 100},  # Duplicate
            {'id': 2, 'userId': 100}
        ]
        
        duplicates = find_duplicate_events(events, method='id')
        assert len(duplicates) == 1
        assert '1' in duplicates
        assert len(duplicates['1']) == 2
    
    def test_find_duplicates_by_hash(self):
        """Test finding duplicates by hash."""
        events = [
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'},
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'},
            {'id': 2, 'userId': 101, 'dataTime': '2022-01-01T13:00:00Z', 'type': 'Seizure'}
        ]
        
        duplicates = find_duplicate_events(events, method='hash')
        assert len(duplicates) == 1  # One group of duplicates
    
    def test_remove_duplicates_keep_first(self):
        """Test removing duplicates and keeping first."""
        events = [
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure', 'desc': 'first'},
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure', 'desc': 'second'},
            {'id': 2, 'userId': 101, 'dataTime': '2022-01-01T13:00:00Z', 'type': 'Seizure'}
        ]
        
        deduplicated, info = remove_duplicate_events(events, method='id', keep='first')
        assert len(deduplicated) == 2
        assert info['duplicates_removed'] == 1
        assert deduplicated[0]['desc'] == 'first'  # Kept first
    
    def test_remove_duplicates_keep_most_datapoints(self):
        """Test removing duplicates and keeping event with most datapoints."""
        events = [
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure',
             'datapoints': [{'time': 0}]},
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure',
             'datapoints': [{'time': 0}, {'time': 1}, {'time': 2}]},  # More datapoints
        ]
        
        deduplicated, info = remove_duplicate_events(events, method='id', keep='most_datapoints')
        assert len(deduplicated) == 1
        assert len(deduplicated[0]['datapoints']) == 3
    
    def test_no_duplicates(self):
        """Test with no duplicates."""
        events = [
            {'id': 1, 'userId': 100},
            {'id': 2, 'userId': 100},
            {'id': 3, 'userId': 100}
        ]
        
        deduplicated, info = remove_duplicate_events(events, method='id')
        assert len(deduplicated) == 3
        assert info['duplicates_removed'] == 0


class TestPhase2Integration:
    """Integration tests for Phase 2 features."""
    
    def test_grouping_with_concatenation(self):
        """Test that grouping + concatenation works end-to-end."""
        events = [
            {
                'id': 1,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:00:00Z',
                'osdAlarmState': 2,
                'desc': '',
                'datapoints': [{'time': 0}, {'time': 100}]
            },
            {
                'id': 2,
                'userId': 100,
                'type': 'Seizure',
                'dataTime': '2022-01-01T12:01:00Z',  # Within 3 min
                'osdAlarmState': 1,
                'desc': '',
                'datapoints': [{'time': 200}, {'time': 300}]
            }
        ]
        
        unique_events, group_info = apply_sliding_window_grouping(
            events,
            time_threshold='3min',
            concatenate_datapoints_flag=True,
            show_progress=False
        )
        
        assert len(unique_events) == 1
        assert group_info['concatenate_datapoints'] is True
        assert len(unique_events[0]['datapoints']) == 4  # All datapoints merged
        assert unique_events[0]['_merged_event_count'] == 2
        assert group_info['total_datapoints_before'] == 4
        assert group_info['total_datapoints_after'] == 4
    
    def test_grouping_without_concatenation(self):
        """Test that grouping without concatenation preserves original datapoints."""
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'time': 0}, {'time': 100}]},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 200}, {'time': 300}]}
        ]
        
        unique_events, group_info = apply_sliding_window_grouping(
            events,
            time_threshold='3min',
            concatenate_datapoints_flag=False,
            show_progress=False
        )
        
        assert len(unique_events) == 1
        assert group_info['concatenate_datapoints'] is False
        assert len(unique_events[0]['datapoints']) == 2  # Only from first event


# Allow running tests without pytest
if __name__ == '__main__':
    try:
        pytest.main([__file__, '-v'])
    except:
        print("Running manual Phase 2 tests...")
        
        # Test concatenation
        print("\n1. Testing datapoint concatenation...")
        events = [
            {'id': 1, 'datapoints': [{'time': 0}, {'time': 100}]},
            {'id': 2, 'datapoints': [{'time': 200}, {'time': 300}]}
        ]
        result = concatenate_datapoints(events)
        if len(result) == 4:
            print("   ✓ Concatenation working!")
        else:
            print(f"   ✗ Expected 4 datapoints, got {len(result)}")
        
        # Test deduplication
        print("\n2. Testing deduplication...")
        events = [
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'},
            {'id': 1, 'userId': 100, 'dataTime': '2022-01-01T12:00:00Z', 'type': 'Seizure'},
            {'id': 2, 'userId': 101, 'dataTime': '2022-01-01T13:00:00Z', 'type': 'Seizure'}
        ]
        deduplicated, info = remove_duplicate_events(events, method='id')
        if len(deduplicated) == 2 and info['duplicates_removed'] == 1:
            print("   ✓ Deduplication working!")
        else:
            print(f"   ✗ Deduplication failed")
        
        # Test integration
        print("\n3. Testing grouping with concatenation...")
        events = [
            {'id': 1, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:00:00Z',
             'osdAlarmState': 2, 'desc': '', 'datapoints': [{'time': 0}, {'time': 100}]},
            {'id': 2, 'userId': 100, 'type': 'Seizure', 'dataTime': '2022-01-01T12:01:00Z',
             'osdAlarmState': 1, 'desc': '', 'datapoints': [{'time': 200}, {'time': 300}]}
        ]
        unique_events, group_info = apply_sliding_window_grouping(
            events, concatenate_datapoints_flag=True, show_progress=False
        )
        if len(unique_events) == 1 and len(unique_events[0]['datapoints']) == 4:
            print("   ✓ Integration working!")
        else:
            print(f"   ✗ Integration failed")
        
        print("\n✅ Phase 2 tests completed")
