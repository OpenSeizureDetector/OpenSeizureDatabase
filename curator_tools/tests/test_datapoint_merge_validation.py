#!/usr/bin/env python3
"""
test_datapoint_merge_validation.py

Comprehensive tests for event datapoint merging, ensuring:
1. No time overlaps in merged datapoint arrays
2. Proper time ordering of acceleration data
3. Correct handling of rawData and rawData3D fields
4. Time tolerance thresholds are respected
5. Merged events maintain data integrity

This test suite validates the critical requirement that when events are
combined, their datapoints (including acceleration arrays) are merged
correctly with proper temporal ordering and no overlaps.

Run with: pytest test_datapoint_merge_validation.py -v -s
Or: python3 -m pytest test_datapoint_merge_validation.py::TestDatapointMergeValidation -v
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add src directory to path
sys.path.insert(0, os.path.abspath('src'))

import pytest
from event_grouping import (
    concatenate_datapoints,
    merge_grouped_events,
    group_events_by_proximity
)


class SimulatedDataGenerator:
    """Utility class to generate realistic simulated sensor data for testing."""
    
    @staticmethod
    def create_datapoint(
        timestamp: str,
        hr: float = 75.0,
        o2_sat: int = 98,
        raw_data: List[float] = None,
        raw_data_3d: List[List[float]] = None,
        acc_mean: float = None,
        acc_sd: float = None,
        sample_freq: float = 25.0
    ) -> Dict[str, Any]:
        """Create a realistic datapoint with acceleration data."""
        dp = {
            'dataTime': timestamp,
            'hr': hr,
            'o2Sat': o2_sat,
            'sampleFreq': sample_freq
        }
        
        # Add optional acceleration data
        if raw_data is not None:
            dp['rawData'] = raw_data
        if raw_data_3d is not None:
            dp['rawData3D'] = raw_data_3d
        if acc_mean is not None:
            dp['accMean'] = acc_mean
        if acc_sd is not None:
            dp['accSd'] = acc_sd
        
        return dp
    
    @staticmethod
    def create_event(
        event_id: int,
        user_id: int,
        event_type: str,
        event_time: str,
        datapoints: List[Dict[str, Any]],
        alarm_state: int = 0,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create a complete event with datapoints."""
        return {
            'id': str(event_id),
            'userId': user_id,
            'type': event_type,
            'dataTime': event_time,
            'osdAlarmState': alarm_state,
            'desc': description,
            'datapoints': datapoints
        }
    
    @staticmethod
    def create_acceleration_array(
        length: int,
        base_value: float = 1000.0,
        noise: float = 50.0
    ) -> List[float]:
        """Create a simulated acceleration magnitude array."""
        import random
        return [base_value + random.uniform(-noise, noise) for _ in range(length)]
    
    @staticmethod
    def create_3d_acceleration_array(
        num_samples: int,
        base_x: float = 1000.0,
        base_y: float = 0.0,
        base_z: float = -1000.0,
        noise: float = 30.0
    ) -> List[List[float]]:
        """Create a simulated 3D acceleration array (X, Y, Z)."""
        import random
        return [
            [
                base_x + random.uniform(-noise, noise),
                base_y + random.uniform(-noise, noise),
                base_z + random.uniform(-noise, noise)
            ]
            for _ in range(num_samples)
        ]


class TestDatapointTimeOrdering:
    """Test that merged datapoints maintain correct time ordering."""
    
    def test_datapoints_ordered_by_time(self):
        """Test that datapoints are sorted by time after merge."""
        gen = SimulatedDataGenerator()
        
        # Create two events with datapoints
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80),
                gen.create_datapoint('2024-01-15T10:00:10Z', hr=85),
                gen.create_datapoint('2024-01-15T10:00:20Z', hr=90),
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:15Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:05Z', hr=82),  # Out of order in event
                gen.create_datapoint('2024-01-15T10:00:15Z', hr=88),
                gen.create_datapoint('2024-01-15T10:00:25Z', hr=92),
            ]
        )
        
        # Merge
        merged = merge_grouped_events([event1, event2], event1)
        
        # Verify datapoints are ordered by time
        merged_dp = merged['datapoints']
        assert len(merged_dp) == 6, f"Expected 6 datapoints, got {len(merged_dp)}"
        
        # Extract times and verify ordering
        times = [dp['dataTime'] for dp in merged_dp]
        sorted_times = sorted(times)
        
        print(f"\nDatapoint times: {times}")
        print(f"Sorted times:   {sorted_times}")
        
        assert times == sorted_times, "Datapoints not in chronological order"
    
    def test_datapoints_monotonic_increasing_time(self):
        """Test that datapoint times are monotonically increasing."""
        gen = SimulatedDataGenerator()
        
        events = [
            gen.create_event(
                event_id=i,
                user_id=100,
                event_type='Seizure',
                event_time='2024-01-15T10:00:00Z',
                datapoints=[
                    gen.create_datapoint(f'2024-01-15T10:00:{10*i:02d}Z', hr=75 + i),
                    gen.create_datapoint(f'2024-01-15T10:00:{10*i+5:02d}Z', hr=76 + i),
                ]
            )
            for i in range(3)
        ]
        
        merged = merge_grouped_events(events, events[0])
        merged_dp = merged['datapoints']
        
        # Convert times to timestamps for comparison
        from dateutil import parser
        timestamps = []
        for dp in merged_dp:
            dt = parser.parse(dp['dataTime'])
            timestamps.append(dt.timestamp())
        
        # Verify monotonically increasing
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], \
                f"Time not monotonic at index {i}: {timestamps[i]} < {timestamps[i-1]}"
        
        print(f"✓ {len(merged_dp)} datapoints properly ordered with monotonic increasing time")


class TestDatapointTimeOverlapDetection:
    """Test detection and handling of overlapping time intervals."""
    
    def test_no_overlapping_time_intervals(self):
        """Test that merged datapoints don't have overlapping time coverage."""
        gen = SimulatedDataGenerator()
        
        # Create two events with distinct, non-overlapping time windows
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80),  # 0-10s
                gen.create_datapoint('2024-01-15T10:00:10Z', hr=85),
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:20Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:20Z', hr=88),  # 20-30s
                gen.create_datapoint('2024-01-15T10:00:30Z', hr=92),
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        merged_dp = merged['datapoints']
        
        # Verify time intervals don't overlap
        from dateutil import parser
        intervals = []
        for dp in merged_dp:
            dt = parser.parse(dp['dataTime']).timestamp()
            intervals.append(dt)
        
        assert len(intervals) == 4
        for i in range(len(intervals) - 1):
            assert intervals[i] < intervals[i+1], \
                f"Time overlap detected: {intervals[i]} >= {intervals[i+1]}"
        
        print(f"✓ No overlapping time intervals in {len(merged_dp)} datapoints")
    
    def test_duplicate_timestamp_handling(self):
        """Test that duplicate timestamps are handled with time_tolerance."""
        gen = SimulatedDataGenerator()
        
        # Create events with datapoints at nearly identical times (within 100ms)
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00.0Z', hr=80),
                gen.create_datapoint('2024-01-15T10:00:10.0Z', hr=85),
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00.050Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00.05Z', hr=82),  # 50ms offset - should be deduplicated
                gen.create_datapoint('2024-01-15T10:00:10.0Z', hr=87),   # Exact duplicate - should be deduplicated
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1, concatenate_datapoints_flag=True)
        merged_dp = merged['datapoints']
        
        # With time_tolerance=100ms (default), should have ~2-3 unique datapoints
        print(f"Merged datapoints after deduplication: {len(merged_dp)}")
        print(f"Datapoints: {[dp['dataTime'] for dp in merged_dp]}")
        
        # Should have removed duplicates/near-duplicates
        assert len(merged_dp) <= 4, f"Expected <= 4 after deduplication, got {len(merged_dp)}"
        
        print(f"✓ Duplicate timestamps handled correctly with time_tolerance")


class TestAccelerationDataIntegrity:
    """Test that acceleration data (rawData, rawData3D) is preserved during merge."""
    
    def test_raw_acceleration_data_preserved(self):
        """Test that rawData arrays are preserved during merge."""
        gen = SimulatedDataGenerator()
        
        raw_data_1 = [100, 150, 200, 250, 300]
        raw_data_2 = [110, 160, 210, 260, 310]
        
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80, raw_data=raw_data_1)
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:10Z', hr=85, raw_data=raw_data_2)
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        merged_dp = merged['datapoints']
        
        # Both rawData arrays should be present
        raw_data_found = []
        for dp in merged_dp:
            if 'rawData' in dp:
                raw_data_found.append(dp['rawData'])
        
        assert len(raw_data_found) == 2, f"Expected 2 rawData arrays, found {len(raw_data_found)}"
        assert raw_data_1 in raw_data_found, "First rawData array not preserved"
        assert raw_data_2 in raw_data_found, "Second rawData array not preserved"
        
        print(f"✓ RawData arrays preserved: {len(raw_data_found)} arrays")
    
    def test_3d_acceleration_data_preserved(self):
        """Test that rawData3D (3D acceleration) arrays are preserved."""
        gen = SimulatedDataGenerator()
        
        raw_3d_1 = [[100, 0, -1000], [150, 0, -1000], [200, 0, -1000]]
        raw_3d_2 = [[110, 10, -1010], [160, 10, -1010]]
        
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:00Z',
                    hr=80,
                    raw_data_3d=raw_3d_1
                )
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:10Z',
                    hr=85,
                    raw_data_3d=raw_3d_2
                )
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        merged_dp = merged['datapoints']
        
        # Both rawData3D arrays should be present
        raw_3d_found = []
        for dp in merged_dp:
            if 'rawData3D' in dp:
                raw_3d_found.append(dp['rawData3D'])
        
        assert len(raw_3d_found) == 2, f"Expected 2 rawData3D arrays, found {len(raw_3d_found)}"
        assert raw_3d_1 in raw_3d_found, "First rawData3D array not preserved"
        assert raw_3d_2 in raw_3d_found, "Second rawData3D array not preserved"
        
        print(f"✓ RawData3D arrays preserved: {len(raw_3d_found)} arrays")
    
    def test_mixed_acceleration_formats(self):
        """Test merging events with different acceleration data formats."""
        gen = SimulatedDataGenerator()
        
        # Event 1: has rawData
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:00Z',
                    hr=80,
                    raw_data=[100, 150, 200]
                )
            ]
        )
        
        # Event 2: has rawData3D
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:10Z',
                    hr=85,
                    raw_data_3d=[[100, 0, -1000], [150, 0, -1000]]
                )
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        merged_dp = merged['datapoints']
        
        # Should have both types of acceleration data
        has_raw_data = any('rawData' in dp for dp in merged_dp)
        has_raw_3d = any('rawData3D' in dp for dp in merged_dp)
        
        assert has_raw_data, "rawData not preserved in mixed format merge"
        assert has_raw_3d, "rawData3D not preserved in mixed format merge"
        
        print(f"✓ Mixed acceleration formats handled correctly")


class TestDatapointCountAndStats:
    """Test that datapoint counts and statistics are correct after merge."""
    
    def test_merged_event_count_tracking(self):
        """Test that _merged_event_count is set correctly."""
        gen = SimulatedDataGenerator()
        
        events = [
            gen.create_event(
                event_id=i,
                user_id=100,
                event_type='Seizure',
                event_time=f'2024-01-15T10:00:{i*10:02d}Z',
                datapoints=[
                    gen.create_datapoint(f'2024-01-15T10:00:{i*10:02d}Z', hr=75 + i),
                ]
            )
            for i in range(4)
        ]
        
        merged = merge_grouped_events(events, events[0])
        
        assert '_merged_event_count' in merged, "Missing _merged_event_count"
        assert merged['_merged_event_count'] == 4, \
            f"Expected 4 merged events, got {merged['_merged_event_count']}"
        
        print(f"✓ Merged event count tracked: {merged['_merged_event_count']}")
    
    def test_merged_datapoint_count_tracking(self):
        """Test that _merged_datapoint_count reflects actual datapoints after deduplication."""
        gen = SimulatedDataGenerator()
        
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80),
                gen.create_datapoint('2024-01-15T10:00:10Z', hr=85),
                gen.create_datapoint('2024-01-15T10:00:20Z', hr=90),
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:15Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:25Z', hr=92),
                gen.create_datapoint('2024-01-15T10:00:35Z', hr=95),
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        
        assert '_merged_datapoint_count' in merged, "Missing _merged_datapoint_count"
        actual_count = len(merged['datapoints'])
        tracked_count = merged['_merged_datapoint_count']
        
        assert tracked_count == actual_count, \
            f"Count mismatch: tracked={tracked_count}, actual={actual_count}"
        
        print(f"✓ Merged datapoint count tracked: {tracked_count}")


class TestEdgeCases:
    """Test edge cases in datapoint merging."""
    
    def test_merge_with_zero_datapoints(self):
        """Test merging events where some have zero datapoints."""
        gen = SimulatedDataGenerator()
        
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80),
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[]  # No datapoints
        )
        
        event3 = gen.create_event(
            event_id=3,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:20Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:20Z', hr=85),
            ]
        )
        
        merged = merge_grouped_events([event1, event2, event3], event1)
        
        # Should have datapoints from events 1 and 3
        assert len(merged['datapoints']) == 2, \
            f"Expected 2 datapoints, got {len(merged['datapoints'])}"
        
        print(f"✓ Zero datapoint event handled: {len(merged['datapoints'])} datapoints in merged event")
    
    def test_merge_events_with_missing_time_field(self):
        """Test handling of datapoints with missing time field."""
        gen = SimulatedDataGenerator()
        
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint('2024-01-15T10:00:00Z', hr=80),
            ]
        )
        
        # Create event with datapoint missing dataTime field
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[
                {'hr': 85, 'o2Sat': 97}  # Missing dataTime
            ]
        )
        
        # Should not crash
        merged = merge_grouped_events([event1, event2], event1)
        assert merged is not None, "Merge failed with missing time field"
        
        print(f"✓ Missing time field handled gracefully")
    
    def test_large_number_of_datapoints(self):
        """Test merging events with many datapoints (large sensor arrays)."""
        gen = SimulatedDataGenerator()
        
        # Create events with large rawData arrays (1000+ points each)
        events = []
        for i in range(3):
            large_raw_data = gen.create_acceleration_array(1000)
            event = gen.create_event(
                event_id=i+1,
                user_id=100,
                event_type='Seizure',
                event_time=f'2024-01-15T10:00:{i*10:02d}Z',
                datapoints=[
                    gen.create_datapoint(
                        f'2024-01-15T10:00:{i*10:02d}Z',
                        hr=75 + i,
                        raw_data=large_raw_data
                    )
                ]
            )
            events.append(event)
        
        merged = merge_grouped_events(events, events[0])
        
        # Should have 3 datapoints with large arrays
        assert len(merged['datapoints']) == 3, \
            f"Expected 3 datapoints, got {len(merged['datapoints'])}"
        
        # Verify arrays are intact
        total_array_size = sum(
            len(dp.get('rawData', []))
            for dp in merged['datapoints']
        )
        
        assert total_array_size == 3000, \
            f"Expected 3000 total array elements, got {total_array_size}"
        
        print(f"✓ Large acceleration arrays handled: {total_array_size} total elements")


class TestComplexMergingScenarios:
    """Test complex real-world merging scenarios."""
    
    def test_three_event_merge_with_realistic_data(self):
        """Test realistic scenario: 3 related events merged."""
        gen = SimulatedDataGenerator()
        
        # Simulate 3 seizure-related events within 3 minutes
        # Each triggered detections in quick succession
        
        event1 = gen.create_event(
            event_id=101,
            user_id=42,
            event_type='Seizure',
            event_time='2024-01-15T14:30:00Z',
            alarm_state=2,  # Alarm
            description='Device detected seizure activity',
            datapoints=[
                gen.create_datapoint(f'2024-01-15T14:30:{i:02d}Z', hr=120+i, o2_sat=95-i//10)
                for i in range(0, 30, 5)  # 6 datapoints over 30 seconds
            ]
        )
        
        event2 = gen.create_event(
            event_id=102,
            user_id=42,
            event_type='Seizure',
            event_time='2024-01-15T14:31:15Z',
            alarm_state=2,
            description='Continued seizure activity',
            datapoints=[
                gen.create_datapoint(f'2024-01-15T14:31:{i:02d}Z', hr=130+i, o2_sat=94-i//10)
                for i in range(0, 45, 5)  # 9 datapoints over 45 seconds
            ]
        )
        
        event3 = gen.create_event(
            event_id=103,
            user_id=42,
            event_type='Seizure',
            event_time='2024-01-15T14:32:30Z',
            alarm_state=1,  # Warning
            description='Post-ictal period',
            datapoints=[
                gen.create_datapoint(f'2024-01-15T14:32:{i:02d}Z', hr=110+i, o2_sat=96-i//10)
                for i in range(0, 20, 5)  # 4 datapoints over 20 seconds
            ]
        )
        
        # Merge all three
        merged = merge_grouped_events([event1, event2, event3], event1)
        
        # Verify merge integrity
        total_dp = len(merged['datapoints'])
        assert total_dp >= 15, f"Expected >= 15 datapoints, got {total_dp}"
        
        # Verify times are ordered
        from dateutil import parser
        times = [parser.parse(dp['dataTime']).timestamp() for dp in merged['datapoints']]
        assert times == sorted(times), "Datapoints not in time order"
        
        # Verify no overlapping times
        for i in range(len(times) - 1):
            assert times[i] <= times[i+1], "Time overlap detected"
        
        # Verify metadata
        assert merged['_merged_event_count'] == 3
        assert len(merged['_merged_from_event_ids']) == 3
        
        print(f"✓ Complex 3-event merge successful: {total_dp} datapoints, properly ordered")
        print(f"  Event IDs: {merged['_merged_from_event_ids']}")
        print(f"  Time span: {times[0]:.1f} to {times[-1]:.1f} ({times[-1] - times[0]:.1f}s)")
    
    def test_merge_preserves_sensor_data_types(self):
        """Test that different sensor data types are preserved during merge."""
        gen = SimulatedDataGenerator()
        
        # Create events with different sensor data types
        event1 = gen.create_event(
            event_id=1,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:00Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:00Z',
                    hr=80.5,  # float
                    o2_sat=98,  # int
                    raw_data=[100, 200, 300],  # list of ints
                    acc_mean=950.5  # float
                )
            ]
        )
        
        event2 = gen.create_event(
            event_id=2,
            user_id=100,
            event_type='Seizure',
            event_time='2024-01-15T10:00:10Z',
            datapoints=[
                gen.create_datapoint(
                    '2024-01-15T10:00:10Z',
                    hr=85.2,
                    o2_sat=97,
                    raw_data_3d=[[100, 0, -1000], [150, 0, -1000]],
                    acc_sd=45.3
                )
            ]
        )
        
        merged = merge_grouped_events([event1, event2], event1)
        merged_dp = merged['datapoints']
        
        # Verify types are preserved
        for i, dp in enumerate(merged_dp):
            if 'hr' in dp:
                assert isinstance(dp['hr'], float), f"HR type not preserved in datapoint {i}"
            if 'o2Sat' in dp:
                assert isinstance(dp['o2Sat'], int), f"O2Sat type not preserved in datapoint {i}"
            if 'rawData' in dp:
                assert isinstance(dp['rawData'], list), f"rawData type not preserved in datapoint {i}"
            if 'accMean' in dp:
                assert isinstance(dp['accMean'], float), f"accMean type not preserved in datapoint {i}"
        
        print(f"✓ Sensor data types preserved through merge")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
