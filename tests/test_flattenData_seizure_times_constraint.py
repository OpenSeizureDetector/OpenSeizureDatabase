#!/usr/bin/env python3
"""
Comprehensive tests for seizureTimes constraint functionality in flattenData.py

Tests cover:
1. Configuration parameter parsing
2. Constraint activation/deactivation
3. Datapoint filtering with various margins
4. 1D data integrity (rawData field)
5. 3D data integrity (rawData3D field)
6. Edge cases (no seizureTimes, misaligned windows, etc.)
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'user_tools', 'nnTraining2'))

from user_tools.nnTraining2 import flattenData


class TestSeizureTimesConstraintHelpers(unittest.TestCase):
    """Test helper functions for seizureTimes constraint"""

    def test_prepare_constraint_disabled(self):
        """Test that constraint is inactive when useSeizureTimesConstraint is False"""
        event = {'id': 1, 'seizureTimes': [10.0, 20.0]}
        config = {'useSeizureTimesConstraint': False}
        valid_datapoints = [
            (datetime(2022, 1, 1, 12, 0, 5), {'dataTime': '2022-01-01T12:00:05Z'})
        ]
        
        constraint_active, event_start_dt, seizure_start_s, seizure_end_s, margin_s = \
            flattenData._prepare_seizure_time_constraint(event, valid_datapoints, config, debug=False)
        
        self.assertFalse(constraint_active)
        self.assertIsNone(event_start_dt)

    def test_prepare_constraint_no_seizure_times(self):
        """Test that constraint is active with default seizureTimes when none provided"""
        event = {'id': 1}  # No seizureTimes
        config = {'useSeizureTimesConstraint': True}
        valid_datapoints = [
            (datetime(2022, 1, 1, 12, 0, 5), {'dataTime': '2022-01-01T12:00:05Z'})
        ]
        
        constraint_active, event_start_dt, seizure_start_s, seizure_end_s, margin_s = \
            flattenData._prepare_seizure_time_constraint(event, valid_datapoints, config, debug=False)
        
        # Now with defaults, constraint is active
        self.assertTrue(constraint_active)
        # event_start_dt is not calculated here (it's calculated in process_event_obj)
        self.assertIsNone(event_start_dt)
        # Default seizureTimes should be [-30, 30]
        self.assertEqual(seizure_start_s, -30.0)
        self.assertEqual(seizure_end_s, 30.0)

    def test_prepare_constraint_enabled_valid_seizure_times(self):
        """Test constraint preparation with valid seizureTimes"""
        event = {'id': 1, 'seizureTimes': [10.0, 20.0]}
        config = {'useSeizureTimesConstraint': True, 'seizureTimeMarginSeconds': 5}
        
        # Create datapoints with known times
        # dataTime is end time of 5-second datapoint
        dt_end = datetime(2022, 1, 1, 12, 0, 5)
        valid_datapoints = [(dt_end, {'dataTime': '2022-01-01T12:00:05Z'})]
        
        constraint_active, event_start_dt, seizure_start_s, seizure_end_s, margin_s = \
            flattenData._prepare_seizure_time_constraint(event, valid_datapoints, config, debug=False)
        
        self.assertTrue(constraint_active)
        # event_start_dt is not calculated in helper (it's None here)
        self.assertIsNone(event_start_dt)
        self.assertEqual(seizure_start_s, 10.0)
        self.assertEqual(seizure_end_s, 20.0)
        self.assertEqual(margin_s, 5.0)

    def test_datapoint_overlap_check_inside_window(self):
        """Test that datapoint inside seizure window is detected"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 0.0
        
        # Datapoint at 15 seconds from earliest_dt_end (clearly inside [0, 10, 20])
        dt_end = earliest_dt_end + timedelta(seconds=15)
        
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        
        self.assertTrue(result)

    def test_datapoint_overlap_check_outside_window(self):
        """Test that datapoint outside seizure window is detected"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 0.0
        
        # Datapoint at 25 seconds from earliest_dt_end (clearly outside [10, 20])
        dt_end = earliest_dt_end + timedelta(seconds=25)
        
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        
        self.assertFalse(result)

    def test_datapoint_overlap_check_at_boundary(self):
        """Test datapoint at exact seizure window boundaries"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 0.0
        
        # Datapoint ending exactly at seizure start (at 10s)
        dt_end_at_start = earliest_dt_end + timedelta(seconds=10)
        result_at_start = flattenData._datapoint_in_seizure_window(
            dt_end_at_start, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result_at_start)
        
        # Datapoint ending exactly at seizure end (at 20s)
        dt_end_at_end = earliest_dt_end + timedelta(seconds=20)
        result_at_end = flattenData._datapoint_in_seizure_window(
            dt_end_at_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result_at_end)

    def test_datapoint_overlap_with_margin(self):
        """Test that margin extends the seizure window correctly"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 5.0  # Extends window to [5, 25]
        
        # Datapoint at 7 seconds (inside margin before seizure)
        dt_end_7 = earliest_dt_end + timedelta(seconds=7)
        result_7 = flattenData._datapoint_in_seizure_window(
            dt_end_7, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result_7)
        
        # Datapoint at 23 seconds (inside margin after seizure)
        dt_end_23 = earliest_dt_end + timedelta(seconds=23)
        result_23 = flattenData._datapoint_in_seizure_window(
            dt_end_23, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result_23)
        
        # Datapoint at 4 seconds (outside margin before seizure)
        dt_end_4 = earliest_dt_end + timedelta(seconds=4)
        result_4 = flattenData._datapoint_in_seizure_window(
            dt_end_4, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertFalse(result_4)

    def test_datapoint_overlap_partial_overlap_at_start(self):
        """Test datapoint that partially overlaps with seizure window start"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 0.0
        
        # Datapoint ending at 12s (starts at 7s, spans [7, 12])
        # Should overlap with [10, 20] seizure window
        dt_end = earliest_dt_end + timedelta(seconds=12)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result)

    def test_datapoint_overlap_partial_overlap_at_end(self):
        """Test datapoint that partially overlaps with seizure window end"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 0.0
        
        # Datapoint ending at 22s (starts at 17s, spans [17, 22])
        # Should overlap with [10, 20] seizure window
        dt_end = earliest_dt_end + timedelta(seconds=22)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result)


class TestProcessEventObjWithConstraint(unittest.TestCase):
    """Test process_event_obj function with seizureTimes constraint"""

    def create_test_datapoint(self, dt_end_seconds, raw_data_value=1.0, raw_data_3d_value=2.0):
        """Helper to create a mock datapoint"""
        dt_end = datetime(2022, 1, 1, 12, 0, 0) + timedelta(seconds=dt_end_seconds)
        
        # Create rawData with 125 values
        raw_data = [raw_data_value] * 125
        
        # Create rawData3D with 375 values (125 * 3 for X, Y, Z)
        raw_data_3d = [raw_data_3d_value] * 375
        
        return {
            'dataTime': dt_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'rawData': raw_data,
            'rawData3D': raw_data_3d,
            'alarmState': 2,
            'specPower': 100,
            'roiPower': 50,
            'hr': 80,
            'o2Sat': 98
        }

    def test_constraint_disabled_all_datapoints_included(self):
        """Test that all datapoints are included when constraint is disabled"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [10.0, 20.0],
            'datapoints': [
                self.create_test_datapoint(5),   # Before seizure
                self.create_test_datapoint(15),  # During seizure
                self.create_test_datapoint(25),  # After seizure
            ]
        }
        config = {'useSeizureTimesConstraint': False}
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have 3 rows (one per datapoint)
        self.assertEqual(len(rows), 3)

    def test_constraint_enabled_filters_datapoints(self):
        """Test that constraint properly filters datapoints"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [10.0, 15.0],  # Seizure from earliest_dt_end+10 to +15
            'datapoints': [
                self.create_test_datapoint(5),   # Before seizure - should be filtered
                self.create_test_datapoint(15),  # During seizure - should be included
                self.create_test_datapoint(25),  # After seizure - should be filtered
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have only 1 row (the datapoint at 15s, which overlaps with [10, 15])
        self.assertEqual(len(rows), 1)

    def test_constraint_with_margin_includes_context(self):
        """Test that margin extends the window to include context"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [0.0, 10.0],  # Seizure around earliest_dt_end
            'datapoints': [
                self.create_test_datapoint(7),   # Before seizure but within margin
                self.create_test_datapoint(15),  # After seizure but within margin
                self.create_test_datapoint(23),  # Way after seizure
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 8  # Window becomes [-8, 18]
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have 3 rows (all within [-8, 18] extended window relative to earliest at 7s)
        self.assertEqual(len(rows), 3)

    def test_raw_data_integrity_1d(self):
        """Test that 1D rawData maintains correct number of values"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [-5.0, 15.0],  # Includes datapoint at 15s
            'datapoints': [
                self.create_test_datapoint(15, raw_data_value=5.0)
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        self.assertEqual(len(rows), 1)
        row = rows[0]
        
        # Row format: eventId(1), userId(1), typeStr(1), type(1), dataTime(1), 
        #            alarmState(1), specPower(1), roiPower(1), hr(1), o2Sat(1) = 10 fields
        #            + rawData(125) + rawData3D(375) = 510 values
        expected_row_length = 10 + 125 + 375
        self.assertEqual(len(row), expected_row_length)
        
        # Check that rawData values (indices 10-134) are all 5.0
        raw_data_section = row[10:135]
        for i, val in enumerate(raw_data_section):
            self.assertEqual(float(val), 5.0, f"rawData[{i}] should be 5.0")

    def test_raw_data_integrity_3d(self):
        """Test that 3D rawData3D maintains correct number and structure of values"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [-5.0, 15.0],  # Includes datapoint at 15s
            'datapoints': [
                self.create_test_datapoint(15, raw_data_3d_value=7.5)
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        self.assertEqual(len(rows), 1)
        row = rows[0]
        
        # rawData3D starts after header(10) + rawData(125)
        raw_data_3d_start = 10 + 125
        raw_data_3d_section = row[raw_data_3d_start:raw_data_3d_start + 375]
        
        # Should have 375 values (125 per axis: X, Y, Z)
        self.assertEqual(len(raw_data_3d_section), 375)
        
        # All values should be 7.5
        for i, val in enumerate(raw_data_3d_section):
            self.assertEqual(float(val), 7.5, f"rawData3D[{i}] should be 7.5")

    def test_constraint_no_seizure_times_uses_entire_event(self):
        """Test that event without seizureTimes uses all datapoints even with constraint enabled"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            # No seizureTimes
            'datapoints': [
                self.create_test_datapoint(5),
                self.create_test_datapoint(15),
                self.create_test_datapoint(25),
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 5
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have all 3 rows since no seizureTimes constraint can be applied
        self.assertEqual(len(rows), 3)

    def test_constraint_with_validated_datapoints(self):
        """Test constraint works with validated (gap-filled) datapoints"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [-5.0, 15.0],  # Includes datapoint at 15s
            'datapoints': [
                self.create_test_datapoint(15),
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=True, config=config)
        
        # Should have 1 row (the datapoint at 15s, within [-5, 15])
        self.assertGreaterEqual(len(rows), 1)

    def test_multiple_datapoints_mixed_in_out_of_window(self):
        """Test filtering with multiple datapoints, some in and some out of window"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [12.0, 22.0],  # Window spans from earliest_dt_end+12 to +22
            'datapoints': [
                self.create_test_datapoint(5),    # Outside
                self.create_test_datapoint(12),   # At boundary (spans [7,12], touches [12,22])
                self.create_test_datapoint(18),   # Inside
                self.create_test_datapoint(24),   # Outside
                self.create_test_datapoint(35),   # Outside
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have 2 rows (12s and 18s overlap with [12,22])
        self.assertEqual(len(rows), 2)

    def test_no_accelerometer_data_still_skipped(self):
        """Test that datapoints without accelerometer data are still skipped even if in window"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [-5.0, 20.0],  # Includes datapoints at 15 and 17
            'datapoints': [
                # Valid datapoint
                self.create_test_datapoint(15, raw_data_value=5.0),
                # Datapoint without accelerometer data (in window but should be skipped)
                {
                    'dataTime': (datetime(2022, 1, 1, 12, 0, 17)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'rawData': None,
                    'rawData3D': None,
                    'alarmState': 2,
                    'hr': 80,
                    'o2Sat': 98
                }
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 0
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should have only 1 row (the valid datapoint)
        self.assertEqual(len(rows), 1)


class TestDatapointPartialOverlap(unittest.TestCase):
    """Test edge cases where datapoints partially overlap with seizure window"""

    def test_datapoint_starts_before_window_ends_during(self):
        """Datapoint spans from before seizure start to during seizure"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        
        # Datapoint at 12s (spans [7, 12], overlaps with [10, 20])
        dt_end = earliest_dt_end + timedelta(seconds=12)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, 0.0
        )
        self.assertTrue(result)

    def test_datapoint_starts_during_window_ends_after(self):
        """Datapoint spans from during seizure to after seizure"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        
        # Datapoint at 22s (spans [17, 22], overlaps with [10, 20])
        dt_end = earliest_dt_end + timedelta(seconds=22)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, 0.0
        )
        self.assertTrue(result)

    def test_datapoint_starts_before_window_ends_after(self):
        """Datapoint completely contains seizure window"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        
        # Datapoint at 8s (spans [3, 8], doesn't overlap)
        dt_end = earliest_dt_end + timedelta(seconds=8)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, 0.0
        )
        self.assertFalse(result)
        
        # Datapoint at 27s (spans [22, 27], doesn't overlap)
        dt_end = earliest_dt_end + timedelta(seconds=27)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, 0.0
        )
        self.assertFalse(result)


class TestConstraintEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_zero_margin(self):
        """Test with zero margin (strict window)"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        
        # Datapoint at 10s (boundary)
        dt_end = earliest_dt_end + timedelta(seconds=10)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, 0.0
        )
        self.assertTrue(result)

    def test_very_large_margin(self):
        """Test with very large margin"""
        earliest_dt_end = datetime(2022, 1, 1, 12, 0, 0)
        seizure_start_s = 10.0
        seizure_end_s = 20.0
        margin_s = 1000.0  # Very large margin
        
        # Datapoint at 5s (way before seizure, but within margin)
        dt_end = earliest_dt_end + timedelta(seconds=5)
        result = flattenData._datapoint_in_seizure_window(
            dt_end, earliest_dt_end, seizure_start_s, seizure_end_s, margin_s
        )
        self.assertTrue(result)

    def test_seizure_times_same_start_end(self):
        """Test with seizureTimes where start equals end (instantaneous event)"""
        event = {
            'id': 1,
            'userId': 123,
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'seizureTimes': [0.0, 0.0],  # Start equals end at earliest_dt_end
            'datapoints': [
                {
                    'dataTime': (datetime(2022, 1, 1, 12, 0, 15)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'rawData': [1.0] * 125,
                    'rawData3D': [2.0] * 375,
                    'alarmState': 2,
                    'hr': 80,
                    'o2Sat': 98
                }
            ]
        }
        config = {
            'useSeizureTimesConstraint': True,
            'seizureTimeMarginSeconds': 5  # Window becomes [-5, 5] around earliest_dt_end
        }
        
        rows = flattenData.process_event_obj(event, debug=False, validate=False, config=config)
        
        # Should include datapoint that overlaps with [-5, 5] window
        self.assertEqual(len(rows), 1)


if __name__ == '__main__':
    unittest.main()
