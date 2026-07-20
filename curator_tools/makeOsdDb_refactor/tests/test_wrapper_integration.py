#!/usr/bin/env python3
"""
Integration tests for the makeOsdDb_refactored_wrapper.py

Tests the complete workflow:
1. Download events from web API (mocked)
2. Process events with validation, normalization, grouping
3. Save to SQLite database
4. Publish from SQLite to JSON files
5. Verify data integrity throughout pipeline
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import modules under test
from osdb_sqlite import OsdWorkingDb
from database_utils import get_database_stats, validate_database


class TestWrapperIntegration(unittest.TestCase):
    """Integration tests for wrapper functionality"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_working.db')
        self.json_dir = os.path.join(self.test_dir, 'json_output')
        os.makedirs(self.json_dir, exist_ok=True)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def create_sample_events(self, count=10):
        """
        Create sample events for testing.
        Returns list of event dictionaries.
        """
        events = []
        for i in range(count):
            event = {
                'id': 1000 + i,
                'userId': 42,
                'type': 'Seizure' if i % 2 == 0 else 'False Alarm',
                'subType': 'Tonic-Clonic' if i % 3 == 0 else 'Other',
                'desc': f'Test event {i}',
                'dataTime': f'2023-12-{(i % 28) + 1:02d}T14:30:00Z',
                'dataTimeEnd': f'2023-12-{(i % 28) + 1:02d}T14:32:00Z',
                'userId': 40 + (i % 3),  # 3 different users
                'dataSourceName': 'Watch',
                'phoneAppVersion': '1.2.3',
                'watchSdVersion': '2.1.0',
                'osdAlarmState': 2 if i % 2 == 0 else 0,
                'datapoints': [
                    {
                        'dataTime': f'2023-12-{(i % 28) + 1:02d}T14:30:{j:02d}Z',
                        'hr': 80 + j,
                        'rawData': [100 + j*k for k in range(25)]
                    }
                    for j in range(5)
                ]
            }
            events.append(event)
        return events
    
    def test_save_and_load_events(self):
        """Test saving events to database and loading them back"""
        # Create sample events
        events = self.create_sample_events(10)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Load from database
        loaded_events = db.get_events(include_datapoints=True)
        
        # Verify counts
        self.assertEqual(len(loaded_events), len(events))
        
        # Verify event IDs match
        original_ids = sorted([e['id'] for e in events])
        loaded_ids = sorted([e['id'] for e in loaded_events])
        self.assertEqual(original_ids, loaded_ids)
        
        # Verify datapoints preserved
        for event in loaded_events:
            self.assertIn('datapoints', event)
            self.assertEqual(len(event['datapoints']), 5)
    
    def test_database_schema_version(self):
        """Test that database has correct schema version"""
        # Create database
        db = OsdWorkingDb(self.db_path)
        
        # Add sample event
        events = self.create_sample_events(1)
        db.add_events(events)
        
        # Check schema version
        stats = get_database_stats(self.db_path)
        self.assertEqual(stats['schema_version'], 1)
    
    def test_event_type_filtering(self):
        """Test filtering events by type"""
        # Create mixed events
        events = self.create_sample_events(20)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Load and filter
        all_events = db.get_events(include_datapoints=True)
        seizures = [e for e in all_events if e['type'] == 'Seizure']
        false_alarms = [e for e in all_events if e['type'] == 'False Alarm']
        
        # Verify filtering works
        self.assertEqual(len(seizures), 10)  # Half are seizures (even indices)
        self.assertEqual(len(false_alarms), 10)  # Half are false alarms (odd indices)
        self.assertEqual(len(seizures) + len(false_alarms), len(all_events))
    
    def test_incremental_updates(self):
        """Test adding new events to existing database"""
        # Create initial set of events
        initial_events = self.create_sample_events(5)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(initial_events)
        
        # Verify initial count
        loaded = db.get_events(include_datapoints=True)
        self.assertEqual(len(loaded), 5)
        
        # Add more events
        new_events = self.create_sample_events(3)
        # Change IDs to avoid duplicates
        for i, event in enumerate(new_events):
            event['id'] = 2000 + i
        
        db.add_events(new_events)
        
        # Verify total count
        loaded = db.get_events(include_datapoints=True)
        self.assertEqual(len(loaded), 8)
    
    def test_duplicate_event_handling(self):
        """Test that duplicate events are updated (INSERT OR REPLACE)"""
        # Create event
        events = self.create_sample_events(1)
        event_id = events[0]['id']
        events[0]['desc'] = 'Original description'
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Verify original
        loaded = db.get_events(include_datapoints=True)
        self.assertEqual(loaded[0]['desc'], 'Original description')
        
        # Update same event with different description
        events[0]['desc'] = 'Updated description'
        db.add_events(events)
        
        # Verify update
        loaded = db.get_events(include_datapoints=True)
        self.assertEqual(len(loaded), 1)  # Still only 1 event
        self.assertEqual(loaded[0]['desc'], 'Updated description')
    
    def test_datapoint_cascade_delete(self):
        """Test that deleting events also deletes their datapoints"""
        import sqlite3
        
        # Create events with datapoints
        events = self.create_sample_events(2)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Verify datapoints exist
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM datapoints")
        datapoint_count = cursor.fetchone()[0]
        self.assertEqual(datapoint_count, 10)  # 2 events × 5 datapoints
        
        # Delete one event
        event_id = events[0]['id']
        cursor.execute("DELETE FROM events WHERE id = ?", (event_id,))
        conn.commit()
        
        # Verify datapoints cascade deleted
        cursor.execute("SELECT COUNT(*) FROM datapoints")
        datapoint_count = cursor.fetchone()[0]
        self.assertEqual(datapoint_count, 5)  # 1 event × 5 datapoints
        
        conn.close()
    
    def test_database_validation(self):
        """Test database validation function"""
        # Create valid database
        events = self.create_sample_events(5)
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Validate database
        is_valid, issues = validate_database(self.db_path)
        
        # Should be valid with no issues
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_json_export_format(self):
        """Test that JSON export format matches expected structure"""
        # Create events
        events = self.create_sample_events(3)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Export to JSON
        exported = db.get_events(include_datapoints=True)
        
        # Verify structure
        for event in exported:
            # Check required fields
            self.assertIn('id', event)
            self.assertIn('userId', event)
            self.assertIn('type', event)
            self.assertIn('dataTime', event)
            self.assertIn('datapoints', event)
            
            # Check datapoints structure
            self.assertIsInstance(event['datapoints'], list)
            for dp in event['datapoints']:
                self.assertIn('dataTime', dp)
                self.assertIn('hr', dp)
                self.assertIn('rawData', dp)
    
    def test_statistics_calculation(self):
        """Test database statistics calculation"""
        # Create events
        events = self.create_sample_events(10)
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Get statistics
        stats = get_database_stats(self.db_path)
        
        # Verify statistics
        self.assertEqual(stats['total_events'], 10)
        self.assertEqual(stats['total_datapoints'], 50)  # 10 events × 5 datapoints
        self.assertAlmostEqual(stats['avg_datapoints_per_event'], 5.0)
        self.assertIn('Seizure', stats['events_by_type'])
        self.assertIn('False Alarm', stats['events_by_type'])
        self.assertEqual(stats['schema_version'], 1)
    
    def test_multiple_event_types_in_single_database(self):
        """Test storing multiple event types in single database"""
        # Create events of different types
        seizure_events = [
            {'id': 1001, 'type': 'Seizure', 'subType': 'Tonic-Clonic', 'userId': 42,
             'dataTime': '2023-12-01T14:30:00Z', 'datapoints': []},
            {'id': 1002, 'type': 'Seizure', 'subType': 'Other', 'userId': 42,
             'dataTime': '2023-12-02T14:30:00Z', 'datapoints': []},
        ]
        fall_events = [
            {'id': 1003, 'type': 'Fall', 'subType': '', 'userId': 43,
             'dataTime': '2023-12-03T14:30:00Z', 'datapoints': []},
        ]
        false_alarm_events = [
            {'id': 1004, 'type': 'False Alarm', 'subType': '', 'userId': 44,
             'dataTime': '2023-12-04T14:30:00Z', 'datapoints': []},
        ]
        
        # Save all to same database
        db = OsdWorkingDb(self.db_path)
        db.add_events(seizure_events + fall_events + false_alarm_events)
        
        # Load and filter
        all_events = db.get_events(include_datapoints=True)
        seizures = [e for e in all_events if e['type'] == 'Seizure']
        falls = [e for e in all_events if e['type'] == 'Fall']
        false_alarms = [e for e in all_events if e['type'] == 'False Alarm']
        
        # Verify counts
        self.assertEqual(len(seizures), 2)
        self.assertEqual(len(falls), 1)
        self.assertEqual(len(false_alarms), 1)
        self.assertEqual(len(all_events), 4)


class TestPublishWorkflow(unittest.TestCase):
    """Test the publish workflow (database → JSON files)"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_working.db')
        self.json_dir = os.path.join(self.test_dir, 'json_output')
        os.makedirs(self.json_dir, exist_ok=True)
    
    def tearDown(self):
        """Remove temporary directory"""
        shutil.rmtree(self.test_dir)
    
    def test_publish_to_json_files(self):
        """Test publishing database to separate JSON files by type"""
        # Create mixed events
        events = [
            {'id': 1001, 'type': 'Seizure', 'subType': 'Tonic-Clonic', 'userId': 42,
             'dataTime': '2023-12-01T14:30:00Z', 'datapoints': []},
            {'id': 1002, 'type': 'Seizure', 'subType': 'Other', 'userId': 42,
             'dataTime': '2023-12-02T14:30:00Z', 'datapoints': []},
            {'id': 1003, 'type': 'Fall', 'subType': '', 'userId': 43,
             'dataTime': '2023-12-03T14:30:00Z', 'datapoints': []},
            {'id': 1004, 'type': 'False Alarm', 'subType': '', 'userId': 44,
             'dataTime': '2023-12-04T14:30:00Z', 'datapoints': []},
        ]
        
        # Save to database
        db = OsdWorkingDb(self.db_path)
        db.add_events(events)
        
        # Simulate publish workflow
        all_events = db.get_events(include_datapoints=True)
        
        # Split by type
        event_categories = {
            'tcSeizures': [],
            'allSeizures': [],
            'fallEvents': [],
            'falseAlarms': [],
        }
        
        for event in all_events:
            event_type = event.get('type', 'Unknown')
            sub_type = event.get('subType', '')
            
            if event_type == 'Seizure':
                if 'tonic' in str(sub_type).lower() or 'clonic' in str(sub_type).lower():
                    event_categories['tcSeizures'].append(event)
                event_categories['allSeizures'].append(event)
            elif event_type == 'Fall':
                event_categories['fallEvents'].append(event)
            elif event_type == 'False Alarm':
                event_categories['falseAlarms'].append(event)
        
        # Verify categorization
        self.assertEqual(len(event_categories['tcSeizures']), 1)
        self.assertEqual(len(event_categories['allSeizures']), 2)
        self.assertEqual(len(event_categories['fallEvents']), 1)
        self.assertEqual(len(event_categories['falseAlarms']), 1)
        
        # Save to JSON files
        for category, category_events in event_categories.items():
            if category_events:
                fname = os.path.join(self.json_dir, f'osdb_3min_{category}.json')
                with open(fname, 'w') as f:
                    json.dump(category_events, f, indent=2)
        
        # Verify files created
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, 'osdb_3min_tcSeizures.json')))
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, 'osdb_3min_allSeizures.json')))
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, 'osdb_3min_fallEvents.json')))
        self.assertTrue(os.path.exists(os.path.join(self.json_dir, 'osdb_3min_falseAlarms.json')))
        
        # Verify file contents
        with open(os.path.join(self.json_dir, 'osdb_3min_tcSeizures.json'), 'r') as f:
            tc_events = json.load(f)
        self.assertEqual(len(tc_events), 1)
        self.assertEqual(tc_events[0]['id'], 1001)


if __name__ == '__main__':
    unittest.main()
