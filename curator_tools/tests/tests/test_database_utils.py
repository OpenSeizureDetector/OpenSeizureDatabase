#!/usr/bin/env python3
"""
test_database_utils.py - Tests for Database Utility Functions

Tests backup, deletion, validation, and other database utility operations.
"""

import unittest
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from database_utils import (
    backup_database, safe_delete_events, update_event_metadata,
    validate_database, get_database_stats, list_backups, get_schema_version
)
from osdb_sqlite import OsdWorkingDb


class TestDatabaseBackup(unittest.TestCase):
    """Test database backup functionality."""
    
    def setUp(self):
        """Create temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        # Create test database with some data
        self.db = OsdWorkingDb(self.db_path)
        test_events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2023-01-01T10:00:00Z',
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-01T10:00:00Z', 'alarmState': 2, 'hr': 120}
                ]
            }
        ]
        self.db.add_events(test_events)
        self.db.conn.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_backup_creates_file(self):
        """Test that backup creates a new file."""
        backup_path = backup_database(self.db_path)
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.startswith(self.db_path))
        self.assertIn('.backup.', backup_path)
    
    def test_backup_preserves_data(self):
        """Test that backup contains the same data as original."""
        backup_path = backup_database(self.db_path)
        
        # Connect to backup and verify data
        conn = sqlite3.connect(backup_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1)
    
    def test_backup_to_custom_directory(self):
        """Test backup to a custom directory."""
        backup_dir = os.path.join(self.temp_dir, 'backups')
        os.makedirs(backup_dir)
        
        backup_path = backup_database(self.db_path, backup_dir)
        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(backup_path.startswith(backup_dir))
    
    def test_list_backups(self):
        """Test listing backups."""
        # Create multiple backups with slight delay to avoid timestamp collision
        import time
        backup1 = backup_database(self.db_path)
        time.sleep(1.1)  # Ensure different timestamp
        backup2 = backup_database(self.db_path)
        
        backups = list_backups(self.db_path)
        self.assertEqual(len(backups), 2)
        
        # Check format
        for backup_path, backup_time, size_mb in backups:
            self.assertTrue(os.path.exists(backup_path))
            self.assertIsInstance(size_mb, float)


class TestSafeDelete(unittest.TestCase):
    """Test safe deletion of events and datapoints."""
    
    def setUp(self):
        """Create test database with multiple events."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        self.db = OsdWorkingDb(self.db_path)
        test_events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2023-01-01T10:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-01T10:00:00Z', 'alarmState': 2, 'hr': 120},
                    {'dataTime': '2023-01-01T10:00:05Z', 'alarmState': 2, 'hr': 125}
                ]
            },
            {
                'id': 2,
                'userId': 100,
                'dataTime': '2023-01-02T10:00:00Z',
                'type': 'False Alarm',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-02T10:00:00Z', 'alarmState': 2, 'hr': 80}
                ]
            },
            {
                'id': 3,
                'userId': 101,
                'dataTime': '2023-01-03T10:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': []
            }
        ]
        self.db.add_events(test_events)
        self.db.conn.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_delete_single_event(self):
        """Test deleting a single event."""
        events_deleted, datapoints_deleted = safe_delete_events(
            self.db_path, [1], create_backup=False
        )
        
        self.assertEqual(events_deleted, 1)
        self.assertEqual(datapoints_deleted, 2)
        
        # Verify event is gone
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events WHERE id = 1")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0)
        
        # Verify other events still exist
        cursor.execute("SELECT COUNT(*) FROM events")
        total = cursor.fetchone()[0]
        self.assertEqual(total, 2)
        conn.close()
    
    def test_delete_multiple_events(self):
        """Test deleting multiple events."""
        events_deleted, datapoints_deleted = safe_delete_events(
            self.db_path, [1, 2], create_backup=False
        )
        
        self.assertEqual(events_deleted, 2)
        self.assertEqual(datapoints_deleted, 3)  # 2 from event 1, 1 from event 2
    
    def test_delete_with_backup(self):
        """Test that delete creates backup when requested."""
        events_deleted, datapoints_deleted = safe_delete_events(
            self.db_path, [1], create_backup=True
        )
        
        # Check backup was created
        backups = list_backups(self.db_path)
        self.assertGreater(len(backups), 0)
    
    def test_cascade_delete_datapoints(self):
        """Test that datapoints are cascade deleted."""
        safe_delete_events(self.db_path, [1], create_backup=False)
        
        # Verify datapoints are also deleted
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM datapoints WHERE event_id = 1")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 0)
        conn.close()


class TestUpdateEventMetadata(unittest.TestCase):
    """Test updating event metadata."""
    
    def setUp(self):
        """Create test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        self.db = OsdWorkingDb(self.db_path)
        test_events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2023-01-01T10:00:00Z',
                'type': 'Seizure',
                'subType': 'Other',
                'desc': 'Original description',
                'osdAlarmState': 2,
                'datapoints': []
            }
        ]
        self.db.add_events(test_events)
        self.db.conn.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_update_description(self):
        """Test updating event description."""
        success = update_event_metadata(
            self.db_path, 1, 'desc', 'Updated description', create_backup=False
        )
        self.assertTrue(success)
        
        # Verify update
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT desc FROM events WHERE id = 1")
        desc = cursor.fetchone()[0]
        self.assertEqual(desc, 'Updated description')
        conn.close()
    
    def test_update_type(self):
        """Test updating event type."""
        success = update_event_metadata(
            self.db_path, 1, 'type', 'False Alarm', create_backup=False
        )
        self.assertTrue(success)
    
    def test_update_seizure_times(self):
        """Test updating seizure times."""
        seizure_times = [10.5, 25.3]
        success = update_event_metadata(
            self.db_path, 1, 'seizureTimes', seizure_times, create_backup=False
        )
        self.assertTrue(success)
        
        # Verify JSON storage
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT seizureTimes FROM events WHERE id = 1")
        stored = cursor.fetchone()[0]
        parsed = json.loads(stored)
        self.assertEqual(parsed, seizure_times)
        conn.close()
    
    def test_update_nonexistent_event(self):
        """Test updating non-existent event returns False."""
        success = update_event_metadata(
            self.db_path, 999, 'desc', 'Test', create_backup=False
        )
        self.assertFalse(success)
    
    def test_update_invalid_field(self):
        """Test updating non-editable field returns False."""
        success = update_event_metadata(
            self.db_path, 1, 'userId', 999, create_backup=False
        )
        self.assertFalse(success)


class TestDatabaseValidation(unittest.TestCase):
    """Test database validation."""
    
    def setUp(self):
        """Create test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        self.db = OsdWorkingDb(self.db_path)
        test_events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2023-01-01T10:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-01T10:00:00Z', 'alarmState': 2}
                ]
            }
        ]
        self.db.add_events(test_events)
        self.db.conn.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_validate_good_database(self):
        """Test validation of healthy database."""
        is_valid, issues = validate_database(self.db_path)
        self.assertTrue(is_valid)
        # May have warnings (like schema version) but no errors
        non_warnings = [i for i in issues if not i.startswith('Warning')]
        self.assertEqual(len(non_warnings), 0)
    
    def test_detect_orphaned_datapoints(self):
        """Test detection of orphaned datapoints."""
        # Manually create orphaned datapoint
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO datapoints (event_id, dataTime, alarmState)
            VALUES (999, '2023-01-01T10:00:00Z', 2)
        """)
        conn.commit()
        conn.close()
        
        is_valid, issues = validate_database(self.db_path)
        self.assertFalse(is_valid)
        self.assertTrue(any('orphaned' in i.lower() for i in issues))


class TestDatabaseStats(unittest.TestCase):
    """Test database statistics."""
    
    def setUp(self):
        """Create test database with various events."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        
        self.db = OsdWorkingDb(self.db_path)
        test_events = [
            {
                'id': 1,
                'userId': 100,
                'dataTime': '2023-01-01T10:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-01T10:00:00Z', 'alarmState': 2}
                ]
            },
            {
                'id': 2,
                'userId': 100,
                'dataTime': '2023-01-02T10:00:00Z',
                'type': 'False Alarm',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2023-01-02T10:00:00Z', 'alarmState': 2},
                    {'dataTime': '2023-01-02T10:00:05Z', 'alarmState': 2}
                ]
            },
            {
                'id': 3,
                'userId': 101,
                'dataTime': '2023-01-03T10:00:00Z',
                'type': 'Seizure',
                'osdAlarmState': 2,
                'datapoints': []
            }
        ]
        self.db.add_events(test_events)
        self.db.conn.close()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = get_database_stats(self.db_path)
        
        self.assertEqual(stats['total_events'], 3)
        self.assertEqual(stats['total_datapoints'], 3)
        self.assertEqual(stats['events_by_type']['Seizure'], 2)
        self.assertEqual(stats['events_by_type']['False Alarm'], 1)
        self.assertIn('database_size_mb', stats)
        self.assertIn('schema_version', stats)


class TestSchemaVersion(unittest.TestCase):
    """Test schema versioning."""
    
    def setUp(self):
        """Create test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_schema_version_in_new_database(self):
        """Test that new databases have schema version."""
        db = OsdWorkingDb(self.db_path)
        db.conn.close()
        
        version = get_schema_version(self.db_path)
        self.assertEqual(version, 1)
    
    def test_schema_version_in_old_database(self):
        """Test handling of database without schema_info table."""
        # Create database without schema_info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE events (id INTEGER PRIMARY KEY, userId INTEGER)
        """)
        conn.commit()
        conn.close()
        
        version = get_schema_version(self.db_path)
        self.assertEqual(version, 0)


if __name__ == '__main__':
    unittest.main()
