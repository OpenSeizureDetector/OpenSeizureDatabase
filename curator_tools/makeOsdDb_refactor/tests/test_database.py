#!/usr/bin/env python3
"""
test_database.py - Tests for SQLite Working Database (Phase 4)

Tests verify:
- Data integrity through import/export
- Consistency with original JSON format
- Query functionality
- Database statistics
"""

import sys
import os
import json
import tempfile
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from osdb_sqlite import OsdWorkingDb


class TestDatabaseImportExport(unittest.TestCase):
    """Test database import and export functionality."""
    
    def setUp(self):
        """Create temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.db = OsdWorkingDb(self.db_path, debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_import_simple_events(self):
        """Test importing simple events to database."""
        events = [
            {
                'id': '1',
                'userId': 42,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 80, 'o2Sat': 95}
                ]
            },
            {
                'id': '2',
                'userId': 42,
                'dataTime': '2024-01-01T11:00:00Z',
                'type': 'FalseAlarm',
                'datapoints': []
            }
        ]
        
        count = self.db.add_events(events)
        
        assert count == 2
        
        # Verify events are in database
        retrieved = self.db.get_events()
        assert len(retrieved) == 2
        assert retrieved[0]['id'] == '1'
        assert retrieved[1]['id'] == '2'
    
    def test_datapoint_preservation(self):
        """Test that datapoints are preserved correctly."""
        events = [
            {
                'id': '100',
                'userId': 1,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 70, 'o2Sat': 98},
                    {'dataTime': '2024-01-01T10:00:05Z', 'hr': 75, 'o2Sat': 97},
                    {'dataTime': '2024-01-01T10:00:10Z', 'hr': 80, 'o2Sat': 96}
                ]
            }
        ]
        
        self.db.add_events(events)
        retrieved = self.db.get_events(event_ids=['100'])
        
        assert len(retrieved) == 1
        assert len(retrieved[0]['datapoints']) == 3
        assert retrieved[0]['datapoints'][0]['hr'] == 70
        assert retrieved[0]['datapoints'][1]['hr'] == 75
        assert retrieved[0]['datapoints'][2]['hr'] == 80
    
    def test_rawData_preservation(self):
        """Test that rawData and rawData3D are preserved."""
        events = [
            {
                'id': '200',
                'userId': 1,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'datapoints': [
                    {
                        'dataTime': '2024-01-01T10:00:00Z',
                        'hr': 70,
                        'rawData': [100, 200, 300],
                        'rawData3D': [[1, 2, 3], [4, 5, 6]]
                    }
                ]
            }
        ]
        
        self.db.add_events(events)
        retrieved = self.db.get_events(event_ids=['200'])
        
        assert len(retrieved[0]['datapoints']) == 1
        dp = retrieved[0]['datapoints'][0]
        assert 'rawData' in dp
        assert dp['rawData'] == [100, 200, 300]
        assert 'rawData3D' in dp
        assert dp['rawData3D'] == [[1, 2, 3], [4, 5, 6]]
    
    def test_metadata_preservation(self):
        """Test that extra metadata fields are preserved."""
        events = [
            {
                'id': '300',
                'userId': 1,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'customField1': 'value1',
                'customField2': 123,
                'customField3': {'nested': 'data'},
                'datapoints': []
            }
        ]
        
        self.db.add_events(events)
        retrieved = self.db.get_events(event_ids=['300'])
        
        assert retrieved[0]['customField1'] == 'value1'
        assert retrieved[0]['customField2'] == 123
        assert retrieved[0]['customField3'] == {'nested': 'data'}
    
    def test_merged_events_preservation(self):
        """Test that merged event metadata is preserved."""
        events = [
            {
                'id': '400',
                'userId': 1,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'merged_from_events': ['401', '402', '403'],
                'merged_event_count': 3,
                'datapoints': []
            }
        ]
        
        self.db.add_events(events)
        retrieved = self.db.get_events(event_ids=['400'])
        
        assert retrieved[0]['merged_from_events'] == ['401', '402', '403']
        assert retrieved[0]['merged_event_count'] == 3


class TestDatabaseQuerying(unittest.TestCase):
    """Test database query functionality."""
    
    def setUp(self):
        """Create temporary database with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.db = OsdWorkingDb(self.db_path, debug=False)
        
        # Add sample events
        self.sample_events = [
            {
                'id': '1', 'userId': 1, 'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure', 'subType': 'Tonic-Clonic', 'datapoints': []
            },
            {
                'id': '2', 'userId': 1, 'dataTime': '2024-01-01T11:00:00Z',
                'type': 'Seizure', 'subType': 'Absence', 'datapoints': []
            },
            {
                'id': '3', 'userId': 2, 'dataTime': '2024-01-01T12:00:00Z',
                'type': 'FalseAlarm', 'datapoints': []
            },
            {
                'id': '4', 'userId': 1, 'dataTime': '2024-01-02T10:00:00Z',
                'type': 'Fall', 'datapoints': []
            }
        ]
        self.db.add_events(self.sample_events)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.db.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_query_by_user(self):
        """Test querying by user ID."""
        events = self.db.get_events(user_id=1)
        assert len(events) == 3
        assert all(e['userId'] == 1 for e in events)
    
    def test_query_by_type(self):
        """Test querying by event type."""
        events = self.db.get_events(event_type='Seizure')
        assert len(events) == 2
        assert all(e['type'] == 'Seizure' for e in events)
    
    def test_query_by_subtype(self):
        """Test querying by event subtype."""
        events = self.db.get_events(event_type='Seizure', event_subtype='Tonic-Clonic')
        assert len(events) == 1
        assert events[0]['id'] == '1'
    
    def test_query_by_time_range(self):
        """Test querying by time range."""
        events = self.db.get_events(
            start_time='2024-01-01T11:00:00Z',
            end_time='2024-01-01T23:59:59Z'
        )
        assert len(events) == 2
        assert events[0]['id'] == '2'
        assert events[1]['id'] == '3'
    
    def test_query_by_event_ids(self):
        """Test querying by specific event IDs."""
        events = self.db.get_events(event_ids=['1', '3'])
        assert len(events) == 2
        assert {e['id'] for e in events} == {'1', '3'}


class TestDatabaseConsistency(unittest.TestCase):
    """Test consistency between JSON and database."""
    
    def setUp(self):
        """Create temporary files."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test.db')
        self.json_path = os.path.join(self.temp_dir, 'test.json')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_roundtrip_consistency(self):
        """Test that JSON → DB → JSON preserves data."""
        # Create sample data
        original_events = [
            {
                'id': '100',
                'userId': 42,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'desc': 'Test seizure',
                'osdAlarmState': 2,
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 80, 'o2Sat': 95},
                    {'dataTime': '2024-01-01T10:00:05Z', 'hr': 85, 'o2Sat': 94}
                ]
            },
            {
                'id': '101',
                'userId': 42,
                'dataTime': '2024-01-01T11:00:00Z',
                'type': 'FalseAlarm',
                'merged_from_events': ['101', '102'],
                'merged_event_count': 2,
                'datapoints': []
            }
        ]
        
        # Save to JSON
        with open(self.json_path, 'w') as f:
            json.dump(original_events, f)
        
        # Import to database
        db = OsdWorkingDb(self.db_path, debug=False)
        db.import_from_json(self.json_path)
        
        # Export back to JSON
        export_path = os.path.join(self.temp_dir, 'export.json')
        db.export_to_json(export_path)
        db.close()
        
        # Load exported data
        with open(export_path, 'r') as f:
            exported_events = json.load(f)
        
        # Verify consistency
        assert len(exported_events) == len(original_events)
        
        # Check first event
        orig_e1 = original_events[0]
        exp_e1 = exported_events[0]
        assert exp_e1['id'] == orig_e1['id']
        assert exp_e1['userId'] == orig_e1['userId']
        assert exp_e1['type'] == orig_e1['type']
        assert len(exp_e1['datapoints']) == len(orig_e1['datapoints'])
        
        # Check second event
        exp_e2 = exported_events[1]
        assert exp_e2['merged_from_events'] == ['101', '102']
        assert exp_e2['merged_event_count'] == 2


class TestDatabaseStatistics(unittest.TestCase):
    """Test database statistics functionality."""
    
    def test_statistics_calculation(self):
        """Test that statistics are calculated correctly."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test.db')
        
        try:
            db = OsdWorkingDb(db_path, debug=False)
            
            # Add sample events
            events = [
                {
                    'id': i,
                    'userId': i % 3,  # 3 users
                    'dataTime': f'2024-01-0{(i%5)+1}T10:00:00Z',
                    'type': 'Seizure' if i % 2 == 0 else 'FalseAlarm',
                    'datapoints': [{'dataTime': f'2024-01-0{(i%5)+1}T10:00:00Z'}] * (i % 10)
                }
                for i in range(1, 21)
            ]
            
            db.add_events(events)
            
            # Get statistics
            stats = db.get_statistics()
            
            assert stats['total_events'] == 20
            assert stats['unique_users'] == 3
            assert stats['unique_types'] == 2
            assert 'Seizure' in stats['events_by_type']
            assert 'FalseAlarm' in stats['events_by_type']
            
            db.close()
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
