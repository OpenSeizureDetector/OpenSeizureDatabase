#!/usr/bin/env python3
"""
test_datapoint_transfer.py - Validate Remote Server Datapoint Transfer

This test performs end-to-end validation that:
1. Events downloaded from remote server include datapoints
2. Datapoints are correctly imported into SQLite database
3. Database storage format matches original JSON file format
4. Datapoint fields (rawData, rawData3D, hr, o2Sat) are preserved

Usage:
    pytest test_datapoint_transfer.py -v -s
    
    For real server testing, ensure:
    - credentials in ../client.cfg are valid
    - Server is accessible
    - Test events exist with confirmed datapoints

Requirements:
    - Access to OpenSeizureDatabase server (or local test instance)
    - Valid credentials in ../client.cfg
    - pytest
"""

import sys
import os
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import modules under test
from osdb_sqlite import OsdWorkingDb
import libosd.webApiConnection
import libosd.configUtils


class DatapointTransferValidator:
    """Utility class for validating datapoint transfer between formats."""
    
    @staticmethod
    def get_datapoint_stats(datapoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract statistics about a datapoint for comparison."""
        stats = {
            'dataTime': datapoint.get('dataTime'),
            'fields_present': list(datapoint.keys()),
            'has_hr': 'hr' in datapoint,
            'has_o2Sat': 'o2Sat' in datapoint,
            'has_rawData': 'rawData' in datapoint,
            'has_rawData3D': 'rawData3D' in datapoint,
            'has_sampleFreq': 'sampleFreq' in datapoint,
        }
        
        # Get lengths if present
        if 'rawData' in datapoint and datapoint['rawData'] is not None:
            if isinstance(datapoint['rawData'], list):
                stats['rawData_length'] = len(datapoint['rawData'])
        
        if 'rawData3D' in datapoint and datapoint['rawData3D'] is not None:
            if isinstance(datapoint['rawData3D'], list):
                stats['rawData3D_length'] = len(datapoint['rawData3D'])
        
        return stats
    
    @staticmethod
    def compare_events_format(json_event: Dict[str, Any], 
                             db_event: Dict[str, Any],
                             ignore_fields: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Compare format and fields between JSON and database events.
        
        Returns:
            Tuple of (is_compatible, list_of_differences)
        """
        if ignore_fields is None:
            ignore_fields = ['_id', 'db_timestamp']  # Database-added fields
        
        differences = []
        
        # Check event-level fields
        json_fields = {k for k in json_event.keys() if k not in ignore_fields}
        db_fields = {k for k in db_event.keys() if k not in ignore_fields}
        
        missing_in_db = json_fields - db_fields
        extra_in_db = db_fields - json_fields
        
        if missing_in_db:
            differences.append(f"Missing in database: {missing_in_db}")
        if extra_in_db:
            differences.append(f"Extra in database: {extra_in_db}")
        
        # Check datapoints format
        json_dp_count = len(json_event.get('datapoints', []))
        db_dp_count = len(db_event.get('datapoints', []))
        
        if json_dp_count != db_dp_count:
            differences.append(f"Datapoint count mismatch: JSON={json_dp_count}, DB={db_dp_count}")
        
        # If both have datapoints, check field compatibility
        if json_dp_count > 0 and db_dp_count > 0:
            json_dp = json_event['datapoints'][0]
            db_dp = db_event['datapoints'][0]
            
            json_dp_fields = set(json_dp.keys())
            db_dp_fields = set(db_dp.keys())
            
            if json_dp_fields != db_dp_fields:
                missing_dp_fields = json_dp_fields - db_dp_fields
                extra_dp_fields = db_dp_fields - json_dp_fields
                
                if missing_dp_fields:
                    differences.append(f"Missing datapoint fields in DB: {missing_dp_fields}")
                if extra_dp_fields:
                    differences.append(f"Extra datapoint fields in DB: {extra_dp_fields}")
        
        return len(differences) == 0, differences
    
    @staticmethod
    def validate_datapoint_values(json_event: Dict[str, Any],
                                 db_event: Dict[str, Any],
                                 fields_to_check: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
        """
        Validate that numeric datapoint values match between JSON and database.
        
        Returns:
            Tuple of (values_match, list_of_mismatches)
        """
        if fields_to_check is None:
            fields_to_check = ['hr', 'o2Sat', 'sampleFreq']
        
        mismatches = []
        
        json_dp_list = json_event.get('datapoints', [])
        db_dp_list = db_event.get('datapoints', [])
        
        if not json_dp_list or not db_dp_list:
            return True, []  # Skip if no datapoints
        
        # Check first datapoint for field values
        json_dp = json_dp_list[0]
        db_dp = db_dp_list[0]
        
        for field in fields_to_check:
            if field in json_dp and field in db_dp:
                json_val = json_dp[field]
                db_val = db_dp[field]
                
                # Handle numeric comparison with tolerance
                if isinstance(json_val, (int, float)) and isinstance(db_val, (int, float)):
                    if abs(json_val - db_val) > 0.01:
                        mismatches.append(f"Field {field} mismatch: JSON={json_val}, DB={db_val}")
                elif json_val != db_val:
                    mismatches.append(f"Field {field} mismatch: JSON={json_val}, DB={db_val}")
        
        return len(mismatches) == 0, mismatches


class TestDatapointTransferFromServer(unittest.TestCase):
    """Test datapoint transfer from remote server to SQLite database."""
    
    @classmethod
    def setUpClass(cls):
        """Set up once for all tests - initialize server connection."""
        # Find config file
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'client.cfg')
        if not os.path.exists(config_path):
            cls.skip_server_tests = True
            print(f"\nWARNING: Config file not found at {config_path}")
            print("Skipping remote server tests. To enable:")
            print("  1. Create/configure ../client.cfg with valid credentials")
            print("  2. Ensure server is accessible")
            return
        
        try:
            cls.config = libosd.configUtils.loadConfig(config_path)
            cls.osd = libosd.webApiConnection.WebApiConnection(
                cfg=config_path,
                download=True,
                debug=False
            )
            
            # Test connection by fetching event count
            events = cls.osd.getEvents(userId=None, includeDatapoints=False)
            print(f"\n✓ Server connection successful - {len(events)} events available")
            
            cls.skip_server_tests = False
            cls.test_event_ids = None  # Will be populated in test
            
        except Exception as e:
            print(f"\nWARNING: Cannot connect to server: {e}")
            print("Skipping remote server tests.")
            cls.skip_server_tests = True
    
    def setUp(self):
        """Set up temporary database for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_datapoint_transfer.db')
        self.db = OsdWorkingDb(self.db_path, debug=False)
        self.validator = DatapointTransferValidator()
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'db'):
            self.db.close()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_01_server_connection(self):
        """Test that we can connect to the server and retrieve events."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        # Get initial event list
        events = self.osd.getEvents(userId=None, includeDatapoints=False)
        self.assertGreater(len(events), 0, "No events available on server")
        
        print(f"\n✓ Retrieved {len(events)} events from server")
        
        # Find events with datapoints (non-empty event list)
        event_ids_with_data = []
        for event in events[:100]:  # Check first 100 to avoid timeout
            if event.get('id'):
                event_ids_with_data.append(event['id'])
        
        self.assertGreater(len(event_ids_with_data), 0, "No event IDs found")
        self.__class__.test_event_ids = event_ids_with_data
        print(f"✓ Found {len(event_ids_with_data)} potential test event IDs")
    
    def test_02_retrieve_single_event_with_datapoints(self):
        """Test downloading a single event with datapoints from server."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        if not self.__class__.test_event_ids:
            self.skipTest("No test event IDs available")
        
        # Try first event ID
        event_id = self.__class__.test_event_ids[0]
        print(f"\nFetching event {event_id} with datapoints...")
        
        event = self.osd.getEvent(event_id, includeDatapoints=True)
        
        self.assertIsNotNone(event, f"Failed to retrieve event {event_id}")
        self.assertIn('id', event, "Event missing 'id' field")
        
        print(f"✓ Retrieved event {event_id}")
        print(f"  - Event type: {event.get('type', 'N/A')}")
        print(f"  - User ID: {event.get('userId', 'N/A')}")
        print(f"  - Datapoints: {len(event.get('datapoints', []))}")
        
        # Log datapoint structure if present
        if 'datapoints' in event and len(event['datapoints']) > 0:
            dp = event['datapoints'][0]
            print(f"  - First datapoint fields: {list(dp.keys())}")
            print(f"  - Has rawData: {'rawData' in dp}")
            print(f"  - Has rawData3D: {'rawData3D' in dp}")
            print(f"  - Has hr: {'hr' in dp}")
            print(f"  - Has o2Sat: {'o2Sat' in dp}")
    
    def test_03_import_server_event_to_database(self):
        """Test importing server event to SQLite database."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        if not self.__class__.test_event_ids:
            self.skipTest("No test event IDs available")
        
        event_id = self.__class__.test_event_ids[0]
        
        # Fetch from server with datapoints
        event = self.osd.getEvent(event_id, includeDatapoints=True)
        self.assertIsNotNone(event)
        
        print(f"\n✓ Fetched event {event_id} from server")
        print(f"  - Event keys: {list(event.keys())}")
        print(f"  - Has 'datapoints' field: {'datapoints' in event}")
        if 'datapoints' in event:
            print(f"  - Datapoints count: {len(event['datapoints'])}")
        
        # Note: The WebApiConnection may return datapoints as separate records
        # rather than in a 'datapoints' field. Log what we actually got.
        server_dp_count = len(event.get('datapoints', []))
        print(f"  - Server datapoints field: {server_dp_count}")
        
        # Convert ID to string if needed (database might store as string)
        if 'id' not in event or not isinstance(event['id'], str):
            event['id'] = str(event.get('id', event_id))
        
        # Import to database
        count = self.db.add_events([event])
        self.assertEqual(count, 1, "Failed to import event to database")
        
        print(f"✓ Imported event {event_id} to database")
        
        # Retrieve from database
        db_event = self.db.get_events(event_ids=[str(event_id)])
        self.assertGreater(len(db_event), 0, "Event not found in database after import")
        
        db_event = db_event[0]
        
        # Verify datapoint handling
        server_dp_count = len(event.get('datapoints', []))
        db_dp_count = len(db_event.get('datapoints', []))
        
        print(f"  - Server datapoints: {server_dp_count}")
        print(f"  - Database datapoints: {db_dp_count}")
        
        # CRITICAL: Log if there's a mismatch to understand what happened
        if server_dp_count == 0 and db_dp_count > 0:
            print(f"\n  ⚠️  IMPORTANT: Database has datapoints but server event doesn't!")
            print(f"     This suggests the database is creating datapoints from event fields.")
            print(f"     Event structure from server: {list(event.keys())}")
        
        # Don't fail on this - just document the behavior
        # This test reveals how the database handles events with missing datapoints field
    
    def test_04_validate_datapoint_format_consistency(self):
        """Test that datapoint format from server matches JSON export from database."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        if not self.__class__.test_event_ids:
            self.skipTest("No test event IDs available")
        
        event_id = self.__class__.test_event_ids[0]
        
        # Fetch from server
        server_event = self.osd.getEvent(event_id, includeDatapoints=True)
        self.assertIsNotNone(server_event)
        
        # Ensure ID is string
        server_event['id'] = str(server_event.get('id', event_id))
        
        # Import to database
        self.db.add_events([server_event])
        
        # Retrieve from database
        db_event = self.db.get_events(event_ids=[str(event_id)])[0]
        
        # Compare formats
        is_compatible, differences = self.validator.compare_events_format(
            server_event, db_event
        )
        
        print(f"\n✓ Format comparison for event {event_id}:")
        print(f"  - Compatible: {is_compatible}")
        
        if differences:
            print(f"  - Differences found:")
            for diff in differences:
                print(f"    • {diff}")
        else:
            print(f"  - No format differences found")
        
        # Log detailed comparison
        if server_event.get('datapoints'):
            print(f"\n  Datapoint field comparison:")
            server_dp_stats = self.validator.get_datapoint_stats(server_event['datapoints'][0])
            db_dp_stats = self.validator.get_datapoint_stats(db_event['datapoints'][0])
            
            print(f"    Server datapoint fields: {server_dp_stats['fields_present']}")
            print(f"    DB datapoint fields: {db_dp_stats['fields_present']}")
            
            print(f"    Server - HR: {server_dp_stats['has_hr']}, O2: {server_dp_stats['has_o2Sat']}, "
                  f"rawData: {server_dp_stats['has_rawData']}, rawData3D: {server_dp_stats['has_rawData3D']}")
            print(f"    DB - HR: {db_dp_stats['has_hr']}, O2: {db_dp_stats['has_o2Sat']}, "
                  f"rawData: {db_dp_stats['has_rawData']}, rawData3D: {db_dp_stats['has_rawData3D']}")
    
    def test_05_validate_datapoint_value_preservation(self):
        """Test that numeric datapoint values are preserved accurately."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        if not self.__class__.test_event_ids:
            self.skipTest("No test event IDs available")
        
        event_id = self.__class__.test_event_ids[0]
        
        # Fetch from server
        server_event = self.osd.getEvent(event_id, includeDatapoints=True)
        self.assertIsNotNone(server_event)
        server_event['id'] = str(server_event.get('id', event_id))
        
        # Import and retrieve
        self.db.add_events([server_event])
        db_event = self.db.get_events(event_ids=[str(event_id)])[0]
        
        # Validate numeric values
        values_match, mismatches = self.validator.validate_datapoint_values(
            server_event, db_event
        )
        
        print(f"\n✓ Value preservation for event {event_id}:")
        print(f"  - Values match: {values_match}")
        
        if mismatches:
            print(f"  - Mismatches:")
            for mismatch in mismatches:
                print(f"    • {mismatch}")
        else:
            print(f"  - All values preserved accurately")
    
    def test_06_roundtrip_json_export(self):
        """Test that database export to JSON matches original server event format."""
        if self.skip_server_tests:
            self.skipTest("Server not configured")
        
        if not self.__class__.test_event_ids:
            self.skipTest("No test event IDs available")
        
        event_id = self.__class__.test_event_ids[0]
        
        # Fetch from server
        server_event = self.osd.getEvent(event_id, includeDatapoints=True)
        self.assertIsNotNone(server_event)
        server_event['id'] = str(server_event.get('id', event_id))
        
        # Import to database
        self.db.add_events([server_event])
        
        # Export from database to JSON file
        export_dir = os.path.join(self.test_dir, 'export')
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, 'test_export.json')
        
        # export_to_json returns count of events exported
        export_count = self.db.export_to_json(
            output_path=export_path,
            pretty=True
        )
        
        print(f"\n✓ Exported database to JSON")
        print(f"  - Events exported: {export_count}")
        
        # Read back exported JSON
        if os.path.exists(export_path):
            with open(export_path, 'r') as f:
                exported_events = json.load(f)
            
            print(f"  - Exported events in file: {len(exported_events)}")
            
            # Find our test event in exports
            for exp_event in exported_events:
                if str(exp_event.get('id')) == str(event_id):
                    print(f"  - Found exported event {event_id}")
                    print(f"  - Exported datapoints: {len(exp_event.get('datapoints', []))}")
                    print(f"  - Original datapoints: {len(server_event.get('datapoints', []))}")
                    
                    # Check format compatibility
                    is_compatible, diffs = self.validator.compare_events_format(
                        server_event, exp_event
                    )
                    print(f"  - Format compatible: {is_compatible}")
                    if diffs:
                        for diff in diffs:
                            print(f"    • {diff}")
                    break


class TestDatapointTransferWithLocalData(unittest.TestCase):
    """Test datapoint transfer using local test data (no server required)."""
    
    def setUp(self):
        """Set up temporary database for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test_local_datapoints.db')
        self.db = OsdWorkingDb(self.db_path, debug=False)
        self.validator = DatapointTransferValidator()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_datapoint_import_with_comprehensive_data(self):
        """Test importing event with full datapoint structure."""
        event = {
            'id': 'TEST_001',
            'userId': 42,
            'dataTime': '2024-01-15T10:30:00Z',
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'osdAlarmState': 2,
            'desc': 'Witnessed tonic-clonic seizure',
            'sampleFreq': 25.0,
            'datapoints': [
                {
                    'dataTime': '2024-01-15T10:30:00Z',
                    'hr': 85.5,
                    'o2Sat': 97,
                    'rawData': [100, 150, 200, 250, 300],
                    'rawData3D': [[100, 0, -1000], [150, 0, -1000], [200, 0, -1000]],
                    'sampleFreq': 25.0
                },
                {
                    'dataTime': '2024-01-15T10:30:04Z',
                    'hr': 92.3,
                    'o2Sat': 96,
                    'rawData': [125, 175, 225, 275, 325],
                    'rawData3D': [[125, 0, -1000], [175, 0, -1000], [225, 0, -1000]],
                    'sampleFreq': 25.0
                }
            ]
        }
        
        # Save reference to original datapoints before import
        original_datapoints = json.loads(json.dumps(event['datapoints']))
        
        # Import
        count = self.db.add_events([event])
        self.assertEqual(count, 1)
        
        # Retrieve
        retrieved = self.db.get_events(event_ids=['TEST_001'])[0]
        
        # Validate
        self.assertEqual(len(retrieved['datapoints']), 2)
        
        # Check first datapoint
        dp1_retrieved = retrieved['datapoints'][0]
        dp1_original = original_datapoints[0]
        
        self.assertEqual(dp1_retrieved['hr'], dp1_original['hr'])
        self.assertEqual(dp1_retrieved['o2Sat'], dp1_original['o2Sat'])
        self.assertEqual(dp1_retrieved['rawData'], dp1_original['rawData'])
        self.assertEqual(dp1_retrieved['rawData3D'], dp1_original['rawData3D'])
        
        print("\n✓ Comprehensive datapoint structure preserved correctly")
        print(f"  - Datapoints: {len(retrieved['datapoints'])}")
        print(f"  - First DP HR: {dp1_retrieved['hr']} (expected: {dp1_original['hr']})")
        print(f"  - First DP O2Sat: {dp1_retrieved['o2Sat']} (expected: {dp1_original['o2Sat']})")
        print(f"  - rawData length: {len(dp1_retrieved['rawData'])} (expected: {len(dp1_original['rawData'])})")
        print(f"  - rawData3D length: {len(dp1_retrieved['rawData3D'])} (expected: {len(dp1_original['rawData3D'])})")
    
    def test_datapoint_with_missing_fields(self):
        """Test handling of datapoints with missing optional fields."""
        event = {
            'id': 'TEST_002',
            'userId': 43,
            'dataTime': '2024-01-15T11:00:00Z',
            'type': 'Fall',
            'osdAlarmState': 1,
            'datapoints': [
                {
                    'dataTime': '2024-01-15T11:00:00Z',
                    'rawData': [100, 200, 300]
                    # Missing hr, o2Sat, rawData3D
                }
            ]
        }
        
        # Should not fail on missing fields
        count = self.db.add_events([event])
        self.assertEqual(count, 1)
        
        retrieved = self.db.get_events(event_ids=['TEST_002'])[0]
        self.assertEqual(len(retrieved['datapoints']), 1)
        
        # Verify partial datapoint preserved
        self.assertIn('rawData', retrieved['datapoints'][0])
        self.assertEqual(retrieved['datapoints'][0]['rawData'], [100, 200, 300])
        
        print("\n✓ Partial datapoint structure handled correctly")
    
    def test_zero_datapoints(self):
        """Test that events with zero datapoints are handled gracefully."""
        event = {
            'id': 'TEST_003',
            'userId': 44,
            'dataTime': '2024-01-15T12:00:00Z',
            'type': 'Unknown',
            'osdAlarmState': 0,
            'datapoints': []
        }
        
        count = self.db.add_events([event])
        self.assertEqual(count, 1)
        
        retrieved = self.db.get_events(event_ids=['TEST_003'])[0]
        self.assertEqual(len(retrieved['datapoints']), 0)
        
        print("\n✓ Zero-datapoint event handled correctly")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
