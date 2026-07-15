#!/usr/bin/env python3
"""
Unit tests for metadata modification functionality in OSDB Event Navigator
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import EventNavigatorGUI

class TestMetadataModification(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for our test files
        self.test_dir = tempfile.mkdtemp()
        self.test_db_file = os.path.join(self.test_dir, "test_osdb.json")
        
        # Create a sample database file for testing
        self.sample_events = [
            {
                "id": "1",
                "dataTime": "2023-01-01T00:00:00Z",
                "userId": "user1",
                "type": "seizure",
                "subType": "general",
                "dataSourceName": "test_device",
                "phoneAppVersion": "1.0",
                "watchSdVersion": "1.0",
                "has3dData": True,
                "hasHrData": False,
                "hasO2SatData": False,
                "desc": "Sample seizure event",
                "osdAlarmState": "active",
                "dataJSON": '{"sample": "data"}'
            },
            {
                "id": "2",
                "dataTime": "2023-01-02T00:00:00Z",
                "userId": "user2",
                "type": "fall",
                "subType": "simple",
                "dataSourceName": "test_device",
                "phoneAppVersion": "1.0",
                "watchSdVersion": "1.0",
                "has3dData": True,
                "hasHrData": False,
                "hasO2SatData": False,
                "desc": "Sample fall event",
                "osdAlarmState": "inactive",
                "dataJSON": '{"sample": "data2"}'
            }
        ]
        
        # Write the sample database
        with open(self.test_db_file, 'w') as f:
            json.dump(self.sample_events, f, indent=2)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def test_backup_creation(self):
        """Test that backup files are created correctly in the backups folder"""
        # Create a mock GUI instance
        root = MagicMock()
        app = EventNavigatorGUI(root)
        
        # Test backup creation
        backup_path = app.backup_file(self.test_db_file)
        
        # Verify backup was created
        self.assertIsNotNone(backup_path)
        self.assertTrue(os.path.exists(backup_path))
        
        # Verify backup is in the backups folder
        backup_dir = os.path.join(os.path.dirname(self.test_db_file), "backups")
        self.assertTrue(backup_dir in backup_path)
        
        # Verify the backup file has the correct name format
        backup_filename = os.path.basename(backup_path)
        self.assertTrue(backup_filename.startswith(os.path.basename(self.test_db_file) + "."))
        
        # Verify backup content matches original
        with open(self.test_db_file, 'r') as original_file:
            original_content = json.load(original_file)
        
        with open(backup_path, 'r') as backup_file:
            backup_content = json.load(backup_file)
        
        self.assertEqual(original_content, backup_content)
    
    def test_event_modification_preserves_structure(self):
        """Test that modifying event metadata preserves the overall structure"""
        # Create a mock GUI instance
        root = MagicMock()
        app = EventNavigatorGUI(root)
        
        # Mock the database loading to use our test file
        with patch.object(app, 'db_folder_var') as mock_folder_var:
            mock_folder_var.get.return_value = self.test_dir
            
            # Mock the database connection
            with patch('main.OsdDbConnection') as mock_db:
                mock_instance = MagicMock()
                mock_db.return_value = mock_instance
                
                # Mock the getAllEvents method to return our sample events
                mock_instance.getAllEvents.return_value = self.sample_events
                
                # Load database (this will set up the events)
                app.load_database()
                
                # Verify events were loaded correctly
                self.assertEqual(len(app.events), 2)
                self.assertEqual(app.events[0]['id'], '1')
                self.assertEqual(app.events[1]['id'], '2')
                
                # Modify event metadata
                original_event = app.events[0].copy()
                app.type_var.set("modified_seizure")
                app.subtype_var.set("modified_general")
                app.desc_var.set("Modified description")
                
                # Save changes
                app.save_changes()
                
                # Verify changes were applied
                self.assertEqual(app.events[0]['type'], "modified_seizure")
                self.assertEqual(app.events[0]['subType'], "modified_general")
                self.assertEqual(app.events[0]['desc'], "Modified description")
                
                # Verify other fields are preserved
                self.assertEqual(app.events[0]['id'], '1')
                self.assertEqual(app.events[0]['dataTime'], '2023-01-01T00:00:00Z')
                self.assertEqual(app.events[0]['userId'], 'user1')
                self.assertEqual(app.events[0]['dataSourceName'], 'test_device')
                self.assertEqual(app.events[0]['phoneAppVersion'], '1.0')
                self.assertEqual(app.events[0]['watchSdVersion'], '1.0')
                self.assertEqual(app.events[0]['has3dData'], True)
                self.assertEqual(app.events[0]['hasHrData'], False)
                self.assertEqual(app.events[0]['hasO2SatData'], False)
                self.assertEqual(app.events[0]['osdAlarmState'], 'active')
                self.assertEqual(app.events[0]['dataJSON'], '{"sample": "data"}')
    
    def test_save_to_file_preserves_database_integrity(self):
        """Test that saving to file preserves database integrity"""
        # Create a mock GUI instance
        root = MagicMock()
        app = EventNavigatorGUI(root)
        
        # Mock the database loading to use our test file
        with patch.object(app, 'db_folder_var') as mock_folder_var:
            mock_folder_var.get.return_value = self.test_dir
            
            # Mock the database connection
            with patch('main.OsdDbConnection') as mock_db:
                mock_instance = MagicMock()
                mock_db.return_value = mock_instance
                
                # Mock the getAllEvents method to return our sample events
                mock_instance.getAllEvents.return_value = self.sample_events
                
                # Load database (this will set up the events)
                app.load_database()
                
                # Modify an event
                app.type_var.set("new_type")
                app.subtype_var.set("new_subtype")
                app.desc_var.set("new_description")
                app.save_changes()
                
                # Mock the save_to_file method to test it
                with patch.object(app, 'backup_file') as mock_backup:
                    mock_backup.return_value = os.path.join(self.test_dir, "backups", "test_osdb.json.20230101120000")
                    
                    # Test saving to file
                    result = app.save_to_file(self.test_db_file)
                    
                    # Verify save was successful
                    self.assertTrue(result)
                    
                    # Verify the file was actually written
                    with open(self.test_db_file, 'r') as f:
                        saved_content = json.load(f)
                    
                    # Verify content was updated
                    self.assertEqual(saved_content[0]['type'], "new_type")
                    self.assertEqual(saved_content[0]['subType'], "new_subtype")
                    self.assertEqual(saved_content[0]['desc'], "new_description")
                    
                    # Verify other events are preserved
                    self.assertEqual(len(saved_content), 2)
                    self.assertEqual(saved_content[1]['id'], '2')
    
    def test_backup_folder_creation(self):
        """Test that backup folder is created when needed"""
        # Create a mock GUI instance
        root = MagicMock()
        app = EventNavigatorGUI(root)
        
        # Test backup creation with a file that doesn't have a backups folder yet
        backup_path = app.backup_file(self.test_db_file)
        
        # Verify backup was created
        self.assertIsNotNone(backup_path)
        self.assertTrue(os.path.exists(backup_path))
        
        # Verify backups directory was created
        backup_dir = os.path.join(os.path.dirname(self.test_db_file), "backups")
        self.assertTrue(os.path.exists(backup_dir))
        self.assertTrue(os.path.isdir(backup_dir))
    
    def test_database_loading_with_backups_folder(self):
        """Test that database loading works correctly when backups folder exists"""
        # Create a backup folder
        backup_dir = os.path.join(self.test_dir, "backups")
        os.makedirs(backup_dir)
        
        # Create a backup file
        backup_file = os.path.join(backup_dir, "test_osdb.json.20230101120000")
        with open(backup_file, 'w') as f:
            json.dump(self.sample_events, f, indent=2)
        
        # Create a mock GUI instance
        root = MagicMock()
        app = EventNavigatorGUI(root)
        
        # Mock the database loading to use our test file
        with patch.object(app, 'db_folder_var') as mock_folder_var:
            mock_folder_var.get.return_value = self.test_dir
            
            # Mock the database connection
            with patch('main.OsdDbConnection') as mock_db:
                mock_instance = MagicMock()
                mock_db.return_value = mock_instance
                
                # Mock the getAllEvents method to return our sample events
                mock_instance.getAllEvents.return_value = self.sample_events
                
                # Load database - this should work without issues even with backups folder
                app.load_database()
                
                # Verify events were loaded
                self.assertEqual(len(app.events), 2)

if __name__ == '__main__':
    unittest.main()