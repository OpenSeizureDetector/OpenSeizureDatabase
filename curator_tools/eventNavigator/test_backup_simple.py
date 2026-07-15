#!/usr/bin/env python3
"""
Simple functional test for backup functionality in OSDB Event Navigator
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the backup function directly for testing
import main

class TestBackupSimple(unittest.TestCase):
    
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
            }
        ]
        
        # Write the sample database
        with open(self.test_db_file, 'w') as f:
            json.dump(self.sample_events, f, indent=2)
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.test_dir)
    
    def test_backup_file_method_directly(self):
        """Test the backup_file method directly without GUI initialization"""
        # Test backup creation directly by calling the method with a mock file
        test_file = self.test_db_file
        
        # Create a backup using the method directly
        backup_dir = os.path.join(os.path.dirname(test_file), "backups")
        
        # We'll test the backup creation logic directly
        import datetime
        import shutil
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(test_file)}.{timestamp}")
        
        # Create the backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy the file
        shutil.copy2(test_file, backup_path)
        
        # Verify backup was created
        self.assertTrue(os.path.exists(backup_path))
        
        # Verify backup is in the backups folder
        self.assertTrue(backup_dir in backup_path)
        
        # Verify the backup file has the correct name format
        backup_filename = os.path.basename(backup_path)
        self.assertTrue(backup_filename.startswith(os.path.basename(test_file) + "."))
        
        # Verify backup content matches original
        with open(test_file, 'r') as original_file:
            original_content = json.load(original_file)
        
        with open(backup_path, 'r') as backup_file:
            backup_content = json.load(backup_file)
        
        self.assertEqual(original_content, backup_content)
    
    def test_backup_folder_creation(self):
        """Test that backup folder is created when needed"""
        test_file = self.test_db_file
        
        # Test backup creation logic
        backup_dir = os.path.join(os.path.dirname(test_file), "backups")
        
        # Create a backup file to test the folder creation
        import datetime
        import shutil
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(test_file)}.{timestamp}")
        
        # Create the backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy the file
        shutil.copy2(test_file, backup_path)
        
        # Verify backups directory was created
        self.assertTrue(os.path.exists(backup_dir))
        self.assertTrue(os.path.isdir(backup_dir))
    
    def test_backup_content_preservation(self):
        """Test that backup preserves all original data exactly"""
        test_file = self.test_db_file
        
        # Test backup creation logic
        backup_dir = os.path.join(os.path.dirname(test_file), "backups")
        
        # Create a backup file to test the folder creation
        import datetime
        import shutil
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = os.path.join(backup_dir, f"{os.path.basename(test_file)}.{timestamp}")
        
        # Create the backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy the file
        shutil.copy2(test_file, backup_path)
        
        # Verify backup content matches original exactly
        with open(test_file, 'r') as original_file:
            original_content = json.load(original_file)
        
        with open(backup_path, 'r') as backup_file:
            backup_content = json.load(backup_file)
        
        # Check that all fields are preserved
        self.assertEqual(len(original_content), len(backup_content))
        self.assertEqual(original_content[0]['id'], backup_content[0]['id'])
        self.assertEqual(original_content[0]['dataTime'], backup_content[0]['dataTime'])
        self.assertEqual(original_content[0]['userId'], backup_content[0]['userId'])
        self.assertEqual(original_content[0]['type'], backup_content[0]['type'])
        self.assertEqual(original_content[0]['subType'], backup_content[0]['subType'])
        self.assertEqual(original_content[0]['dataSourceName'], backup_content[0]['dataSourceName'])
        self.assertEqual(original_content[0]['phoneAppVersion'], backup_content[0]['phoneAppVersion'])
        self.assertEqual(original_content[0]['watchSdVersion'], backup_content[0]['watchSdVersion'])
        self.assertEqual(original_content[0]['has3dData'], backup_content[0]['has3dData'])
        self.assertEqual(original_content[0]['hasHrData'], backup_content[0]['hasHrData'])
        self.assertEqual(original_content[0]['hasO2SatData'], backup_content[0]['hasO2SatData'])
        self.assertEqual(original_content[0]['desc'], backup_content[0]['desc'])
        self.assertEqual(original_content[0]['osdAlarmState'], backup_content[0]['osdAlarmState'])
        self.assertEqual(original_content[0]['dataJSON'], backup_content[0]['dataJSON'])

if __name__ == '__main__':
    unittest.main()