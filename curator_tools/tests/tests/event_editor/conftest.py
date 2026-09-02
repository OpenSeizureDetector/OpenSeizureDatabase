"""
Pytest configuration and shared fixtures for event_editor tests.
"""

import pytest
import sqlite3
import json
import tempfile
import shutil
import os
import sys
from typing import Dict, Any, List

# Add src directory to path to import OsdWorkingDb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import OsdWorkingDb (database manager without GUI dependencies)
from osdb_sqlite import OsdWorkingDb


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def empty_db(temp_dir):
    """Create an empty database with schema only."""
    db_path = os.path.join(temp_dir, 'empty.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create events table
    cursor.execute("""
        CREATE TABLE events (
            id TEXT PRIMARY KEY,
            userId INTEGER,
            dataTime TEXT,
            type TEXT,
            subType TEXT,
            desc TEXT,
            metadata TEXT,
            seizureTimes TEXT,
            datapoint_count INTEGER
        )
    """)
    
    # Create datapoints table
    cursor.execute("""
        CREATE TABLE datapoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT,
            dataTime TEXT,
            rawData TEXT,
            rawData3D TEXT,
            hr INTEGER,
            FOREIGN KEY (event_id) REFERENCES events(id)
        )
    """)
    
    conn.commit()
    conn.close()
    
    return db_path


@pytest.fixture
def sample_events_db(empty_db):
    """Create a database with sample events for testing."""
    conn = sqlite3.connect(empty_db)
    cursor = conn.cursor()
    
    # Insert test events
    test_events = [
        {
            'id': 'test_001',
            'userId': 1,
            'dataTime': '2024-01-01T10:00:00Z',
            'type': 'Seizure',
            'subType': 'Tonic-Clonic',
            'desc': 'Test seizure event 1',
            'metadata': json.dumps({'source': 'test', 'version': '1.0'}),
            'seizureTimes': json.dumps([-20.0, 50.0]),
            'datapoint_count': 3
        },
        {
            'id': 'test_002',
            'userId': 1,
            'dataTime': '2024-01-02T14:30:00Z',
            'type': 'Seizure',
            'subType': 'Absence',
            'desc': 'Test seizure event 2',
            'metadata': json.dumps({'source': 'test', 'version': '1.0'}),
            'seizureTimes': json.dumps([-5.0, 15.0]),
            'datapoint_count': 2
        },
        {
            'id': 'test_003',
            'userId': 2,
            'dataTime': '2024-01-03T08:15:00Z',
            'type': 'False Alarm',
            'subType': 'Movement',
            'desc': 'False alarm event',
            'metadata': json.dumps({'source': 'test', 'version': '1.0'}),
            'seizureTimes': None,
            'datapoint_count': 2
        },
        {
            'id': 'test_004',
            'userId': 1,
            'dataTime': '2024-01-04T16:45:00Z',
            'type': 'Fall',
            'subType': 'Detected',
            'desc': 'Fall detection event',
            'metadata': json.dumps({'source': 'test', 'version': '1.0'}),
            'seizureTimes': None,
            'datapoint_count': 2
        }
    ]
    
    for event in test_events:
        cursor.execute(
            """INSERT INTO events (id, userId, dataTime, type, subType, desc, metadata, seizureTimes, datapoint_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event['id'], event['userId'], event['dataTime'], event['type'], 
             event['subType'], event['desc'], event['metadata'], 
             event['seizureTimes'], event['datapoint_count'])
        )
    
    # Insert sample datapoints for each event
    for event in test_events:
        for i in range(event['datapoint_count']):
            cursor.execute(
                """INSERT INTO datapoints (event_id, dataTime, rawData, hr)
                   VALUES (?, ?, ?, ?)""",
                (event['id'], event['dataTime'], 
                 json.dumps([0.5] * 125), 70 + i)
            )
    
    conn.commit()
    conn.close()
    
    return empty_db


@pytest.fixture
def filter_test_db(empty_db):
    """Create a database with diverse events for filtering tests."""
    conn = sqlite3.connect(empty_db)
    cursor = conn.cursor()
    
    test_data = [
        ('filter_001', 1, '2024-01-01T10:00:00Z', 'Seizure', 'Tonic-Clonic', 'First seizure', None, None, 3),
        ('filter_002', 1, '2024-01-02T14:00:00Z', 'Seizure', 'Absence', 'Second seizure', None, None, 2),
        ('filter_003', 2, '2024-01-03T08:00:00Z', 'Seizure', 'Tonic-Clonic', 'User 2 seizure', None, None, 3),
        ('filter_004', 1, '2024-01-04T12:00:00Z', 'False Alarm', 'Movement', 'False alarm', None, None, 2),
        ('filter_005', 2, '2024-01-05T16:00:00Z', 'Fall', 'Detected', 'Fall event', None, None, 2),
        ('filter_006', 1, '2024-01-06T10:00:00Z', 'Deleted', 'Tonic-Clonic', 'Deleted event', None, None, 1),
        ('filter_007', 3, '2024-01-07T09:00:00Z', 'Seizure', 'Focal', 'User 3 seizure', None, None, 2),
    ]
    
    for data in test_data:
        cursor.execute(
            """INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data
        )
    
    conn.commit()
    conn.close()
    
    return empty_db


@pytest.fixture
def db_manager(sample_events_db):
    """Create an OsdWorkingDb instance with sample data."""
    manager = OsdWorkingDb(sample_events_db)
    yield manager
    manager.close()


@pytest.fixture
def filter_db_manager(filter_test_db):
    """Create an OsdWorkingDb instance with filtering test data."""
    manager = OsdWorkingDb(filter_test_db)
    yield manager
    manager.close()

