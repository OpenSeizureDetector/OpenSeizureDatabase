#!/usr/bin/env python3
"""
Create a test database with sample events for testing the PDF generator.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from osdb_sqlite import OsdWorkingDb

def create_test_db(db_path: str = 'test_osdb.db'):
    """Create a test database with sample events."""
    
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create database
    db = OsdWorkingDb(db_path, debug=True)
    
    # Create sample events with datapoints
    sample_events = []
    
    for event_num in range(1, 6):
        event_id = f"test_event_{event_num}"
        user_id = 100 + event_num
        
        # Create event data
        event = {
            'id': event_id,
            'userId': user_id,
            'dataTime': f"2024-01-{10+event_num:02d}T10:30:{event_num*10:02d}Z",
            'type': 'Seizure' if event_num % 2 == 0 else 'Fall',
            'subType': 'Tonic-Clonic' if event_num % 2 == 0 else 'Uncontrolled',
            'desc': f'Test event {event_num} - Sample description for testing',
            'osdAlarmState': 2,
            'dataSourceName': 'Watch',
            'hasHrData': True,
            'has3dData': True,
            'datapoint_count': 3,
            'seizureTimes': [5.0, 15.0] if event_num % 2 == 0 else None,
            'datapoints': []
        }
        
        # Create 3 sample datapoints for each event
        for dp_num in range(3):
            dp_time_offset = dp_num * 5
            datapoint = {
                'dataTime': f"2024-01-{10+event_num:02d}T10:30:{event_num*10 + dp_time_offset:02d}Z",
                'alarmState': 2,
                'hr': 80 + dp_num * 10,  # 80, 90, 100 bpm
                'o2Sat': 98,
                'rawData': [500 + i * 10 for i in range(125)],  # Simulated acceleration samples
                'rawData3D': [int(500 + (i%3)*100) for i in range(375)],  # Simulated 3D acceleration
                'specPower': 1.0,
                'roiPower': 0.5,
                'roiRatio': 0.5,
                'maxVal': 600,
                'maxFreq': 50.0
            }
            event['datapoints'].append(datapoint)
        
        sample_events.append(event)
    
    # Import events into database
    db.add_events(sample_events)
    
    # Verify import
    all_events = db.get_events(include_datapoints=False)
    print(f"Created {len(all_events)} test events")
    
    db.close()
    print(f"Test database created: {db_path}")
    return db_path

if __name__ == '__main__':
    db_path = create_test_db()
    print(f"\nTest database: {db_path}")
    print("You can now test the PDF generator:")
    print(f"  python3 generate_pdf_summary.py {db_path} test_output.pdf")
