#!/usr/bin/env python3
"""
Demo script showing how to use the OSDB Event Navigator
"""

import os
import sys
import json

# Add the current directory to the path so we can import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_sample_db():
    """Create a sample database file for testing"""
    sample_data = [
        {
            "id": "101",
            "dataTime": "2024-01-02T12:00:00Z",
            "userId": "U9",
            "type": "seizure",
            "subType": "clonic",
            "dataSourceName": "Watch",
            "phoneAppVersion": "1.2.3",
            "watchSdVersion": "2.0.1",
            "has3dData": True,
            "hasHrData": False,
            "hasO2SatData": True,
            "desc": "Sample seizure event",
            "osdAlarmState": "active"
        },
        {
            "id": "102",
            "dataTime": "2024-01-03T14:30:00Z",
            "userId": "U12",
            "type": "seizure",
            "subType": "tonic",
            "dataSourceName": "Phone",
            "phoneAppVersion": "1.2.4",
            "watchSdVersion": "2.0.2",
            "has3dData": False,
            "hasHrData": True,
            "hasO2SatData": False,
            "desc": "Another seizure event",
            "osdAlarmState": "inactive"
        }
    ]
    
    # Write to a test file
    with open('sample_events.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("Created sample_events.json file with 2 sample events")
    return 'sample_events.json'

if __name__ == "__main__":
    print("OSDB Event Navigator Demo")
    print("=" * 30)
    
    # Create a sample database for testing
    db_file = create_sample_db()
    
    print(f"\nTo use the event navigator:")
    print(f"1. Run: python main.py")
    print(f"2. Browse to the directory containing {db_file}")
    print(f"3. Load the database file")
    print(f"4. Navigate through events using Previous/Next buttons")
    
    print("\nFeatures demonstrated:")
    print("- Database folder selection")
    print("- Event loading and display")
    print("- Event navigation")
    print("- Metadata display")
    print("- Graph generation placeholders")