#!/usr/bin/env python3
"""
test_generateGraphs.py
Test script for generateGraphs.py functionality
"""

import sys
import os
import json
import tempfile
import shutil

# Make the curator_tools folder accessible
sys.path.insert(0, os.path.dirname(__file__))
import generateGraphs


def create_test_json_file(filename, event_type, num_events):
    """Create a test JSON file with sample events"""
    events = []
    users = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']
    
    for i in range(num_events):
        user_id = users[i % len(users)]
        event = {
            'id': 1000 + i,
            'userId': user_id,
            'type': event_type,
            'subType': 'test',
            'dataTime': f'2024-{1 + (i // 20):02d}-{1 + (i % 20):02d}T12:00:00Z',
            'desc': f'Test {event_type} event {i}',
            'flags': {'reviewed': False}
        }
        events.append(event)
    
    with open(filename, 'w') as f:
        json.dump(events, f, indent=2)
    
    return filename


def test_graph_generation():
    """Test the graph generation functionality"""
    print("Testing generateGraphs.py...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, 'output')
    
    try:
        # Create test JSON files
        seizure_file = os.path.join(temp_dir, 'seizures.json')
        false_alarm_file = os.path.join(temp_dir, 'false_alarms.json')
        nda_file = os.path.join(temp_dir, 'nda.json')
        
        print(f"Creating test data in {temp_dir}...")
        create_test_json_file(seizure_file, 'seizure', 50)
        create_test_json_file(false_alarm_file, 'false alarm', 20)
        create_test_json_file(nda_file, 'nda', 30)
        
        print(f"Generating graphs to {output_dir}...")
        success = generateGraphs.generate_all_graphs(
            [seizure_file, false_alarm_file, nda_file],
            output_dir,
            threshold=5,
            debug=True
        )
        
        if success:
            print("\nTest PASSED!")
            print(f"Output files:")
            for f in os.listdir(output_dir):
                filepath = os.path.join(output_dir, f)
                size = os.path.getsize(filepath)
                print(f"  - {f} ({size} bytes)")
            return True
        else:
            print("\nTest FAILED: Graph generation returned False")
            return False
            
    except Exception as e:
        print(f"\nTest FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    success = test_graph_generation()
    sys.exit(0 if success else 1)
