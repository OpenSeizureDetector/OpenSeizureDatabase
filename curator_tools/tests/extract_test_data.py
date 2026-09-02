#!/usr/bin/env python3
"""
extract_test_data.py

Extract sample events from existing OSDB JSON files to create test datasets.
This helps build Phase 0 test cases by selecting edge cases and representative samples.

Usage:
    python3 extract_test_data.py --edge-cases
    python3 extract_test_data.py --real-sample --type falseAlarms --count 50
    
"""

import json
import sys
import os
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for libosd imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def load_osdb_file(filepath):
    """Load OSDB JSON file and return list of events."""
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'events' in data:
        return data['events']
    else:
        raise ValueError(f"Unexpected JSON structure in {filepath}")

def analyze_event(event):
    """Analyze an event to extract useful metadata."""
    info = {
        'id': event.get('id'),
        'userId': event.get('userId'),
        'type': event.get('type'),
        'subType': event.get('subType'),
        'dataTime': event.get('dataTime'),
        'num_datapoints': len(event.get('datapoints', [])),
        'has_datapoints_field': 'datapoints' in event,
        'desc': event.get('desc', ''),
        'osdAlarmState': event.get('osdAlarmState')
    }
    return info

def find_edge_cases(events, max_per_category=3):
    """
    Find edge case events suitable for testing:
    - Events with 0 datapoints
    - Events with very few datapoints (1-3)
    - Events with no datapoints field
    - Events with various alarm states
    """
    edge_cases = {
        'zero_datapoints': [],
        'few_datapoints': [],
        'no_datapoints_field': [],
        'large_datapoints': [],
        'various_alarm_states': {}
    }
    
    for event in events:
        info = analyze_event(event)
        
        # Events with 0 datapoints
        if info['has_datapoints_field'] and info['num_datapoints'] == 0:
            if len(edge_cases['zero_datapoints']) < max_per_category:
                edge_cases['zero_datapoints'].append(event)
        
        # Events with 1-3 datapoints
        elif info['has_datapoints_field'] and 1 <= info['num_datapoints'] <= 3:
            if len(edge_cases['few_datapoints']) < max_per_category:
                edge_cases['few_datapoints'].append(event)
        
        # Events without datapoints field (shouldn't exist but check)
        elif not info['has_datapoints_field']:
            if len(edge_cases['no_datapoints_field']) < max_per_category:
                edge_cases['no_datapoints_field'].append(event)
        
        # Events with many datapoints (>100)
        elif info['num_datapoints'] > 100:
            if len(edge_cases['large_datapoints']) < max_per_category:
                edge_cases['large_datapoints'].append(event)
        
        # Various alarm states
        alarm_state = info['osdAlarmState']
        if alarm_state not in edge_cases['various_alarm_states']:
            edge_cases['various_alarm_states'][alarm_state] = []
        if len(edge_cases['various_alarm_states'][alarm_state]) < 2:
            edge_cases['various_alarm_states'][alarm_state].append(event)
    
    return edge_cases

def find_time_boundary_cases(events, max_pairs=3):
    """
    Find pairs of events from the same user that are:
    - Very close in time (< 60 seconds)
    - Near 3-minute boundary (175-185 seconds apart)
    - Just over 3 minutes (185-200 seconds apart)
    """
    from dateutil import parser
    
    # Sort by userId and time
    sorted_events = sorted(events, key=lambda e: (e.get('userId'), e.get('dataTime', '')))
    
    boundary_cases = {
        'very_close': [],      # < 60s apart
        'near_boundary': [],   # 175-185s apart
        'just_over': []        # 185-200s apart
    }
    
    for i in range(len(sorted_events) - 1):
        e1 = sorted_events[i]
        e2 = sorted_events[i + 1]
        
        # Same user?
        if e1.get('userId') != e2.get('userId'):
            continue
        
        # Same type?
        if e1.get('type') != e2.get('type'):
            continue
        
        try:
            t1 = parser.parse(e1.get('dataTime'))
            t2 = parser.parse(e2.get('dataTime'))
            diff_seconds = abs((t2 - t1).total_seconds())
            
            if diff_seconds < 60 and len(boundary_cases['very_close']) < max_pairs * 2:
                boundary_cases['very_close'].extend([e1, e2])
            elif 175 <= diff_seconds <= 185 and len(boundary_cases['near_boundary']) < max_pairs * 2:
                boundary_cases['near_boundary'].extend([e1, e2])
            elif 185 <= diff_seconds <= 200 and len(boundary_cases['just_over']) < max_pairs * 2:
                boundary_cases['just_over'].extend([e1, e2])
        except:
            continue
    
    return boundary_cases

def extract_random_sample(events, count=50, seed=42):
    """Extract a random sample of events."""
    import random
    random.seed(seed)
    
    if len(events) <= count:
        return events
    return random.sample(events, count)

def save_test_dataset(events, output_path, description=""):
    """Save events to JSON file with metadata."""
    output = {
        'description': description,
        'created': datetime.now().isoformat(),
        'num_events': len(events),
        'events': events
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(events)} events to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract test data from OSDB files')
    parser.add_argument('--source-dir', default='/home/graham/osd/osdb/V1.10',
                       help='Directory containing source OSDB files')
    parser.add_argument('--output-dir', 
                       default='../test_data',
                       help='Output directory for test data')
    parser.add_argument('--edge-cases', action='store_true',
                       help='Extract edge cases for testing')
    parser.add_argument('--time-boundaries', action='store_true',
                       help='Extract time boundary test cases')
    parser.add_argument('--real-sample', action='store_true',
                       help='Extract random sample from real data')
    parser.add_argument('--type', default='falseAlarms',
                       choices=['falseAlarms', 'allSeizures', 'fallEvents', 'ndaEvents'],
                       help='Event type to extract')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of events for random sample')
    
    args = parser.parse_args()
    
    # Load source data
    source_file = os.path.join(args.source_dir, f'osdb_3min_{args.type}.json')
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return 1
    
    print(f"Loading data from {source_file}...")
    events = load_osdb_file(source_file)
    print(f"Loaded {len(events)} events")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract edge cases
    if args.edge_cases:
        print("\nExtracting edge cases...")
        edge_cases = find_edge_cases(events)
        
        all_edge_events = []
        print("\nEdge case summary:")
        for category, case_events in edge_cases.items():
            if category == 'various_alarm_states':
                for state, state_events in case_events.items():
                    print(f"  Alarm state {state}: {len(state_events)} events")
                    all_edge_events.extend(state_events)
            else:
                print(f"  {category}: {len(case_events)} events")
                all_edge_events.extend(case_events)
        
        output_path = os.path.join(args.output_dir, 'edge_cases.json')
        save_test_dataset(all_edge_events, output_path,
                         description="Edge cases for validation testing")
    
    # Extract time boundary cases
    if args.time_boundaries:
        print("\nExtracting time boundary cases...")
        boundary_cases = find_time_boundary_cases(events)
        
        all_boundary_events = []
        print("\nTime boundary summary:")
        for category, case_events in boundary_cases.items():
            print(f"  {category}: {len(case_events)} events")
            all_boundary_events.extend(case_events)
        
        output_path = os.path.join(args.output_dir, 'time_boundary_cases.json')
        save_test_dataset(all_boundary_events, output_path,
                         description="Time boundary cases for grouping testing")
    
    # Extract random sample
    if args.real_sample:
        print(f"\nExtracting random sample of {args.count} events...")
        sample_events = extract_random_sample(events, args.count)
        
        output_path = os.path.join(args.output_dir, f'real_sample_{args.type}.json')
        save_test_dataset(sample_events, output_path,
                         description=f"Random sample of {args.count} {args.type} events")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
