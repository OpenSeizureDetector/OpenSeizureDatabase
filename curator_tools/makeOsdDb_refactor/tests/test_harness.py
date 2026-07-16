#!/usr/bin/env python3
"""
test_harness.py

Test harness for comparing current vs. new makeOsdDb implementations.
Runs test data through makeOsdDb logic and captures results.

Usage:
    python3 test_harness.py --version current
    python3 test_harness.py --version new
    python3 test_harness.py --compare

"""

import json
import sys
import os
import argparse
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import libosd.configUtils

# Import new modules for testing
try:
    import event_validation
    import event_grouping
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: New modules not available: {e}")
    NEW_MODULES_AVAILABLE = False

def load_test_data(test_data_dir='../test_data'):
    """Load all test data files."""
    test_files = {
        'edge_cases': os.path.join(test_data_dir, 'edge_cases.json'),
        'time_boundaries': os.path.join(test_data_dir, 'time_boundary_cases.json'),
        'real_sample_falseAlarms': os.path.join(test_data_dir, 'real_sample_falseAlarms.json'),
        'real_sample_allSeizures': os.path.join(test_data_dir, 'real_sample_allSeizures.json'),
    }
    
    datasets = {}
    for name, filepath in test_files.items():
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
                datasets[name] = data.get('events', data) if isinstance(data, dict) else data
            print(f"Loaded {name}: {len(datasets[name])} events")
        else:
            print(f"Warning: {filepath} not found, skipping")
    
    return datasets

def simulate_current_grouping(events, grouping_period='3min', debug=False):
    """
    Simulate the CURRENT grouping logic from makeOsdDb.py.
    Uses fixed 3-minute time bins via pd.Grouper.
    
    Returns:
        - unique_events: List of selected "unique" events
        - group_info: Details about grouping decisions
    """
    if not events:
        return [], {}
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    
    # Ensure dataTime is datetime (use mixed format to handle variations)
    df['dataTime'] = pd.to_datetime(df['dataTime'], format='mixed', utc=True)
    
    # Group by userId, type, and fixed time period (THIS IS THE BUG)
    grouped = df.groupby(['userId', 'type', pd.Grouper(key='dataTime', freq=grouping_period)])
    
    unique_events = []
    group_info = {
        'total_groups': 0,
        'events_per_group': [],
        'discarded_events': [],
        'selection_method': []
    }
    
    for group_parts, group in grouped:
        userId, eventType, dataTime = group_parts
        group_info['total_groups'] += 1
        group_info['events_per_group'].append(len(group))
        
        if debug:
            print(f"\nGroup: userId={userId}, type={eventType}, time_bin={dataTime}")
            print(f"  Events in group: {len(group)}")
        
        # Select events with ALARM state first
        alarm_rows = group[group['osdAlarmState'] == 2]
        if len(alarm_rows) > 0:
            output_rows = alarm_rows
            method = 'alarm_state'
        else:
            # Select events with non-empty description
            tagged_rows = group[group['desc'].str.len() > 0]
            tagged_rows = tagged_rows[~tagged_rows['desc'].str.contains("null", na=False)]
            if len(tagged_rows) > 0:
                output_rows = tagged_rows
                method = 'tagged'
            else:
                output_rows = group
                method = 'all'
        
        # Pick first event (INDEX 0) - this is what current code does
        selected_event = output_rows.iloc[0].to_dict()
        unique_events.append(selected_event)
        group_info['selection_method'].append(method)
        
        # Track discarded events
        discarded_ids = [int(eid) for eid in group['id'].tolist() if eid != selected_event['id']]
        if discarded_ids:
            group_info['discarded_events'].extend(discarded_ids)
            if debug:
                print(f"  Selected: {selected_event['id']} (method: {method})")
                print(f"  Discarded: {discarded_ids}")
    
    return unique_events, group_info

def validate_events(events, min_datapoints=1):
    """
    Validate events and categorize issues.
    
    Returns:
        - valid_events: List of valid events
        - validation_report: Details about validation issues
    """
    validation_report = {
        'total_checked': len(events),
        'valid': 0,
        'invalid': 0,
        'issues': defaultdict(list)
    }
    
    valid_events = []
    
    for event in events:
        event_id = event.get('id', 'unknown')
        is_valid = True
        
        # Check required fields
        required_fields = ['id', 'userId', 'dataTime', 'type', 'osdAlarmState']
        missing = [f for f in required_fields if f not in event]
        if missing:
            validation_report['issues']['missing_required_fields'].append({
                'id': event_id,
                'missing': missing
            })
            is_valid = False
        
        # Check datapoints
        if 'datapoints' not in event:
            validation_report['issues']['no_datapoints_field'].append(event_id)
            is_valid = False
        elif not isinstance(event['datapoints'], list):
            validation_report['issues']['datapoints_not_list'].append(event_id)
            is_valid = False
        elif len(event['datapoints']) < min_datapoints:
            validation_report['issues']['insufficient_datapoints'].append({
                'id': event_id,
                'count': len(event['datapoints'])
            })
            is_valid = False
        
        if is_valid:
            valid_events.append(event)
            validation_report['valid'] += 1
        else:
            validation_report['invalid'] += 1
    
    return valid_events, validation_report

def analyze_time_proximity(events):
    """
    Analyze time proximity between events to identify grouping edge cases.
    Returns pairs of events and their time differences.
    """
    from dateutil import parser
    
    # Sort by userId and time
    sorted_events = sorted(events, key=lambda e: (e.get('userId'), e.get('dataTime', '')))
    
    proximity_analysis = []
    
    for i in range(len(sorted_events) - 1):
        e1 = sorted_events[i]
        e2 = sorted_events[i + 1]
        
        # Only compare same user and type
        if e1.get('userId') != e2.get('userId') or e1.get('type') != e2.get('type'):
            continue
        
        try:
            t1 = parser.parse(e1.get('dataTime'))
            t2 = parser.parse(e2.get('dataTime'))
            diff_seconds = abs((t2 - t1).total_seconds())
            
            proximity_analysis.append({
                'event1_id': e1.get('id'),
                'event2_id': e2.get('id'),
                'userId': e1.get('userId'),
                'type': e1.get('type'),
                'time_diff_seconds': diff_seconds,
                'should_group_3min': diff_seconds <= 180,
                'time1': e1.get('dataTime'),
                'time2': e2.get('dataTime')
            })
        except:
            pass
    
    return proximity_analysis

def simulate_new_grouping(events, grouping_period='3min', debug=False):
    """
    Simulate the NEW grouping logic using sliding window approach.
    Uses the new event_grouping module.
    
    Returns:
        - unique_events: List of selected "unique" events
        - group_info: Details about grouping decisions
    """
    if not NEW_MODULES_AVAILABLE:
        raise RuntimeError("New modules not available - cannot test new implementation")
    
    if not events:
        return [], {}
    
    # Use the new sliding window grouping
    unique_events, grouping_info = event_grouping.apply_sliding_window_grouping(
        events,
        time_threshold=grouping_period,
        selection_strategy='alarm_first',
        debug=debug
    )
    
    # Convert to format compatible with test harness
    group_info = {
        'total_groups': grouping_info['total_groups'],
        'events_per_group': grouping_info['events_per_group'],
        'discarded_events': grouping_info['discarded_events'],
        'selection_method': [grouping_info['selection_strategy']] * grouping_info['total_groups']
    }
    
    return unique_events, group_info

def run_tests(datasets, version='current', output_dir='../test_results', debug=False):
    """
    Run tests on the specified version.
    
    Args:
        datasets: Dictionary of test datasets
        version: 'current' or 'new'
        output_dir: Where to save results
        debug: Print detailed debug information
    """
    print(f"\n{'='*60}")
    print(f"Running tests for VERSION: {version.upper()}")
    print(f"{'='*60}\n")
    
    results_dir = os.path.join(output_dir, f"{version}_version")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset_name, events in datasets.items():
        print(f"\nTesting dataset: {dataset_name} ({len(events)} events)")
        print("-" * 50)
        
        # Step 1: Validation
        print("  1. Validating events...")
        valid_events, validation_report = validate_events(events)
        print(f"     Valid: {validation_report['valid']}, Invalid: {validation_report['invalid']}")
        
        if validation_report['invalid'] > 0:
            print(f"     Issues found:")
            for issue_type, issue_list in validation_report['issues'].items():
                print(f"       - {issue_type}: {len(issue_list)}")
        
        # Step 2: Grouping
        if version == 'current':
            print("  2. Applying CURRENT grouping (fixed 3-min bins)...")
            unique_events, group_info = simulate_current_grouping(valid_events, debug=debug)
            print(f"     Input: {len(valid_events)} events")
            print(f"     Groups: {group_info['total_groups']}")
            print(f"     Output: {len(unique_events)} unique events")
            print(f"     Discarded: {len(group_info['discarded_events'])} events")
        else:
            print("  2. Applying NEW grouping (sliding window proximity)...")
            if not NEW_MODULES_AVAILABLE:
                print("     ERROR: New modules not available!")
                unique_events = valid_events
                group_info = {'error': 'New implementation not available'}
            else:
                unique_events, group_info = simulate_new_grouping(valid_events, debug=debug)
                print(f"     Input: {len(valid_events)} events")
                print(f"     Groups: {group_info['total_groups']}")
                print(f"     Output: {len(unique_events)} unique events")
                print(f"     Discarded: {len(group_info['discarded_events'])} events")
        
        # Step 3: Time proximity analysis
        print("  3. Analyzing time proximity...")
        proximity_analysis = analyze_time_proximity(events)
        print(f"     Found {len(proximity_analysis)} event pairs to analyze")
        
        # Save results
        result = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'version': version,
            'input_events': len(events),
            'validation_report': validation_report,
            'unique_events': unique_events,
            'group_info': group_info,
            'proximity_analysis': proximity_analysis
        }
        
        output_file = os.path.join(results_dir, f'{dataset_name}_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  4. Saved results to {output_file}")
        
        all_results[dataset_name] = result
    
    # Generate summary report
    summary_file = os.path.join(results_dir, 'test_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Test Summary - {version.upper()} Version\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        
        for dataset_name, result in all_results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"  Input events: {result['input_events']}\n")
            f.write(f"  Valid events: {result['validation_report']['valid']}\n")
            f.write(f"  Invalid events: {result['validation_report']['invalid']}\n")
            f.write(f"  Output events: {len(result['unique_events'])}\n")
            if 'discarded_events' in result['group_info']:
                f.write(f"  Discarded events: {len(result['group_info']['discarded_events'])}\n")
            f.write("\n")
    
    print(f"\n{'='*60}")
    print(f"Test summary saved to {summary_file}")
    print(f"{'='*60}\n")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Test harness for makeOsdDb refactoring')
    parser.add_argument('--version', choices=['current', 'new'], default='current',
                       help='Which version to test')
    parser.add_argument('--test-data-dir', default='../test_data',
                       help='Directory containing test data')
    parser.add_argument('--output-dir', default='../test_results',
                       help='Directory for test results')
    parser.add_argument('--debug', action='store_true',
                       help='Print detailed debug information')
    
    args = parser.parse_args()
    
    # Load test data
    print("Loading test data...")
    datasets = load_test_data(args.test_data_dir)
    
    if not datasets:
        print("Error: No test data found")
        return 1
    
    # Run tests
    results = run_tests(datasets, args.version, args.output_dir, args.debug)
    
    print("\n✅ Testing complete!")
    print(f"Results saved to: {os.path.abspath(args.output_dir)}/{args.version}_version/")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
