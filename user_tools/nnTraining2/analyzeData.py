#!/usr/bin/env python3
"""
Analyze a CSV file produced by the runSequence.py toolchain.

Reads a .csv file and reports key statistics including:
  - Total number of unique events (eventId column)
  - Number of seizure events
  - Number of non-seizure events
  - Number of datapoints (rows) for each event
  - Total number of datapoints
"""

import argparse
import csv
import sys
from collections import defaultdict


def analyze_csv(csv_file):
    """
    Analyze a CSV file from the runSequence.py toolchain.
    
    Args:
        csv_file: Path to the CSV file to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    event_data = defaultdict(dict)
    event_types = {}
    total_rows = 0
    header_row = None
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            header_row = reader.fieldnames
            
            if not header_row or 'eventId' not in header_row:
                print("Error: CSV file must contain an 'eventId' column", file=sys.stderr)
                return None
            
            for row in reader:
                total_rows += 1
                event_id = row['eventId'].strip()
                
                if event_id not in event_data:
                    event_data[event_id] = {'count': 0}
                
                event_data[event_id]['count'] += 1
                
                # Store type information if available
                if 'type' in row:
                    event_types[event_id] = row['type']
    
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        return None
    
    # Calculate statistics
    total_events = len(event_data)
    
    seizure_events = 0
    non_seizure_events = 0
    
    for event_id in event_data:
        if event_id in event_types:
            type_val = event_types[event_id].lower()
            if '1' in type_val:
                seizure_events += 1
            else:
                non_seizure_events += 1
    
    # Prepare results
    results = {
        'file': csv_file,
        'total_events': total_events,
        'seizure_events': seizure_events,
        'non_seizure_events': non_seizure_events,
        'total_datapoints': total_rows,
        'event_datapoints': event_data,
        'header_columns': len(header_row) if header_row else 0
    }
    
    return results


def print_results(results):
    """Print analysis results in a formatted manner."""
    
    if results is None:
        return
    
    print("\n" + "="*70)
    print("CSV FILE ANALYSIS REPORT")
    print("="*70)
    print(f"\nFile: {results['file']}")
    print("-"*70)
    
    print(f"\nGLOBAL STATISTICS:")
    print(f"  Total unique events:     {results['total_events']}")
    print(f"  Total datapoints (rows): {results['total_datapoints']}")
    print(f"  Total columns:           {results['header_columns']}")
    
    print(f"\nEVENT TYPE BREAKDOWN:")
    print(f"  Seizure events:          {results['seizure_events']}")
    print(f"  Non-seizure events:      {results['non_seizure_events']}")
    
    if results['total_events'] > 0:
        avg_datapoints = results['total_datapoints'] / results['total_events']
        print(f"\nAVERAGE DATAPOINTS PER EVENT: {avg_datapoints:.2f}")
    
    print(f"\nDATAPOINTS PER EVENT:")
    print(f"{'EventId':<20} {'Type':<30} {'Datapoint Count':<15}")
    print("-"*65)
    
    # Sort events by eventId for consistent output
    for event_id in sorted(results['event_datapoints'].keys()):
        count = results['event_datapoints'][event_id]['count']
        event_type = results['event_datapoints'][event_id].get('type', 'Unknown')
        print(f"{event_id:<20} {event_type:<30} {count:<15}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CSV files produced by the runSequence.py toolchain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analyzeData.py training_data.csv
  python3 analyzeData.py ./output/training/1/flattened_train.csv
        """
    )
    
    parser.add_argument('csv_file', help='Path to the CSV file to analyze')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Print detailed output')
    
    args = parser.parse_args()
    
    results = analyze_csv(args.csv_file)
    print_results(results)
    
    return 0 if results is not None else 1


if __name__ == '__main__':
    sys.exit(main())
