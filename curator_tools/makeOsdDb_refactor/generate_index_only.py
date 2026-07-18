#!/usr/bin/env python3
"""
generate_index_only.py

Standalone script to generate CSV index files from existing JSON event files.
This is useful when you already have JSON files and just need to regenerate
the CSV indexes.

Usage:
    # Generate indexes for all JSON files in directory
    python3 generate_index_only.py --osdb-dir /path/to/osdb
    
    # Generate index for a specific file
    python3 generate_index_only.py --json-file /path/to/osdb_3min_allSeizures.json
    
    # Specify custom output CSV
    python3 generate_index_only.py --json-file input.json --output output.csv
"""

import sys
import os
import argparse
from pathlib import Path

# Add paths for libosd import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import libosd.osdDbConnection


def generate_index_from_json(json_path, csv_path=None, debug=False):
    """
    Generate a CSV index file from a JSON event file.
    
    Parameters:
    json_path - path to input JSON file
    csv_path - path to output CSV file (default: same as JSON with .csv extension)
    debug - print debug information
    
    Returns:
    True if successful, False otherwise
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return False
    
    # Default CSV path
    if csv_path is None:
        csv_path = json_path.replace('.json', '.csv')
    
    try:
        if debug:
            print(f"Loading events from {json_path}...")
        
        # Load events using OsdDbConnection
        osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=debug)
        osd.loadDbFile(json_path, useCacheDir=False)
        
        if debug:
            print(f"Saving index to {csv_path}...")
        
        # Save index file
        osd.saveIndexFile(csv_path, useCacheDir=False)
        
        print(f"✓ Generated: {csv_path}")
        return True
        
    except Exception as e:
        print(f"Error generating index: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def generate_indexes_for_directory(osdb_dir, grouping_period='3min', debug=False):
    """
    Generate CSV index files for all JSON files in a directory.
    
    Parameters:
    osdb_dir - directory containing JSON files
    grouping_period - grouping period string (default: '3min')
    debug - print debug information
    
    Returns:
    Number of successfully generated index files
    """
    print(f"Generating index files for directory: {osdb_dir}")
    print(f"Grouping period: {grouping_period}")
    print("")
    
    # List of expected JSON files
    json_files = [
        f"osdb_{grouping_period}_allSeizures.json",
        f"osdb_{grouping_period}_tcSeizures.json",
        f"osdb_{grouping_period}_fallEvents.json",
        f"osdb_{grouping_period}_falseAlarms.json",
        f"osdb_{grouping_period}_ndaEvents.json",
    ]
    
    success_count = 0
    for json_fname in json_files:
        json_path = os.path.join(osdb_dir, json_fname)
        
        if not os.path.exists(json_path):
            if debug:
                print(f"Skipping {json_fname} (not found)")
            continue
        
        csv_fname = json_fname.replace('.json', '.csv')
        csv_path = os.path.join(osdb_dir, csv_fname)
        
        if generate_index_from_json(json_path, csv_path, debug=debug):
            success_count += 1
    
    print("")
    print(f"✓ Generated {success_count} index file(s)")
    return success_count


def main():
    parser = argparse.ArgumentParser(
        description='Generate CSV index files from OSDB JSON event files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate indexes for all JSON files in directory
  %(prog)s --osdb-dir /path/to/osdb
  
  # Generate index for specific file
  %(prog)s --json-file /path/to/osdb_3min_allSeizures.json
  
  # Specify custom output
  %(prog)s --json-file input.json --output custom_index.csv
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--osdb-dir', 
                       help='Directory containing OSDB JSON files')
    group.add_argument('--json-file',
                       help='Specific JSON file to process')
    
    parser.add_argument('--output',
                        help='Output CSV file (only used with --json-file)')
    parser.add_argument('--grouping-period', default='3min',
                        help='Grouping period for directory processing (default: 3min)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    if args.osdb_dir:
        # Process entire directory
        success_count = generate_indexes_for_directory(
            args.osdb_dir,
            grouping_period=args.grouping_period,
            debug=args.debug
        )
        sys.exit(0 if success_count > 0 else 1)
    
    else:
        # Process single file
        success = generate_index_from_json(
            args.json_file,
            csv_path=args.output,
            debug=args.debug
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
