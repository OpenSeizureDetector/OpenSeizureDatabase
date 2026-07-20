#!/usr/bin/env python3
"""
generate_graphs_only.py

Standalone script to generate summary graphs from existing JSON event files.
This is useful when you already have JSON files and just need to regenerate
or update the summary graphs.

Usage:
    # Generate graphs from all JSON files in directory
    python3 generate_graphs_only.py --osdb-dir /path/to/osdb
    
    # Generate graphs from specific files
    python3 generate_graphs_only.py --json-files file1.json file2.json file3.json
    
    # Customize output directory and threshold
    python3 generate_graphs_only.py --osdb-dir /path/to/osdb --output /path/to/graphs --threshold 10
"""

import sys
import os
import argparse
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import generateGraphs


def generate_graphs_from_directory(osdb_dir, grouping_period='3min', 
                                   output_dir=None, threshold=5, debug=False):
    """
    Generate summary graphs from all JSON files in a directory.
    
    Parameters:
    osdb_dir - directory containing JSON files
    grouping_period - grouping period string (default: '3min')
    output_dir - output directory for graphs (default: osdb_dir/output)
    threshold - minimum events per user for individual display
    debug - print debug information
    
    Returns:
    True if successful, False otherwise
    """
    print(f"Generating graphs from directory: {osdb_dir}")
    print(f"Grouping period: {grouping_period}")
    
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(osdb_dir, 'output')
    
    print(f"Output directory: {output_dir}")
    print("")
    
    # List of JSON files to process
    json_files = []
    json_file_patterns = [
        f"osdb_{grouping_period}_allSeizures.json",
        f"osdb_{grouping_period}_tcSeizures.json",
        f"osdb_{grouping_period}_fallEvents.json",
        f"osdb_{grouping_period}_falseAlarms.json",
        f"osdb_{grouping_period}_ndaEvents.json",
    ]
    
    for json_fname in json_file_patterns:
        json_path = os.path.join(osdb_dir, json_fname)
        if os.path.exists(json_path):
            json_files.append(json_path)
            if debug:
                print(f"Found: {json_fname}")
    
    if not json_files:
        print("Error: No JSON files found in directory")
        return False
    
    print(f"Processing {len(json_files)} JSON file(s)...")
    print("")
    
    # Generate graphs
    success = generateGraphs.generate_all_graphs(
        json_files,
        output_dir,
        threshold=threshold,
        debug=debug
    )
    
    if success:
        print("")
        print(f"✓ Graphs saved to: {output_dir}")
    else:
        print("")
        print("✗ Graph generation failed")
    
    return success


def generate_graphs_from_files(json_files, output_dir='output', 
                               threshold=5, debug=False):
    """
    Generate summary graphs from specific JSON files.
    
    Parameters:
    json_files - list of JSON file paths
    output_dir - output directory for graphs
    threshold - minimum events per user for individual display
    debug - print debug information
    
    Returns:
    True if successful, False otherwise
    """
    print(f"Generating graphs from {len(json_files)} file(s)...")
    print(f"Output directory: {output_dir}")
    print("")
    
    # Verify files exist
    valid_files = []
    for json_file in json_files:
        if os.path.exists(json_file):
            valid_files.append(json_file)
            if debug:
                print(f"Found: {json_file}")
        else:
            print(f"Warning: File not found: {json_file}")
    
    if not valid_files:
        print("Error: No valid JSON files to process")
        return False
    
    print("")
    
    # Generate graphs
    success = generateGraphs.generate_all_graphs(
        valid_files,
        output_dir,
        threshold=threshold,
        debug=debug
    )
    
    if success:
        print("")
        print(f"✓ Graphs saved to: {output_dir}")
    else:
        print("")
        print("✗ Graph generation failed")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Generate summary graphs from OSDB JSON event files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate graphs from all JSON files in directory
  %(prog)s --osdb-dir /path/to/osdb
  
  # Generate graphs from specific files
  %(prog)s --json-files file1.json file2.json
  
  # Customize output and threshold
  %(prog)s --osdb-dir /path/to/osdb --output /path/to/graphs --threshold 10
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--osdb-dir',
                       help='Directory containing OSDB JSON files')
    group.add_argument('--json-files', nargs='+',
                       help='Specific JSON files to process')
    
    parser.add_argument('--output', default=None,
                        help='Output directory for graphs (default: osdb-dir/output or ./output)')
    parser.add_argument('--grouping-period', default='3min',
                        help='Grouping period for directory processing (default: 3min)')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Minimum events per user for individual display (default: 5)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    if args.osdb_dir:
        # Process entire directory
        success = generate_graphs_from_directory(
            args.osdb_dir,
            grouping_period=args.grouping_period,
            output_dir=args.output,
            threshold=args.threshold,
            debug=args.debug
        )
    else:
        # Process specific files
        output_dir = args.output if args.output else 'output'
        success = generate_graphs_from_files(
            args.json_files,
            output_dir=output_dir,
            threshold=args.threshold,
            debug=args.debug
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
