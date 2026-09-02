#!/usr/bin/env python3
"""
generate_graphs_from_db.py

Generate summary graphs directly from an OSDB SQLite database.
This module provides efficient graph generation without creating intermediate JSON files.

Usage:
    from generate_graphs_from_db import generate_graphs_from_database
    
    success = generate_graphs_from_database(
        db_path='/path/to/osdb.db',
        output_dir='/path/to/output',
        threshold=5,
        debug=False
    )
"""

import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for generateGraphs import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from osdb_sqlite import OsdWorkingDb
import generateGraphs


def generate_graphs_from_database(
    db_path: str,
    output_dir: str,
    threshold: int = 5,
    debug: bool = False
) -> bool:
    """
    Generate summary graphs directly from SQLite database.
    
    This function efficiently generates graphs without creating intermediate JSON files.
    It queries the database directly and passes events to the graphing functions.
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    output_dir : str
        Output directory for graphs
    threshold : int
        Minimum events per user for individual display (default: 5)
    debug : bool
        Enable debug output
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        if debug:
            print(f"Opening database: {db_path}")
        
        # Open database
        db = OsdWorkingDb(db_path, debug=debug)
        
        # Get all events (we don't need datapoints for graphs, just metadata)
        if debug:
            print("Loading events from database...")
        all_events = db.get_events(include_datapoints=False)
        
        if not all_events:
            print("Error: No events found in database")
            return False
        
        if debug:
            print(f"Loaded {len(all_events)} events")
        
        # Categorize events
        if debug:
            print("Categorizing events...")
        categorized = generateGraphs.categorize_events(all_events)
        
        # Generate summary statistics
        if debug:
            print("Generating summary statistics...")
        stats = generateGraphs.create_summary_stats(categorized)
        print(f"  Seizures: {stats['total_seizures']}")
        print(f"  False Alarms: {stats['total_false_alarms']}")
        print(f"  NDA Events: {stats['total_nda']}")
        print(f"  Total Events: {stats['total_events']}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate graphs
        print("\nGenerating graphs...")
        print("  - Summary statistics chart...")
        generateGraphs.create_summary_stats_chart(stats, output_dir)
        
        print("  - Events by user chart...")
        generateGraphs.create_events_by_user_chart(
            categorized['seizures'],
            output_dir,
            threshold=threshold,
            debug=debug
        )
        
        print("  - Cumulative seizures per month...")
        generateGraphs.create_cumulative_seizures_per_month(
            categorized['seizures'],
            output_dir,
            threshold=threshold,
            debug=debug
        )
        
        # Close database
        db.close()
        
        print(f"\n✓ Graphs saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error generating graphs: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Command-line interface for graph generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate summary graphs directly from OSDB SQLite database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate graphs from database
  %(prog)s --db /path/to/osdb.db --output /path/to/graphs
  
  # With custom threshold and debug output
  %(prog)s --db /path/to/osdb.db --output /path/to/graphs --threshold 10 --debug
        """
    )
    
    parser.add_argument('--db', required=True,
                        help='Path to SQLite database')
    parser.add_argument('--output', required=True,
                        help='Output directory for graphs')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Minimum events per user for individual display (default: 5)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    success = generate_graphs_from_database(
        args.db,
        args.output,
        threshold=args.threshold,
        debug=args.debug
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
