#!/usr/bin/env python3
"""
generate_index_from_db.py

Generate CSV index files directly from an OSDB SQLite database.
This module provides efficient index generation without creating intermediate JSON files.

Usage:
    from generate_index_from_db import generate_index_from_database
    
    count = generate_index_from_database(
        db_path='/path/to/osdb.db',
        output_dir='/path/to/output',
        grouping_period='3min',
        debug=False
    )
"""

import sys
import os
import json
import tempfile
from typing import List, Dict, Any

# Add parent directory to path for libosd import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from osdb_sqlite import OsdWorkingDb
import libosd.osdDbConnection


def generate_index_from_database(
    db_path: str,
    output_dir: str,
    grouping_period: str = '3min',
    debug: bool = False
) -> int:
    """
    Generate CSV index files directly from SQLite database.
    
    This function generates index files for each event category without
    persisting JSON files to disk (uses temporary files only).
    
    Parameters:
    -----------
    db_path : str
        Path to SQLite database
    output_dir : str
        Output directory for CSV index files
    grouping_period : str
        Grouping period string for filename (default: '3min')
    debug : bool
        Enable debug output
        
    Returns:
    --------
    int
        Number of index files successfully generated
    """
    try:
        if debug:
            print(f"Opening database: {db_path}")
        
        # Open database
        db = OsdWorkingDb(db_path, debug=debug)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Event categories to process
        categories = {
            'allSeizures': lambda e: e.get('type') == 'Seizure',
            'tcSeizures': lambda e: (
                e.get('type') == 'Seizure' and 
                ('tonic' in str(e.get('subType', '')).lower() or 
                 'clonic' in str(e.get('subType', '')).lower())
            ),
            'fallEvents': lambda e: e.get('type') == 'Fall',
            'falseAlarms': lambda e: e.get('type') == 'False Alarm',
            'ndaEvents': lambda e: e.get('type') == 'NDA',
        }
        
        generated_count = 0
        
        # Process each category
        for category, filter_func in categories.items():
            if debug:
                print(f"\nProcessing category: {category}")
            
            # Get all events and filter by category
            all_events = db.get_events(include_datapoints=True)
            category_events = [e for e in all_events if filter_func(e)]
            
            if not category_events:
                if debug:
                    print(f"  No events found for {category}, skipping...")
                continue
            
            if debug:
                print(f"  Found {len(category_events)} events")
            
            # Create temporary JSON file for libosd processing
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                json.dump(category_events, tmp_json, indent=2)
                tmp_json_path = tmp_json.name
            
            try:
                # Output CSV path
                csv_filename = f"osdb_{grouping_period}_{category}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                
                if debug:
                    print(f"  Generating CSV index: {csv_filename}")
                
                # Use libosd to generate CSV index
                osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=debug)
                osd.loadDbFile(tmp_json_path, useCacheDir=False)
                osd.saveIndexFile(csv_path, useCacheDir=False)
                
                print(f"✓ Generated: {csv_filename}")
                generated_count += 1
                
            except Exception as e:
                print(f"Error generating index for {category}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
            finally:
                # Clean up temporary JSON file
                if os.path.exists(tmp_json_path):
                    os.remove(tmp_json_path)
        
        # Close database
        db.close()
        
        print(f"\n✓ Generated {generated_count} index file(s) in: {output_dir}")
        return generated_count
        
    except Exception as e:
        print(f"Error generating indexes: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return 0


def main():
    """Command-line interface for index generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate CSV index files directly from OSDB SQLite database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate indexes from database
  %(prog)s --db /path/to/osdb.db --output /path/to/output
  
  # With custom grouping period
  %(prog)s --db /path/to/osdb.db --output /path/to/output --grouping-period 5min
        """
    )
    
    parser.add_argument('--db', required=True,
                        help='Path to SQLite database')
    parser.add_argument('--output', required=True,
                        help='Output directory for CSV index files')
    parser.add_argument('--grouping-period', default='3min',
                        help='Grouping period for filename (default: 3min)')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug information')
    
    args = parser.parse_args()
    
    count = generate_index_from_database(
        args.db,
        args.output,
        grouping_period=args.grouping_period,
        debug=args.debug
    )
    
    sys.exit(0 if count > 0 else 1)


if __name__ == '__main__':
    main()
