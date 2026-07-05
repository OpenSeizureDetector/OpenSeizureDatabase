#!/usr/bin/env python3
"""
publish_osdb.py - Unified Publication Script

Combines database operations and multi-format publication into a single workflow.

This script demonstrates the complete Phase 4 & 5 workflow:
1. Import JSON to SQLite (Phase 4)
2. Query/filter events from database
3. Export to multiple formats (Phase 5)

Usage:
    # Import and publish in all formats
    python3 publish_osdb.py --input osdb_3min_allSeizures.json --all-formats
    
    # Use existing database, export Seizures only
    python3 publish_osdb.py --db osdb.db --type Seizure --formats parquet csv
    
    # Compare original JSON with database roundtrip
    python3 publish_osdb.py --input original.json --verify-consistency
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from osdb_sqlite import OsdWorkingDb
from osdb_publication import OsdbPublisher


def import_and_publish(
    input_json: str,
    output_prefix: str,
    output_dir: str,
    formats: list,
    use_database: bool = True,
    event_type: str = None,
    verify: bool = False
):
    """
    Complete workflow: Import JSON → Database → Export multiple formats.
    
    Args:
        input_json: Input JSON file path
        output_prefix: Prefix for output files
        output_dir: Output directory
        formats: List of output formats
        use_database: Use SQLite database (Phase 4) or direct conversion
        event_type: Filter by event type
        verify: Verify consistency between input and output
    """
    print(f"\n{'='*70}")
    print("OSDB Unified Publication Pipeline")
    print(f"{'='*70}")
    print(f"Input: {input_json}")
    print(f"Output prefix: {output_prefix}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Using database: {use_database}")
    
    # Load original data
    print(f"\n{'='*70}")
    print("Step 1: Loading Original Data")
    print(f"{'='*70}")
    
    with open(input_json, 'r') as f:
        original_data = json.load(f)
    
    if isinstance(original_data, dict) and 'events' in original_data:
        original_events = original_data['events']
    else:
        original_events = original_data
    
    print(f"Loaded {len(original_events)} events from {input_json}")
    
    if use_database:
        # Phase 4: Use SQLite database
        print(f"\n{'='*70}")
        print("Step 2: Importing to SQLite Database (Phase 4)")
        print(f"{'='*70}")
        
        db_path = os.path.join(output_dir, 'temp_working.db')
        db = OsdWorkingDb(db_path, debug=True)
        
        try:
            # Import
            db.import_from_json(input_json, clear_existing=True)
            
            # Show statistics
            stats = db.get_statistics()
            print(f"\nDatabase Statistics:")
            print(f"  Events: {stats['total_events']:,}")
            print(f"  Datapoints: {stats['total_datapoints']:,}")
            print(f"  Users: {stats['unique_users']}")
            print(f"  Types: {stats['unique_types']}")
            print(f"  Size: {stats['database_size_bytes'] / (1024*1024):.2f} MB")
            
            # Export from database
            print(f"\n{'='*70}")
            print("Step 3: Exporting from Database")
            print(f"{'='*70}")
            
            events = db.get_events(event_type=event_type)
            print(f"Retrieved {len(events)} events from database")
            
        finally:
            db.close()
            # Clean up temporary database
            if os.path.exists(db_path):
                os.remove(db_path)
    else:
        # Direct conversion (no database)
        print(f"\n{'='*70}")
        print("Step 2: Direct Conversion (No Database)")
        print(f"{'='*70}")
        
        events = original_events
        if event_type:
            events = [e for e in events if e.get('type') == event_type]
            print(f"Filtered to {len(events)} events of type '{event_type}'")
    
    # Phase 5: Publish in multiple formats
    print(f"\n{'='*70}")
    print("Step 4: Publishing in Multiple Formats (Phase 5)")
    print(f"{'='*70}")
    
    publisher = OsdbPublisher(debug=True)
    results = publisher.publish_all_formats(
        events,
        output_prefix,
        formats=formats,
        output_dir=output_dir
    )
    
    # Verify consistency
    if verify and use_database:
        print(f"\n{'='*70}")
        print("Step 5: Verifying Consistency")
        print(f"{'='*70}")
        
        verify_consistency(original_events, events)
    
    return results


def verify_consistency(original_events: list, exported_events: list):
    """
    Verify that database import/export preserves data integrity.
    
    Args:
        original_events: Original events from JSON
        exported_events: Events exported from database
    """
    print("Checking data consistency...")
    
    issues = []
    
    # Check event count
    if len(original_events) != len(exported_events):
        issues.append(f"Event count mismatch: {len(original_events)} → {len(exported_events)}")
    
    # Create lookup maps
    original_map = {e['id']: e for e in original_events}
    exported_map = {e['id']: e for e in exported_events}
    
    # Check each event
    for event_id in original_map:
        if event_id not in exported_map:
            issues.append(f"Event {event_id} missing from export")
            continue
        
        orig = original_map[event_id]
        exp = exported_map[event_id]
        
        # Check datapoint count
        orig_dp_count = len(orig.get('datapoints', []))
        exp_dp_count = len(exp.get('datapoints', []))
        if orig_dp_count != exp_dp_count:
            issues.append(f"Event {event_id}: datapoint count mismatch ({orig_dp_count} → {exp_dp_count})")
        
        # Check key fields
        for field in ['userId', 'dataTime', 'type', 'subType']:
            if orig.get(field) != exp.get(field):
                issues.append(f"Event {event_id}: {field} mismatch ({orig.get(field)} → {exp.get(field)})")
    
    # Report
    if issues:
        print(f"❌ Found {len(issues)} consistency issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✓ All consistency checks passed!")
        print("  - Event count: ✓")
        print("  - Event IDs: ✓")
        print("  - Datapoint counts: ✓")
        print("  - Metadata fields: ✓")


def main():
    parser = argparse.ArgumentParser(
        description='Unified OSDB publication pipeline (Phases 4 & 5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import JSON and publish all formats
  python3 publish_osdb.py --input osdb_3min_allSeizures.json --all-formats
  
  # Publish only Parquet and CSV
  python3 publish_osdb.py --input events.json --formats parquet csv
  
  # Filter by event type
  python3 publish_osdb.py --input events.json --type Seizure --all-formats
  
  # Verify database consistency
  python3 publish_osdb.py --input events.json --verify-consistency --all-formats
  
  # Direct conversion without database
  python3 publish_osdb.py --input events.json --no-database --formats parquet
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output-prefix', '-p', help='Output file prefix (default: input basename)')
    parser.add_argument('--output-dir', '-o', default='output', help='Output directory')
    parser.add_argument('--formats', nargs='+',
                       choices=['json', 'json.gz', 'parquet', 'csv'],
                       help='Output formats')
    parser.add_argument('--all-formats', action='store_true', help='Publish in all formats')
    parser.add_argument('--type', help='Filter by event type')
    parser.add_argument('--no-database', action='store_true', help='Skip database (direct conversion)')
    parser.add_argument('--verify-consistency', action='store_true', help='Verify data consistency')
    
    args = parser.parse_args()
    
    # Determine formats
    if args.all_formats:
        formats = ['json', 'json.gz', 'parquet', 'csv']
    elif args.formats:
        formats = args.formats
    else:
        formats = ['json']
    
    # Determine output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = Path(args.input).stem
    
    # Run pipeline
    try:
        results = import_and_publish(
            input_json=args.input,
            output_prefix=output_prefix,
            output_dir=args.output_dir,
            formats=formats,
            use_database=not args.no_database,
            event_type=args.type,
            verify=args.verify_consistency
        )
        
        print(f"\n{'='*70}")
        print("✓ Publication Complete!")
        print(f"{'='*70}")
        print(f"Output directory: {args.output_dir}")
        print(f"Files created: {len(results)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
