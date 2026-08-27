#!/usr/bin/env python3
"""
fill_missing_metadata.py

Fill in missing metadata fields in the OpenSeizureDatabase.

Events downloaded from the remote server should have metadata fields like
dataSourceName, phoneAppVersion, etc. However, if the wrapper wasn't extracting
these fields when they were first downloaded, the database may have NULL values
for these fields.

This script:
1. Loads the current source JSON files
2. Extracts metadata from those files
3. Updates the database to fill in any missing/NULL metadata fields
4. Reports statistics on what was updated

Usage:
    python3 fill_missing_metadata.py --db /path/to/osdb_working.db \
        --json-dir /home/graham/osd/osdb [--dry-run]

Options:
    --db DATABASE_PATH      Path to SQLite database (required)
    --json-dir JSON_DIR     Directory containing source JSON files (required)
    --dry-run              Show what would be updated without making changes
"""

import sys
import os
import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from metadata_extraction import extract_metadata_from_events, print_extraction_summary


def load_events_from_json_files(json_dir: str) -> List[Dict[str, Any]]:
    """
    Load all osdb_*.json files from directory.
    
    Args:
        json_dir: Directory containing JSON files
        
    Returns:
        List of event dictionaries
    """
    events = []
    json_files = list(Path(json_dir).glob('osdb_*.json'))
    
    if not json_files:
        print(f"No osdb_*.json files found in {json_dir}")
        return events
    
    print(f"Loading events from {len(json_files)} JSON files...")
    
    for json_file in sorted(json_files):
        print(f"  Reading {json_file.name}...", end=' ')
        try:
            with open(json_file, 'r') as f:
                file_events = json.load(f)
            
            if isinstance(file_events, list):
                events.extend(file_events)
                print(f"✓ ({len(file_events)} events)")
            else:
                print(f"✗ (not a list)")
        except Exception as e:
            print(f"✗ ({e})")
    
    print(f"Total events loaded: {len(events)}")
    return events


def get_db_events_needing_metadata(db_path: str) -> List[Dict[str, Any]]:
    """
    Get events from database that have missing/NULL metadata fields.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        List of event dictionaries with just ID and any available metadata
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get events that have NULL values in any metadata field
    cursor.execute("""
        SELECT id, dataSourceName, phoneAppVersion, watchSdVersion, 
               watchFwVersion, watchSdName, watchPartNo, watchSerialNo
        FROM events
        WHERE dataSourceName IS NULL 
           OR phoneAppVersion IS NULL
           OR watchSdVersion IS NULL
           OR watchFwVersion IS NULL
           OR watchSdName IS NULL
           OR watchPartNo IS NULL
           OR watchSerialNo IS NULL
    """)
    
    events = []
    for row in cursor.fetchall():
        events.append(dict(row))
    
    conn.close()
    return events


def update_event_metadata_in_db(db_path: str, event_id: str, 
                                 metadata: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    Update metadata fields for a single event in the database.
    
    Args:
        db_path: Path to SQLite database
        event_id: Event ID to update
        metadata: Dictionary with metadata fields to update
        dry_run: If True, don't actually update
        
    Returns:
        True if updated, False otherwise
    """
    if dry_run:
        return True
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        fields_to_update = []
        values = []
        
        for field in ['dataSourceName', 'phoneAppVersion', 'watchSdVersion', 
                      'watchFwVersion', 'watchSdName', 'watchPartNo', 'watchSerialNo']:
            if field in metadata and metadata[field] is not None:
                fields_to_update.append(f"{field} = ?")
                values.append(metadata[field])
        
        if fields_to_update:
            values.append(event_id)
            sql = f"UPDATE events SET {', '.join(fields_to_update)} WHERE id = ?"
            cursor.execute(sql, values)
            conn.commit()
            return True
    except Exception as e:
        print(f"Error updating event {event_id}: {e}")
    finally:
        conn.close()
    
    return False


def fill_missing_metadata(db_path: str, json_dir: str, dry_run: bool = False) -> Dict[str, Any]:
    """
    Fill in missing metadata fields in the database from JSON source files.
    
    Args:
        db_path: Path to SQLite database
        json_dir: Directory containing source JSON files
        dry_run: If True, show what would be updated without making changes
        
    Returns:
        Statistics dictionary
    """
    
    # Load events from JSON files
    json_events = load_events_from_json_files(json_dir)
    
    if not json_events:
        print("No events loaded from JSON files")
        return {'total_events': 0, 'events_needing_metadata': 0, 'events_updated': 0}
    
    # Extract metadata from JSON events
    print("\nExtracting metadata from JSON events...")
    json_events, _ = extract_metadata_from_events(json_events, debug=False)
    
    # Create dict of JSON events by ID for fast lookup
    json_events_by_id = {str(e.get('id')): e for e in json_events}
    print(f"Created index of {len(json_events_by_id)} events from JSON")
    
    # Get DB events needing metadata
    print(f"\nScanning database for events with missing metadata...")
    db_events_needing = get_db_events_needing_metadata(db_path)
    print(f"Found {len(db_events_needing)} events with missing metadata")
    
    # For each DB event needing metadata, look up in JSON and update
    stats = {
        'total_events': len(json_events),
        'events_needing_metadata': len(db_events_needing),
        'events_updated': 0,
        'fields_updated': {},
        'updated_details': []
    }
    
    if len(db_events_needing) == 0:
        print("\n✓ No events needing metadata updates")
        return stats
    
    metadata_fields = ['dataSourceName', 'phoneAppVersion', 'watchSdVersion',
                       'watchFwVersion', 'watchSdName', 'watchPartNo', 'watchSerialNo']
    for field in metadata_fields:
        stats['fields_updated'][field] = 0
    
    print(f"\n{'Updating events with metadata...' if not dry_run else 'Would update events with metadata...'}")
    
    for db_event in db_events_needing:
        event_id = str(db_event['id'])
        
        # Look up in JSON events
        if event_id not in json_events_by_id:
            continue
        
        json_event = json_events_by_id[event_id]
        
        # Extract metadata fields that need updating
        updated_metadata = {}
        has_update = False
        
        for field in metadata_fields:
            # Update if DB has NULL and JSON has a value
            if db_event.get(field) is None and field in json_event and json_event[field] is not None:
                updated_metadata[field] = json_event[field]
                stats['fields_updated'][field] += 1
                has_update = True
        
        if has_update:
            update_event_metadata_in_db(db_path, event_id, updated_metadata, dry_run=dry_run)
            stats['events_updated'] += 1
            stats['updated_details'].append({
                'event_id': event_id,
                'fields': list(updated_metadata.keys()),
                'values': updated_metadata
            })
    
    return stats


def print_fill_summary(stats: Dict[str, Any], dry_run: bool = False) -> None:
    """
    Print summary of metadata fill operation.
    
    Args:
        stats: Statistics dictionary from fill_missing_metadata
        dry_run: If True, indicates this was a dry run
    """
    print("\n" + "="*70)
    print(f"Metadata Fill Summary {'(DRY RUN)' if dry_run else ''}")
    print("="*70)
    
    print(f"\nCounts:")
    print(f"  Total events in JSON: {stats['total_events']}")
    print(f"  Events needing metadata: {stats['events_needing_metadata']}")
    print(f"  Events updated: {stats['events_updated']}")
    
    if stats['events_updated'] > 0:
        print(f"\nFields Updated:")
        for field, count in stats['fields_updated'].items():
            if count > 0:
                print(f"  {field}: {count} events")
        
        if len(stats['updated_details']) <= 10:
            print(f"\nUpdated Events (showing first {len(stats['updated_details'])}):")
            for detail in stats['updated_details'][:10]:
                print(f"  Event {detail['event_id']}: {', '.join(detail['fields'])}")
                for field, value in detail['values'].items():
                    print(f"    {field} = {value}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fill in missing metadata fields in OpenSeizureDatabase"
    )
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--json-dir', required=True, help='Directory containing source JSON files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be updated without making changes')
    
    args = parser.parse_args()
    
    # Verify paths exist
    if not os.path.exists(args.db):
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)
    
    if not os.path.isdir(args.json_dir):
        print(f"Error: JSON directory not found: {args.json_dir}")
        sys.exit(1)
    
    # Run the fill operation
    stats = fill_missing_metadata(args.db, args.json_dir, dry_run=args.dry_run)
    
    # Print summary
    print_fill_summary(stats, dry_run=args.dry_run)
    
    # Exit with status based on updates
    sys.exit(0 if stats['events_updated'] > 0 or stats['events_needing_metadata'] == 0 else 1)


if __name__ == '__main__':
    main()
