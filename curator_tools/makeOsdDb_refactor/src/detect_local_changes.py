#!/usr/bin/env python3
"""
detect_local_changes.py

Detect and mark existing local changes in the database by comparing with source JSON files.

This script reads JSON export files and compares them with the current database to identify
which events have been locally edited (fields differ between JSON and database).

Locally-editable fields checked:
- type
- subType
- desc (description)
- seizureTimes

Usage:
    # Scan for changes and report (no modifications)
    python3 detect_local_changes.py --db /path/to/osdb_working.db \
        --json-dir /home/graham/osd/osdb --dry-run
    
    # Detect and mark changes in database
    python3 detect_local_changes.py --db /path/to/osdb_working.db \
        --json-dir /home/graham/osd/osdb
    
    # Process specific file only
    python3 detect_local_changes.py --db /path/to/osdb_working.db \
        --json-file /home/graham/osd/osdb/osdb_3min_allSeizures.json
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple
from datetime import datetime


def normalize_value(val: Any) -> Any:
    """Normalize values for comparison (handle JSON arrays, None, etc)."""
    if val is None:
        return None
    if isinstance(val, list):
        return tuple(val)  # Convert lists to tuples for comparison
    if isinstance(val, dict):
        return json.dumps(val, sort_keys=True, default=str)
    return val


def compare_events(json_event: Dict[str, Any], db_event: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Tuple[Any, Any]]]:
    """
    Compare JSON version with database version to detect local edits.
    
    Args:
        json_event: Event from JSON file
        db_event: Event from database
        
    Returns:
        Tuple of (set of edited field names, dict of {field: (json_value, db_value)})
    """
    # Locally-editable fields
    EDITABLE_FIELDS = {'type', 'subType', 'desc', 'seizureTimes'}
    
    edited_fields = set()
    differences = {}
    
    for field in EDITABLE_FIELDS:
        json_val = normalize_value(json_event.get(field))
        db_val = normalize_value(db_event.get(field))
        
        if json_val != db_val:
            edited_fields.add(field)
            differences[field] = (json_event.get(field), db_event.get(field))
    
    return edited_fields, differences


def load_json_events(json_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all events from JSON files in directory.
    
    Args:
        json_dir: Directory containing osdb_*.json files
        
    Returns:
        Dictionary mapping event ID -> event dict
    """
    events = {}
    json_path = Path(json_dir)
    
    # Look for all osdb_*.json files
    json_files = list(json_path.glob('osdb_*.json'))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return events
    
    print(f"Loading events from {len(json_files)} JSON files...")
    
    for json_file in sorted(json_files):
        print(f"  Reading {json_file.name}...", end='', flush=True)
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle wrapped format
            if isinstance(data, dict) and 'events' in data:
                file_events = data['events']
            else:
                file_events = data if isinstance(data, list) else []
            
            # Add to dictionary (later files can override earlier ones)
            for event in file_events:
                event_id = event.get('id')
                if event_id:
                    events[str(event_id)] = event
            
            print(f" ✓ ({len(file_events)} events)")
        except Exception as e:
            print(f" ✗ Error: {e}")
            continue
    
    print(f"Total unique events loaded from JSON: {len(events)}")
    return events


def load_json_file(json_file: str) -> Dict[str, Dict[str, Any]]:
    """Load events from a single JSON file."""
    events = {}
    
    print(f"Loading events from {json_file}...")
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle wrapped format
        if isinstance(data, dict) and 'events' in data:
            file_events = data['events']
        else:
            file_events = data if isinstance(data, list) else []
        
        for event in file_events:
            event_id = event.get('id')
            if event_id:
                events[str(event_id)] = event
        
        print(f"Loaded {len(events)} events from JSON file")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
    
    return events


def detect_local_changes(db_path: str, json_events: Dict[str, Dict[str, Any]], dry_run: bool = False) -> Dict[str, Any]:
    """
    Scan database and detect local changes by comparing with JSON.
    
    Args:
        db_path: Path to SQLite database
        json_events: Dictionary of events from JSON (event_id -> event dict)
        dry_run: If True, don't modify database
        
    Returns:
        Statistics dictionary
    """
    print(f"\nConnecting to database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get all events from database
        print("Loading events from database...")
        cursor.execute("SELECT * FROM events")
        db_events = {str(row['id']): dict(row) for row in cursor.fetchall()}
        print(f"Loaded {len(db_events)} events from database")
        
        # Compare and detect changes
        stats = {
            'total_events': len(db_events),
            'events_with_changes': 0,
            'events_in_json': 0,
            'events_not_in_json': 0,
            'changes_by_field': {'type': 0, 'subType': 0, 'desc': 0, 'seizureTimes': 0},
            'change_details': []
        }
        
        print("\nComparing database with JSON...")
        
        for event_id, db_event in db_events.items():
            if event_id in json_events:
                stats['events_in_json'] += 1
                json_event = json_events[event_id]
                
                edited_fields, differences = compare_events(json_event, db_event)
                
                if edited_fields:
                    stats['events_with_changes'] += 1
                    
                    # Count field changes
                    for field in edited_fields:
                        stats['changes_by_field'][field] += 1
                    
                    # Store details
                    stats['change_details'].append({
                        'event_id': event_id,
                        'edited_fields': list(edited_fields),
                        'differences': {
                            k: {'json': v[0], 'db': v[1]}
                            for k, v in differences.items()
                        }
                    })
            else:
                stats['events_not_in_json'] += 1
        
        # Print summary
        print(f"\n{'='*70}")
        print("CHANGE DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total events in database: {stats['total_events']}")
        print(f"Events in JSON files: {stats['events_in_json']}")
        print(f"Events NOT in JSON (local only): {stats['events_not_in_json']}")
        print(f"\nEvents with local changes: {stats['events_with_changes']}")
        
        if stats['events_with_changes'] > 0:
            print(f"\nChanges by field:")
            for field, count in stats['changes_by_field'].items():
                if count > 0:
                    print(f"  {field}: {count} events")
            
            print(f"\nFirst 10 events with changes:")
            print(f"{'-'*70}")
            for detail in stats['change_details'][:10]:
                event_id = detail['event_id']
                edited = ', '.join(detail['edited_fields'])
                print(f"\nEvent ID {event_id}: {edited}")
                for field, values in detail['differences'].items():
                    json_val = values['json']
                    db_val = values['db']
                    print(f"  {field}:")
                    print(f"    JSON: {json_val}")
                    print(f"    DB:   {db_val}")
            
            if len(stats['change_details']) > 10:
                print(f"\n... and {len(stats['change_details']) - 10} more events with changes")
        
        print(f"{'='*70}")
        
        # Update database if not dry run
        if not dry_run and stats['events_with_changes'] > 0:
            print(f"\nUpdating database to mark local changes...")
            updated = 0
            
            for detail in stats['change_details']:
                event_id = detail['event_id']
                edited_fields = detail['edited_fields']
                
                # Update local_edits and has_local_changes
                cursor.execute(
                    """UPDATE events 
                       SET local_edits = ?, has_local_changes = 1, last_modified = ?
                       WHERE id = ?""",
                    (json.dumps(edited_fields), datetime.now().isoformat(), event_id)
                )
                updated += 1
            
            conn.commit()
            print(f"✓ Updated {updated} events with local change tracking")
        elif dry_run:
            print(f"\n[DRY RUN] Would update {stats['events_with_changes']} events")
            print("Run without --dry-run to apply changes")
        
        conn.close()
        
        return stats
        
    except Exception as e:
        print(f"Error: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description='Detect and mark existing local changes in OSDB database'
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Path to SQLite database'
    )
    
    # Either directory or specific file
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--json-dir',
        help='Directory containing osdb_*.json files'
    )
    source_group.add_argument(
        '--json-file',
        help='Path to specific JSON file to compare'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    # Load JSON events
    if args.json_dir:
        json_events = load_json_events(args.json_dir)
    else:
        json_events = load_json_file(args.json_file)
    
    if not json_events:
        print("No JSON events loaded - cannot detect changes")
        return 1
    
    # Detect changes
    stats = detect_local_changes(args.db, json_events, dry_run=args.dry_run)
    
    return 0 if stats else 1


if __name__ == '__main__':
    exit(main())
