#!/usr/bin/env python3
"""
manage_events.py - Event Management CLI Tool

Provides command-line interface for editing and deleting events in the SQLite database.

Usage:
    # Edit event metadata
    python3 manage_events.py edit --db osdb.db --event-id 12345 --field type --value "False Alarm"
    python3 manage_events.py edit --db osdb.db --event-id 12345 --field desc --value "Updated description"
    python3 manage_events.py edit --db osdb.db --event-id 12345 --field seizureTimes --value "[10.5, 25.3]"
    
    # Delete events
    python3 manage_events.py delete --db osdb.db --event-id 12345
    python3 manage_events.py delete --db osdb.db --event-ids 12345,12346,12347
    
    # View event details
    python3 manage_events.py show --db osdb.db --event-id 12345
    
    # List events
    python3 manage_events.py list --db osdb.db --type Seizure --limit 10
    
    # Database statistics
    python3 manage_events.py stats --db osdb.db
"""

import sys
import os
import argparse
import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database_utils import (
    backup_database, safe_delete_events, update_event_metadata,
    validate_database, get_database_stats
)


def show_event(db_path: str, event_id: int) -> None:
    """Display detailed information about an event."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get event
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        event = cursor.fetchone()
        
        if not event:
            print(f"✗ Event {event_id} not found")
            return
        
        # Get datapoints count
        cursor.execute("SELECT COUNT(*) FROM datapoints WHERE event_id = ?", (event_id,))
        dp_count = cursor.fetchone()[0]
        
        print(f"\n{'='*60}")
        print(f"Event ID: {event['id']}")
        print(f"{'='*60}")
        print(f"User ID:          {event['userId']}")
        print(f"Type:             {event['type']}")
        print(f"SubType:          {event['subType']}")
        print(f"Description:      {event['desc'] or '(none)'}")
        print(f"Data Time:        {event['dataTime']}")
        print(f"Data Time End:    {event['dataTimeEnd'] or '(none)'}")
        print(f"Duration:         {event['duration_seconds'] or '(none)'} seconds")
        print(f"Alarm State:      {event['osdAlarmState']}")
        print(f"Alarm Phrase:     {event['alarmPhrase'] or '(none)'}")
        print(f"\nData Source:      {event['dataSourceName'] or '(none)'}")
        print(f"Phone App Ver:    {event['phoneAppVersion'] or '(none)'}")
        print(f"Watch SD Ver:     {event['watchSdVersion'] or '(none)'}")
        print(f"Watch FW Ver:     {event['watchFwVersion'] or '(none)'}")
        print(f"Watch Name:       {event['watchSdName'] or '(none)'}")
        print(f"Battery:          {event['batteryPc'] or '(none)'}%")
        
        print(f"\nData Availability:")
        print(f"  HR Data:        {'Yes' if event['hasHrData'] else 'No'}")
        print(f"  O2Sat Data:     {'Yes' if event['hasO2SatData'] else 'No'}")
        print(f"  3D Accel Data:  {'Yes' if event['has3dData'] else 'No'}")
        
        print(f"\nDatapoints:       {dp_count} (stored: {event['datapoint_count']})")
        
        if event['seizureTimes']:
            seizure_times = json.loads(event['seizureTimes'])
            print(f"Seizure Times:    {seizure_times[0]:.1f}s - {seizure_times[1]:.1f}s")
        
        if event['merged_from_events']:
            merged = json.loads(event['merged_from_events'])
            print(f"Merged From:      {len(merged)} events: {merged}")
        
        if event['metadata']:
            metadata = json.loads(event['metadata'])
            if metadata:
                print(f"\nExtra Metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
        
        print(f"\nLast Modified:    {event['last_modified']}")
        print(f"{'='*60}\n")
        
    finally:
        conn.close()


def list_events(db_path: str, event_type: Optional[str] = None, 
                user_id: Optional[int] = None, limit: int = 20) -> None:
    """List events with optional filtering."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        query = "SELECT id, userId, dataTime, type, subType, desc, datapoint_count FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND userId = ?"
            params.append(user_id)
        
        query += " ORDER BY dataTime DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        events = cursor.fetchall()
        
        if not events:
            print("No events found")
            return
        
        print(f"\n{'ID':<8} {'User':<6} {'Date':<20} {'Type':<15} {'SubType':<15} {'DPs':<5} Description")
        print("-" * 120)
        
        for event in events:
            desc = (event['desc'] or '')[:40]
            print(f"{event['id']:<8} {event['userId']:<6} {event['dataTime']:<20} "
                  f"{event['type']:<15} {event['subType'] or '':<15} "
                  f"{event['datapoint_count']:<5} {desc}")
        
        print(f"\nShowing {len(events)} event(s)")
        
    finally:
        conn.close()


def edit_event(db_path: str, event_id: int, field: str, value: str, 
               no_backup: bool = False) -> None:
    """Edit a single field of an event."""
    # Parse value for special fields
    if field == 'seizureTimes':
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            print(f"✗ Invalid JSON for seizureTimes: {value}")
            print("  Expected format: \"[start_seconds, end_seconds]\"")
            return
    elif field == 'osdAlarmState':
        try:
            value = int(value)
        except ValueError:
            print(f"✗ Invalid integer for osdAlarmState: {value}")
            return
    elif field in ['batteryPc']:
        try:
            value = int(value) if value else None
        except ValueError:
            print(f"✗ Invalid integer for {field}: {value}")
            return
    
    success = update_event_metadata(db_path, event_id, field, value, 
                                    create_backup=not no_backup)
    
    if success:
        print(f"\n✓ Event {event_id} updated successfully")
        show_event(db_path, event_id)


def delete_events(db_path: str, event_ids: List[int], no_backup: bool = False,
                 force: bool = False) -> None:
    """Delete one or more events."""
    if not force:
        print(f"\n⚠️  WARNING: About to delete {len(event_ids)} event(s)")
        print(f"   Event IDs: {event_ids}")
        response = input("\nAre you sure? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Deletion cancelled")
            return
    
    events_deleted, datapoints_deleted = safe_delete_events(
        db_path, event_ids, create_backup=not no_backup
    )
    
    print(f"\n✓ Deleted {events_deleted} event(s) and {datapoints_deleted} datapoint(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Event Management CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show event details')
    show_parser.add_argument('--db', required=True, help='Path to SQLite database')
    show_parser.add_argument('--event-id', required=True, help='Event ID to display')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List events')
    list_parser.add_argument('--db', required=True, help='Path to SQLite database')
    list_parser.add_argument('--type', help='Filter by event type')
    list_parser.add_argument('--user-id', type=int, help='Filter by user ID')
    list_parser.add_argument('--limit', type=int, default=20,
                            help='Maximum number of events to show (default: 20)')
    
    # Edit command
    edit_parser = subparsers.add_parser('edit', help='Edit event field')
    edit_parser.add_argument('--db', required=True, help='Path to SQLite database')
    edit_parser.add_argument('--no-backup', action='store_true',
                            help='Skip automatic backup before edit')
    edit_parser.add_argument('--event-id', required=True, help='Event ID to edit')
    edit_parser.add_argument('--field', required=True,
                            choices=['type', 'subType', 'desc', 'osdAlarmState',
                                   'dataTime', 'dataTimeEnd', 'alarmPhrase',
                                   'alarmRationale', 'seizureTimes', 'batteryPc'],
                            help='Field to update')
    edit_parser.add_argument('--value', required=True,
                            help='New value for the field')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete event(s)')
    delete_parser.add_argument('--db', required=True, help='Path to SQLite database')
    delete_parser.add_argument('--no-backup', action='store_true',
                              help='Skip automatic backup before delete')
    delete_group = delete_parser.add_mutually_exclusive_group(required=True)
    delete_group.add_argument('--event-id', help='Single event ID to delete')
    delete_group.add_argument('--event-ids', type=str,
                             help='Comma-separated list of event IDs to delete')
    delete_parser.add_argument('--force', action='store_true',
                              help='Skip confirmation prompt')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--db', required=True, help='Path to SQLite database')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate database integrity')
    validate_parser.add_argument('--db', required=True, help='Path to SQLite database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'show':
            show_event(args.db, args.event_id)
        
        elif args.command == 'list':
            list_events(args.db, args.type, args.user_id, args.limit)
        
        elif args.command == 'edit':
            edit_event(args.db, args.event_id, args.field, args.value, args.no_backup)
        
        elif args.command == 'delete':
            if args.event_id:
                event_ids = [args.event_id]
            else:
                event_ids = [int(x.strip()) for x in args.event_ids.split(',')]
            delete_events(args.db, event_ids, args.no_backup, args.force)
        
        elif args.command == 'stats':
            stats = get_database_stats(args.db)
            print(f"\nDatabase Statistics: {args.db}")
            print(f"{'='*60}")
            print(f"Schema Version:       {stats['schema_version']}")
            print(f"Total Events:         {stats['total_events']}")
            print(f"Total Datapoints:     {stats['total_datapoints']}")
            print(f"Avg Datapoints/Event: {stats['avg_datapoints_per_event']}")
            print(f"Date Range:           {stats['date_range'][0]} to {stats['date_range'][1]}")
            print(f"Database Size:        {stats['database_size_mb']} MB")
            print(f"\nEvents by Type:")
            for event_type, count in stats['events_by_type'].items():
                print(f"  {event_type:<20} {count:>6}")
            print(f"{'='*60}\n")
        
        elif args.command == 'validate':
            is_valid, issues = validate_database(args.db)
            print(f"\nDatabase: {args.db}")
            print(f"Valid: {'✓ YES' if is_valid else '✗ NO'}\n")
            if issues:
                print("Issues found:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print("✓ No issues found!")
            print()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
