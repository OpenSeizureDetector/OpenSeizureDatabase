#!/usr/bin/env python3
"""
database_utils.py - Utility Functions for Database Management

Provides safe database operations including:
- Auto-backup before destructive operations
- Safe deletion with orphan datapoint cleanup
- Database validation and integrity checks
- Schema migration support

Usage:
    from database_utils import backup_database, safe_delete_events
    
    # Backup before delete
    backup_path = backup_database('osdb_working.db')
    
    # Safe delete (removes orphan datapoints)
    safe_delete_events('osdb_working.db', [event_id1, event_id2])
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import json


def backup_database(db_path: str, backup_dir: Optional[str] = None) -> str:
    """
    Create a timestamped backup of the database.
    
    Args:
        db_path: Path to database file
        backup_dir: Optional backup directory (defaults to same dir as database)
        
    Returns:
        Path to backup file
        
    Example:
        backup_path = backup_database('osdb.db')
        # Creates: osdb.db.backup.20260720_143025
    """
    db_path = Path(db_path)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"{db_path.name}.backup.{timestamp}"
    
    if backup_dir:
        backup_path = Path(backup_dir) / backup_filename
        backup_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        backup_path = db_path.parent / backup_filename
    
    # Copy database file
    shutil.copy2(db_path, backup_path)
    
    print(f"✓ Backup created: {backup_path}")
    return str(backup_path)


def get_schema_version(db_path: str) -> int:
    """
    Get the current schema version from the database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Schema version number (0 if schema_info table doesn't exist)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT MAX(version) FROM schema_info")
        result = cursor.fetchone()
        version = result[0] if result and result[0] is not None else 0
    except sqlite3.OperationalError:
        # schema_info table doesn't exist (old database)
        version = 0
    finally:
        conn.close()
    
    return version


def safe_delete_events(db_path: str, event_ids: List[int], create_backup: bool = True) -> Tuple[int, int]:
    """
    Safely delete events and their orphaned datapoints.
    
    This function:
    1. Creates a backup (if requested)
    2. Deletes specified events
    3. Removes datapoints that are ONLY associated with deleted events
    4. Preserves datapoints that are also referenced by other events (if any)
    
    Args:
        db_path: Path to database file
        event_ids: List of event IDs to delete
        create_backup: If True, create backup before deletion
        
    Returns:
        Tuple of (events_deleted, datapoints_deleted)
    """
    if create_backup:
        backup_database(db_path)
    
    conn = sqlite3.connect(db_path)
    # CRITICAL: Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    try:
        # Start transaction
        cursor.execute("BEGIN TRANSACTION")
        
        # Count events to be deleted
        placeholders = ','.join('?' * len(event_ids))
        cursor.execute(f"SELECT COUNT(*) FROM events WHERE id IN ({placeholders})", event_ids)
        events_count = cursor.fetchone()[0]
        
        # Count datapoints that will be deleted (only those exclusively linked to these events)
        # Note: With ON DELETE CASCADE, this happens automatically, but we count for reporting
        cursor.execute(f"""
            SELECT COUNT(*) FROM datapoints 
            WHERE event_id IN ({placeholders})
        """, event_ids)
        datapoints_count = cursor.fetchone()[0]
        
        # Delete events (CASCADE will delete associated datapoints)
        cursor.execute(f"DELETE FROM events WHERE id IN ({placeholders})", event_ids)
        
        # Commit transaction
        conn.commit()
        
        print(f"✓ Deleted {events_count} event(s) and {datapoints_count} datapoint(s)")
        
        return events_count, datapoints_count
        
    except Exception as e:
        # Rollback on error
        conn.rollback()
        print(f"✗ Error during deletion: {e}")
        raise
    finally:
        conn.close()


def update_event_metadata(db_path: str, event_id: int, field: str, value: any, 
                         create_backup: bool = True) -> bool:
    """
    Update a single field in an event record.
    
    Args:
        db_path: Path to database file
        event_id: Event ID to update
        field: Field name to update
        value: New value
        create_backup: If True, create backup before update
        
    Returns:
        True if update succeeded, False otherwise
    """
    if create_backup:
        backup_database(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    
    # Allowed fields for update (prevent SQL injection)
    allowed_fields = {
        'type', 'subType', 'desc', 'osdAlarmState', 'dataTime', 'dataTimeEnd',
        'alarmPhrase', 'alarmRationale', 'seizureTimes', 'batteryPc'
    }
    
    if field not in allowed_fields:
        print(f"✗ Field '{field}' is not editable")
        return False
    
    try:
        # Convert lists to JSON strings
        if field in ['seizureTimes'] and isinstance(value, list):
            value = json.dumps(value)
        
        # Update last_modified timestamp
        query = f"UPDATE events SET {field} = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?"
        cursor.execute(query, (value, event_id))
        
        if cursor.rowcount == 0:
            print(f"✗ Event {event_id} not found")
            return False
        
        conn.commit()
        print(f"✓ Updated event {event_id}: {field} = {value}")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"✗ Error updating event: {e}")
        return False
    finally:
        conn.close()


def validate_database(db_path: str) -> Tuple[bool, List[str]]:
    """
    Validate database integrity.
    
    Checks:
    - Schema version
    - Foreign key integrity
    - Orphaned datapoints
    - Missing required fields
    
    Args:
        db_path: Path to database file
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    issues = []
    
    try:
        # Check schema version
        version = get_schema_version(db_path)
        if version == 0:
            issues.append("Schema version table not found (old database format)")
        
        # Check foreign key integrity
        cursor.execute("PRAGMA foreign_key_check")
        fk_violations = cursor.fetchall()
        if fk_violations:
            issues.append(f"Foreign key violations found: {len(fk_violations)}")
            for violation in fk_violations[:5]:  # Show first 5
                issues.append(f"  - {violation}")
        
        # Check for orphaned datapoints (shouldn't happen with CASCADE but check anyway)
        cursor.execute("""
            SELECT COUNT(*) FROM datapoints 
            WHERE event_id NOT IN (SELECT id FROM events)
        """)
        orphaned = cursor.fetchone()[0]
        if orphaned > 0:
            issues.append(f"Found {orphaned} orphaned datapoint(s)")
        
        # Check for events without datapoints (warning, not error)
        cursor.execute("""
            SELECT COUNT(*) FROM events 
            WHERE datapoint_count = 0 OR id NOT IN (SELECT DISTINCT event_id FROM datapoints)
        """)
        empty_events = cursor.fetchone()[0]
        if empty_events > 0:
            issues.append(f"Warning: {empty_events} event(s) have no datapoints")
        
        # Check for events with missing required fields
        cursor.execute("""
            SELECT COUNT(*) FROM events 
            WHERE userId IS NULL OR dataTime IS NULL OR type IS NULL
        """)
        incomplete_events = cursor.fetchone()[0]
        if incomplete_events > 0:
            issues.append(f"Found {incomplete_events} event(s) with missing required fields")
        
        is_valid = len([i for i in issues if not i.startswith('Warning')]) == 0
        
        return is_valid, issues
        
    finally:
        conn.close()


def get_database_stats(db_path: str) -> dict:
    """
    Get statistics about the database.
    
    Returns:
        Dictionary with counts and stats
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    
    try:
        # Total events
        cursor.execute("SELECT COUNT(*) FROM events")
        stats['total_events'] = cursor.fetchone()[0]
        
        # Events by type
        cursor.execute("SELECT type, COUNT(*) FROM events GROUP BY type")
        stats['events_by_type'] = dict(cursor.fetchall())
        
        # Total datapoints
        cursor.execute("SELECT COUNT(*) FROM datapoints")
        stats['total_datapoints'] = cursor.fetchone()[0]
        
        # Average datapoints per event
        cursor.execute("SELECT AVG(datapoint_count) FROM events")
        stats['avg_datapoints_per_event'] = round(cursor.fetchone()[0] or 0, 1)
        
        # Date range
        cursor.execute("SELECT MIN(dataTime), MAX(dataTime) FROM events")
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = (min_date, max_date)
        
        # Schema version
        stats['schema_version'] = get_schema_version(db_path)
        
        # Database file size
        db_size = Path(db_path).stat().st_size
        stats['database_size_mb'] = round(db_size / (1024 * 1024), 2)
        
        return stats
        
    finally:
        conn.close()


def list_backups(db_path: str) -> List[Tuple[str, datetime, float]]:
    """
    List all backups for a given database.
    
    Args:
        db_path: Path to database file
        
    Returns:
        List of tuples (backup_path, backup_time, size_mb)
    """
    db_path = Path(db_path)
    backup_pattern = f"{db_path.name}.backup.*"
    
    backups = []
    for backup_file in db_path.parent.glob(backup_pattern):
        # Extract timestamp from filename
        timestamp_str = backup_file.name.split('.')[-1]
        try:
            backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            backups.append((str(backup_file), backup_time, round(size_mb, 2)))
        except ValueError:
            continue
    
    # Sort by time (most recent first)
    backups.sort(key=lambda x: x[1], reverse=True)
    
    return backups


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Database utility operations')
    parser.add_argument('command', choices=['backup', 'validate', 'stats', 'list-backups'],
                       help='Command to execute')
    parser.add_argument('--db', required=True, help='Path to database file')
    parser.add_argument('--backup-dir', help='Backup directory (for backup command)')
    
    args = parser.parse_args()
    
    if args.command == 'backup':
        backup_path = backup_database(args.db, args.backup_dir)
        print(f"Backup created: {backup_path}")
    
    elif args.command == 'validate':
        is_valid, issues = validate_database(args.db)
        print(f"\nDatabase: {args.db}")
        print(f"Valid: {is_valid}\n")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("No issues found!")
    
    elif args.command == 'stats':
        stats = get_database_stats(args.db)
        print(f"\nDatabase Statistics: {args.db}")
        print(f"  Schema Version: {stats['schema_version']}")
        print(f"  Total Events: {stats['total_events']}")
        print(f"  Total Datapoints: {stats['total_datapoints']}")
        print(f"  Avg Datapoints/Event: {stats['avg_datapoints_per_event']}")
        print(f"  Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"  Database Size: {stats['database_size_mb']} MB")
        print(f"  Events by Type:")
        for event_type, count in stats['events_by_type'].items():
            print(f"    {event_type}: {count}")
    
    elif args.command == 'list-backups':
        backups = list_backups(args.db)
        if backups:
            print(f"\nBackups for {args.db}:")
            for backup_path, backup_time, size_mb in backups:
                print(f"  {backup_time.strftime('%Y-%m-%d %H:%M:%S')} - {size_mb} MB - {backup_path}")
        else:
            print(f"No backups found for {args.db}")
