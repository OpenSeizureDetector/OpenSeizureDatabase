#!/usr/bin/env python3
"""
Schema Migration v2: Add local change tracking columns

This migration adds columns to track locally-edited fields so that updates
from the remote server don't overwrite user changes made through event_editor.py

New columns:
- local_edits: JSON array of field names that have been edited locally
- remote_hash: Hash of remote version to detect changes
- last_remote_update: Timestamp of last remote data fetch
- has_local_changes: Boolean flag for quick queries

Usage:
    python3 schema_migration_v2.py --db /path/to/osdb_working.db
"""

import sqlite3
import json
import hashlib
import argparse
from datetime import datetime
from typing import Optional, Dict, Any


def compute_event_hash(event: Dict[str, Any]) -> str:
    """
    Compute hash of event fields from remote server.
    This excludes locally-editable fields so changes can be tracked.
    
    Args:
        event: Event dictionary
        
    Returns:
        MD5 hash of remote fields
    """
    # Fields that are NOT editable locally (come from remote only)
    remote_fields = {
        'id', 'userId', 'dataTime', 'dataTimeEnd', 'osdAlarmState',
        'dataSourceName', 'phoneAppVersion', 'watchSdVersion', 'watchFwVersion',
        'watchSdName', 'watchPartNo', 'watchSerialNo', 'alarmTime', 'alarmPhrase',
        'alarmRationale', 'alarmThresh', 'alarmRatioThresh', 'alarmFreqMin',
        'alarmFreqMax', 'hrThreshMin', 'hrThreshMax', 'o2SatThreshMin',
        'o2SatAlarmActive', 'o2SatAlarmStanding', 'batteryPc', 'datapoint_count'
    }
    
    # Extract only remote fields
    remote_data = {k: v for k, v in event.items() if k in remote_fields}
    
    # Serialize and hash
    data_str = json.dumps(remote_data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode()).hexdigest()


def migrate_database(db_path: str, dry_run: bool = False) -> bool:
    """
    Migrate database to schema v2.
    
    Args:
        db_path: Path to SQLite database
        dry_run: If True, don't actually modify database
        
    Returns:
        True if successful, False otherwise
    """
    print(f"Connecting to database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check current schema version
        cursor.execute("SELECT version FROM schema_info ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        current_version = row['version'] if row else 1
        
        print(f"Current schema version: {current_version}")
        
        if current_version >= 2:
            print("Database already at schema v2 or later - no migration needed")
            conn.close()
            return True
        
        if dry_run:
            print("\n[DRY RUN] Would apply the following migrations:")
            print("- Add column 'local_edits' to events table")
            print("- Add column 'remote_hash' to events table")
            print("- Add column 'last_remote_update' to events table")
            print("- Add column 'has_local_changes' to events table")
            print("- Set default values for existing events")
            print("- Create index on 'has_local_changes' for fast queries")
            print("\nNo changes made (dry run mode)")
            conn.close()
            return True
        
        print("\nApplying schema migration v2...")
        
        # Add new columns if they don't exist
        print("  - Adding 'local_edits' column...")
        try:
            cursor.execute("""
                ALTER TABLE events ADD COLUMN local_edits TEXT DEFAULT NULL
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
            print("    (column already exists)")
        
        print("  - Adding 'remote_hash' column...")
        try:
            cursor.execute("""
                ALTER TABLE events ADD COLUMN remote_hash TEXT DEFAULT NULL
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
            print("    (column already exists)")
        
        print("  - Adding 'last_remote_update' column...")
        try:
            cursor.execute("""
                ALTER TABLE events ADD COLUMN last_remote_update TEXT DEFAULT NULL
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
            print("    (column already exists)")
        
        print("  - Adding 'has_local_changes' column...")
        try:
            cursor.execute("""
                ALTER TABLE events ADD COLUMN has_local_changes INTEGER DEFAULT 0
            """)
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise
            print("    (column already exists)")
        
        # Create index for fast queries
        print("  - Creating index on 'has_local_changes'...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_has_local_changes 
            ON events(has_local_changes)
        """)
        
        # Update schema version
        cursor.execute("""
            INSERT INTO schema_info (version, description) 
            VALUES (?, ?)
        """, (2, "Add local change tracking columns"))
        
        conn.commit()
        
        print("\n✓ Schema migration v2 completed successfully")
        print("\nNew columns added to track local changes:")
        print("  - local_edits: JSON array of locally-edited field names")
        print("  - remote_hash: Hash of remote version for change detection")
        print("  - last_remote_update: Timestamp of last remote fetch")
        print("  - has_local_changes: Boolean flag for quick queries")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate OSDB database to schema v2 (add local change tracking)"
    )
    parser.add_argument(
        '--db',
        required=True,
        help='Path to SQLite database'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    success = migrate_database(args.db, dry_run=args.dry_run)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
