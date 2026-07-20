#!/usr/bin/env python3
"""
init_database.py - Initial Database Setup Script

Reads existing JSON files from the output directory and creates a SQLite database.
This is the one-time setup script that converts all published JSON data into a working SQLite database.

Usage:
    python3 init_database.py --json-dir /path/to/json/files --db osdb.db [--output-dir /path/to/output]
"""

import sqlite3
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re


def normalize_datetime(dt_value: Any) -> Optional[str]:
    """
    Normalize various datetime formats to ISO 8601 format (YYYY-MM-DD HH:MM:SS).
    SQLite can efficiently sort and filter this format.
    
    Handles:
    - ISO 8601 with Z: "2022-11-15T19:33:49Z"
    - ISO 8601 without Z: "2022-11-15T19:33:49"
    - Unix timestamps (integers or floats)
    - Various other formats
    
    Args:
        dt_value: Date/time value (string, int, float)
        
    Returns:
        Normalized ISO 8601 string or None if parsing fails
    """
    if not dt_value:
        return None
    
    # Already in normalized format
    if isinstance(dt_value, str):
        # Remove timezone indicator and normalize
        dt_str = dt_value.strip()
        
        # ISO 8601 with Z
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1]
        
        # Try various formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',           # 2022-11-15T19:33:49
            '%Y-%m-%d %H:%M:%S',           # 2022-11-15 19:33:49
            '%Y-%m-%dT%H:%M:%S.%f',        # With microseconds
            '%Y-%m-%d %H:%M:%S.%f',        # With microseconds
            '%d-%m-%Y %H:%M:%S',           # DD-MM-YYYY format
            '%m/%d/%Y %H:%M:%S',           # US format
            '%Y/%m/%d %H:%M:%S',           # Alternative format
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
        
        # Try to parse partial ISO format (might have fractional seconds)
        try:
            # Handle ISO format with fractional seconds
            match = re.match(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?', dt_str)
            if match:
                dt = datetime.strptime(match.group(1), '%Y-%m-%dT%H:%M:%S')
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass
    
    # Unix timestamp (seconds since epoch)
    elif isinstance(dt_value, (int, float)):
        try:
            # Reasonable range check (1970-2100)
            if 0 < dt_value < 4102444800:
                dt = datetime.fromtimestamp(dt_value)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, OSError):
            pass
    
    # If all parsing fails, return the original string if it looks like a date
    if isinstance(dt_value, str) and len(dt_value) > 8:
        return dt_value
    
    return None


def load_events_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load events from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'events' in data:
        return data['events']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON format in {json_path}")


def create_database(db_path: str, json_dir: str, output_dir: Optional[str] = None) -> int:
    """Create a SQLite database from existing JSON files.

    Args:
        db_path: Path to the new SQLite database file
        json_dir: Directory containing JSON event files
        output_dir: Optional directory for exported JSON (defaults to json_dir)

    Returns:
        Number of events imported
    """
    # Create database connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")

    # Create schema version tracking table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER PRIMARY KEY,
            applied_date TEXT DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    
    # Record current schema version
    cursor.execute("INSERT OR IGNORE INTO schema_info (version, description) VALUES (?, ?)",
                   (1, "Initial schema with full field support"))

    # Create schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            userId INTEGER NOT NULL,
            dataTime TEXT NOT NULL,
            dataTimeEnd TEXT,
            type TEXT,
            subType TEXT,
            desc TEXT,
            osdAlarmState INTEGER,
            dataSourceName TEXT,
            phoneAppVersion TEXT,
            watchSdVersion TEXT,
            watchFwVersion TEXT,
            watchSdName TEXT,
            watchPartNo TEXT,
            watchSerialNo TEXT,
            alarmTime TEXT,
            alarmPhrase TEXT,
            alarmRationale TEXT,
            alarmThresh REAL,
            alarmRatioThresh REAL,
            alarmFreqMin REAL,
            alarmFreqMax REAL,
            hrThreshMin INTEGER,
            hrThreshMax INTEGER,
            o2SatThreshMin INTEGER,
            o2SatAlarmActive INTEGER,
            o2SatAlarmStanding INTEGER,
            batteryPc INTEGER,
            hasHrData INTEGER DEFAULT 0,
            hasO2SatData INTEGER DEFAULT 0,
            has3dData INTEGER DEFAULT 0,
            seizureTimes TEXT,
            merged_from_events TEXT,
            merged_event_count INTEGER DEFAULT 1,
            duration_seconds REAL,
            datapoint_count INTEGER DEFAULT 0,
            metadata TEXT,
            last_modified TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datapoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            dataTime TEXT NOT NULL,
            alarmState INTEGER,
            hr INTEGER,
            o2Sat INTEGER,
            rawData TEXT,
            rawData3D TEXT,
            specPower REAL,
            roiPower REAL,
            roiRatio REAL,
            maxVal REAL,
            maxFreq REAL,
            FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
        )
    """)

    # Create indices for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(userId, dataTime)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, subType)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_datatime ON events(dataTime)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_event ON datapoints(event_id)")

    conn.commit()

    # Find all JSON files in the directory
    json_files = list(Path(json_dir).glob('*.json'))
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return 0

    # Sort files so seizure files are processed first (they take priority over NDA events)
    # Priority order: tcSeizures, allSeizures, fallEvents, falseAlarms, ndaEvents, others
    def file_priority(path):
        name = path.name.lower()
        if 'tcseizure' in name:
            return 0
        elif 'allseizure' in name:
            return 1
        elif 'fall' in name:
            return 2
        elif 'falsealarm' in name:
            return 3
        elif 'nda' in name:
            return 4
        else:
            return 5
    
    json_files_sorted = sorted(json_files, key=lambda p: (file_priority(p), p.name))

    total_imported = 0
    total_skipped = 0
    imported_ids = set()

    for json_file in json_files_sorted:
        try:
            events = load_events_from_json(str(json_file))
            if not events:
                continue

            # Filter out events with duplicate IDs (keep first occurrence across files)
            unique_events = []
            duplicate_count = 0
            for event in events:
                event_id = event.get('id')
                if not event_id:
                    continue
                if event_id in imported_ids:
                    duplicate_count += 1
                else:
                    unique_events.append(event)
            
            if duplicate_count > 0:
                print(f"  Note: {json_file.name} has {duplicate_count} duplicate event(s), importing {len(unique_events)} unique events")
            
            if not unique_events:
                print(f"  Skipping {json_file.name}: all events are duplicates")
                total_skipped += duplicate_count
                continue

            # Import unique events into database
            try:
                imported_count = import_events_batch(conn, unique_events)
                total_imported += imported_count
                total_skipped += duplicate_count
                
                # Track imported IDs
                imported_ids.update(e['id'] for e in unique_events if 'id' in e)
                
                if duplicate_count > 0:
                    print(f"  ✓ Imported {imported_count} events from {json_file.name} ({duplicate_count} duplicates skipped)")
                else:
                    print(f"  ✓ Imported {imported_count} events from {json_file.name}")
            except Exception as import_err:
                print(f"  Error importing {json_file.name}: {import_err}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
            continue

    conn.close()

    if total_skipped > 0:
        print(f"\n✓ Imported {total_imported} events from {len(json_files)} JSON files ({total_skipped} duplicates skipped)")
    else:
        print(f"\n✓ Imported {total_imported} events from {len(json_files)} JSON files")
    
    return total_imported


def import_events_batch(conn, events: List[Dict[str, Any]]) -> int:
    """Import a batch of events into the database.

    Args:
        conn: SQLite connection
        events: List of event dictionaries

    Returns:
        Number of events imported
    """
    cursor = conn.cursor()
    added = 0

    for event in events:
        # Extract datapoints
        datapoints = event.pop('datapoints', [])

        # Normalize datetime fields to ISO 8601 format for efficient filtering
        if 'dataTime' in event:
            event['dataTime'] = normalize_datetime(event['dataTime'])
        if 'dataTimeEnd' in event:
            event['dataTimeEnd'] = normalize_datetime(event['dataTimeEnd'])
        if 'alarmTime' in event:
            event['alarmTime'] = normalize_datetime(event['alarmTime'])

        # Compute statistics (handle None values)
        event['datapoint_count'] = len(datapoints)
        event['hasHrData'] = int(any((dp.get('hr') or 0) > 0 for dp in datapoints)) if datapoints else 0
        event['hasO2SatData'] = int(any((dp.get('o2Sat') or 0) > 0 for dp in datapoints)) if datapoints else 0
        event['has3dData'] = int(any('rawData3D' in dp for dp in datapoints)) if datapoints else 0

        # Store extra fields as JSON
        known_fields = {
            'id', 'userId', 'dataTime', 'dataTimeEnd', 'type', 'subType', 'desc',
            'osdAlarmState', 'dataSourceName', 'phoneAppVersion', 'watchSdVersion', 
            'watchFwVersion', 'watchSdName', 'watchPartNo', 'watchSerialNo', 'alarmTime', 
            'alarmPhrase', 'alarmRationale', 'alarmThresh', 'alarmRatioThresh', 
            'alarmFreqMin', 'alarmFreqMax', 'hrThreshMin', 'hrThreshMax', 'o2SatThreshMin',
            'o2SatAlarmActive', 'o2SatAlarmStanding', 'batteryPc', 'seizureTimes',
            'merged_from_events', 'merged_event_count', 'duration_seconds', 'datapoint_count', 
            'hasHrData', 'hasO2SatData', 'has3dData'
        }

        metadata = {k: v for k, v in event.items() if k not in known_fields}
        event['metadata'] = json.dumps(metadata) if metadata else None

        # Convert arrays/lists to JSON strings
        if 'merged_from_events' in event and isinstance(event['merged_from_events'], list):
            event['merged_from_events'] = json.dumps(event['merged_from_events'])
        if 'seizureTimes' in event and isinstance(event['seizureTimes'], list):
            event['seizureTimes'] = json.dumps(event['seizureTimes'])

        # Insert event (replace if exists)
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO events 
                (id, userId, dataTime, dataTimeEnd, type, subType, desc, osdAlarmState,
                 dataSourceName, phoneAppVersion, watchSdVersion, watchFwVersion, watchSdName, 
                 watchPartNo, watchSerialNo, alarmTime, alarmPhrase, alarmRationale,
                 alarmThresh, alarmRatioThresh, alarmFreqMin, alarmFreqMax,
                 hrThreshMin, hrThreshMax, o2SatThreshMin, o2SatAlarmActive, o2SatAlarmStanding,
                 batteryPc, hasHrData, hasO2SatData, has3dData, seizureTimes,
                 merged_from_events, merged_event_count, duration_seconds, datapoint_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get('id'), event.get('userId'), event.get('dataTime'),
                event.get('dataTimeEnd'), event.get('type'), event.get('subType'),
                event.get('desc'), event.get('osdAlarmState'), event.get('dataSourceName'),
                event.get('phoneAppVersion'), event.get('watchSdVersion'), event.get('watchFwVersion'),
                event.get('watchSdName'), event.get('watchPartNo'), event.get('watchSerialNo'), 
                event.get('alarmTime'), event.get('alarmPhrase'), event.get('alarmRationale'),
                event.get('alarmThresh'), event.get('alarmRatioThresh'), event.get('alarmFreqMin'),
                event.get('alarmFreqMax'), event.get('hrThreshMin'), event.get('hrThreshMax'),
                event.get('o2SatThreshMin'), event.get('o2SatAlarmActive'), event.get('o2SatAlarmStanding'),
                event.get('batteryPc'), event.get('hasHrData'), event.get('hasO2SatData'), 
                event.get('has3dData'), event.get('seizureTimes'), event.get('merged_from_events'),
                event.get('merged_event_count', 1), event.get('duration_seconds'),
                event.get('datapoint_count'), event.get('metadata')
            ))
        except sqlite3.IntegrityError as e:
            print(f"    Warning: Skipping event {event.get('id')}: {e}")
            continue

        event_id = event['id']

        # Delete old datapoints for this event
        cursor.execute("DELETE FROM datapoints WHERE event_id = ?", (event_id,))

        # Insert datapoints
        for dp in datapoints:
            # Normalize datapoint datetime
            dp_time = normalize_datetime(dp.get('dataTime'))
            
            cursor.execute("""
                INSERT INTO datapoints 
                (event_id, dataTime, alarmState, hr, o2Sat, rawData, rawData3D, 
                 specPower, roiPower, roiRatio, maxVal, maxFreq)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                dp_time,
                dp.get('alarmState'),
                dp.get('hr'),
                dp.get('o2Sat'),
                json.dumps(dp.get('rawData')) if 'rawData' in dp else None,
                json.dumps(dp.get('rawData3D')) if 'rawData3D' in dp else None,
                dp.get('specPower'),
                dp.get('roiPower'),
                dp.get('roiRatio'),
                dp.get('maxVal'),
                dp.get('maxFreq')
            ))

        added += 1

    conn.commit()
    return added


def export_to_json(db_path: str, output_dir: Optional[str] = None) -> int:
    """Export events from the database back to JSON files.

    Args:
        db_path: Path to SQLite database
        output_dir: Directory for exported JSON (defaults to same as input dir)

    Returns:
        Number of events exported
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all event IDs
    cursor.execute("SELECT id FROM events")
    event_ids = [row[0] for row in cursor.fetchall()]

    if not event_ids:
        print("No events found in database")
        return 0

    # Load events from database
    conn.row_factory = sqlite3.Row
    cursor.execute("""
        SELECT * FROM events WHERE id IN ({}) ORDER BY dataTime
    """.format(','.join('?' for _ in event_ids)), event_ids)

    rows = cursor.fetchall()
    events = [{key: row[key] for key in row.keys()} for row in rows]

    # Restore metadata and parse JSON fields
    for event in events:
        if event['metadata']:
            event.update(json.loads(event['metadata']))
        del event['metadata']
        del event['last_modified']

        if event['merged_from_events']:
            event['merged_from_events'] = json.loads(event['merged_from_events'])

        for field in ['hasHrData', 'hasO2SatData', 'has3dData', 'datapoint_count']:
            if field in event and event[field] is not None:
                del event[field]

        # Load datapoints
        cursor.execute(
            "SELECT * FROM datapoints WHERE event_id = ? ORDER BY dataTime",
            (event['id'],)
        )
        datapoints = []
        for dp_row in cursor.fetchall():
            dp = {key: dp_row[key] for key in dp_row.keys()}
            del dp['id']
            del dp['event_id']

            if dp['rawData']:
                dp['rawData'] = json.loads(dp['rawData'])
            else:
                del dp['rawData']

            if dp['rawData3D']:
                dp['rawData3D'] = json.loads(dp['rawData3D'])
            else:
                del dp['rawData3D']

            dp = {k: v for k, v in dp.items() if v is not None}
            datapoints.append(dp)

        event['datapoints'] = datapoints

    # Determine output directory
    if output_dir is None:
        import os
        output_dir = str(Path(db_path).parent)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export to JSON files (one per type or all in one file)
    event_types = set(e.get('type', '') for e in events if 'type' in e)
    print(f"Exporting {len(events)} events of types: {event_types}")

    with open(str(Path(output_dir) / 'osdb_exported.json'), 'w') as f:
        json.dump({'events': events}, f, indent=2)

    print(f"✓ Exported to osdb_exported.json")
    return len(events)


def main():
    """CLI for database initialization."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize OSDB from existing JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize database from current directory's JSON files
  python3 init_database.py --json-dir . --db osdb.db

  # Specify output directory for exported JSON
  python3 init_database.py --json-dir /path/to/json --db osdb.db --output-dir /path/to/output
        """
    )

    parser.add_argument('--json-dir', required=True, help='Directory containing JSON event files')
    parser.add_argument('--db', default='osdb.db', help='Output SQLite database path')
    parser.add_argument('--output-dir', help='Directory for exported JSON (default: same as json-dir)')

    parser.add_argument('--verify', action='store_true', 
                        help='Verify import by exporting to JSON')

    args = parser.parse_args()

    print(f"Initializing OSDB from {args.json_dir}...")
    total = create_database(args.db, args.json_dir, args.output_dir)
    
    if args.verify:
        print("\nVerifying import by exporting to JSON...")
        try:
            export_to_json(args.db, args.output_dir)
        except Exception as e:
            print(f"Warning: Export verification failed: {e}")
    
    print(f"\n✓ Database successfully created: {args.db}")


if __name__ == '__main__':
    main()