#!/usr/bin/env python3
"""
database_manager.py - SQLite Database Manager for OSDB Refactored Workflow

Provides database-backed storage and retrieval for the makeOsdDb workflow.
Replaces direct JSON file manipulation with efficient SQLite operations.

Usage:
    from database_manager import OsdDatabaseManager
    
    db = OsdDatabaseManager('osdb.db')
    
    # Add events to database
    db.add_events(events)
    
    # Query existing events by ID
    existing_ids = {e['id'] for e in db.get_existing_event_ids()}
    
    # Get all events with datapoints
    all_events = db.get_all_events(include_datapoints=True)
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


class OsdDatabaseManager:
    """SQLite-backed database manager for OSDB workflow.
    
    Provides efficient storage and retrieval of events during the curation process.
    Maintains backward compatibility with existing JSON-based workflows.
    """

    def __init__(self, db_path: str = 'osdb.db', debug: bool = False):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
            debug: Enable debug output
        """
        self.db_path = db_path
        self.debug = debug
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
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
                watchSdName TEXT,
                watchPartNo TEXT,
                watchSerialNo TEXT,
                alarmTime TEXT,
                alarmPhrase TEXT,
                alarmRationale TEXT,
                hasHrData INTEGER DEFAULT 0,
                hasO2SatData INTEGER DEFAULT 0,
                has3dData INTEGER DEFAULT 0,
                merged_from_events TEXT,
                merged_event_count INTEGER DEFAULT 1,
                duration_seconds REAL,
                datapoint_count INTEGER DEFAULT 0,
                metadata TEXT,
                last_modified TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Datapoints table (separate for efficiency)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datapoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                dataTime TEXT NOT NULL,
                alarmState INTEGER,
                hr INTEGER,
                o2Sat INTEGER,
                rawData TEXT,
                rawData3D TEXT,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            )
        """)

        # Create indices for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_user_time ON events(userId, dataTime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type, subType)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_datatime ON events(dataTime)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_event ON datapoints(event_id)")

        self.conn.commit()

    def add_events(self, events: List[Dict[str, Any]]) -> int:
        """Add events to database.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            Number of events added
        """
        cursor = self.conn.cursor()
        added = 0

        for event in events:
            # Extract datapoints
            datapoints = event.pop('datapoints', [])

            # Compute statistics
            event['datapoint_count'] = len(datapoints)
            if datapoints:
                event['hasHrData'] = int(any(dp.get('hr', 0) > 0 for dp in datapoints))
                event['hasO2SatData'] = int(any(dp.get('o2Sat', 0) > 0 for dp in datapoints))
                event['has3dData'] = int(any('rawData3D' in dp for dp in datapoints))
            else:
                event['hasHrData'] = 0
                event['hasO2SatData'] = 0
                event['has3dData'] = 0

            # Store extra fields as JSON
            known_fields = {
                'id', 'userId', 'dataTime', 'dataTimeEnd', 'type', 'subType', 'desc',
                'osdAlarmState', 'dataSourceName', 'phoneAppVersion', 'watchSdVersion',
                'watchSdName', 'watchPartNo', 'watchSerialNo', 'alarmTime', 'alarmPhrase',
                'alarmRationale', 'merged_from_events', 'merged_event_count',
                'duration_seconds', 'datapoint_count', 'hasHrData', 'hasO2SatData', 'has3dData'
            }

            metadata = {k: v for k, v in event.items() if k not in known_fields}
            event['metadata'] = json.dumps(metadata) if metadata else None

            # Convert merged_from_events to JSON string
            if 'merged_from_events' in event and isinstance(event['merged_from_events'], list):
                event['merged_from_events'] = json.dumps(event['merged_from_events'])

            # Insert event (replace if exists)
            cursor.execute("""
                INSERT OR REPLACE INTO events 
                (id, userId, dataTime, dataTimeEnd, type, subType, desc, osdAlarmState,
                 dataSourceName, phoneAppVersion, watchSdVersion, watchSdName, watchPartNo,
                 watchSerialNo, alarmTime, alarmPhrase, alarmRationale, hasHrData, hasO2SatData,
                 has3dData, merged_from_events, merged_event_count, duration_seconds,
                 datapoint_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.get('id'), event.get('userId'), event.get('dataTime'),
                event.get('dataTimeEnd'), event.get('type'), event.get('subType'),
                event.get('desc'), event.get('osdAlarmState'), event.get('dataSourceName'),
                event.get('phoneAppVersion'), event.get('watchSdVersion'), event.get('watchSdName'),
                event.get('watchPartNo'), event.get('watchSerialNo'), event.get('alarmTime'),
                event.get('alarmPhrase'), event.get('alarmRationale'), event.get('hasHrData'),
                event.get('hasO2SatData'), event.get('has3dData'), event.get('merged_from_events'),
                event.get('merged_event_count', 1), event.get('duration_seconds'),
                event.get('datapoint_count'), event.get('metadata')
            ))

            event_id = event['id']

            # Delete old datapoints for this event
            cursor.execute("DELETE FROM datapoints WHERE event_id = ?", (event_id,))

            # Insert datapoints
            for dp in datapoints:
                cursor.execute("""
                    INSERT INTO datapoints (event_id, dataTime, alarmState, hr, o2Sat, rawData, rawData3D)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    dp.get('dataTime'),
                    dp.get('alarmState'),
                    dp.get('hr'),
                    dp.get('o2Sat'),
                    json.dumps(dp.get('rawData')) if 'rawData' in dp else None,
                    json.dumps(dp.get('rawData3D')) if 'rawData3D' in dp else None
                ))

            added += 1

        self.conn.commit()
        return added

    def get_existing_event_ids(self) -> Set[int]:
        """Get all existing event IDs from the database.
        
        Returns:
            Set of existing event IDs
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM events")
        rows = cursor.fetchall()
        return {row[0] for row in rows}

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Get a single event by ID.
        
        Args:
            event_id: Event ID to look up
            
        Returns:
            Event dictionary or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM events WHERE id = ?
        """, (event_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        event = dict(row)

        # Restore metadata and parse JSON fields
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
            dp = dict(dp_row)
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
        return event

    def get_all_events(self, include_datapoints: bool = True) -> List[Dict[str, Any]]:
        """Get all events from the database.
        
        Args:
            include_datapoints: Whether to include datapoints in results
            
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY dataTime")
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event = dict(row)

            # Restore metadata and parse JSON fields
            if event['metadata']:
                event.update(json.loads(event['metadata']))
            del event['metadata']
            del event['last_modified']

            if event['merged_from_events']:
                event['merged_from_events'] = json.loads(event['merged_from_events'])

            for field in ['hasHrData', 'hasO2SatData', 'has3dData', 'datapoint_count']:
                if field in event and event[field] is not None:
                    del event[field]

            # Load datapoints if requested
            if include_datapoints:
                cursor.execute(
                    "SELECT * FROM datapoints WHERE event_id = ? ORDER BY dataTime",
                    (event['id'],)
                )
                datapoints = []
                for dp_row in cursor.fetchall():
                    dp = dict(dp_row)
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

            events.append(event)

        return events

    def export_to_json(self, output_path: str) -> int:
        """Export all events from database to JSON file.
        
        Args:
            output_path: Path for the exported JSON file
            
        Returns:
            Number of events exported
        """
        events = self.get_all_events(include_datapoints=True)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2)

        return len(events)

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """CLI for database operations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OSDB SQLite Database Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import JSON to database
  python3 database_manager.py import --input osdb_3min_allSeizures.json --db osdb.db
  
  # Export database to JSON
  python3 database_manager.py export --db osdb.db --output output.json
  
  # Show statistics
  python3 database_manager.py stats --db osdb.db
        """
    )
    
    parser.add_argument('--db', default='osdb_working.db', help='Database file path')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import from JSON')
    import_parser.add_argument('--input', required=True, help='Input JSON file')
    import_parser.add_argument('--clear', action='store_true', help='Clear existing data')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export to JSON')
    export_parser.add_argument('--output', required=True, help='Output JSON file')
    export_parser.add_argument('--type', help='Filter by event type')
    export_parser.add_argument('--user', type=int, help='Filter by user ID')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db = OsdDatabaseManager(args.db, debug=args.debug)
    
    try:
        if args.command == 'import':
            count = db.import_from_json(args.input, clear_existing=args.clear)
            print(f"✓ Imported {count} events")
        
        elif args.command == 'export':
            count = db.export_to_json(args.output, user_id=args.user, event_type=args.type)
            print(f"✓ Exported {count} events")
        
        elif args.command == 'stats':
            stats = db.get_statistics()
            print("\nDatabase Statistics:")
            print("=" * 50)
            print(f"Total Events:      {stats['total_events']:,}")
            print(f"Total Datapoints:  {stats['total_datapoints']:,}")
            print(f"Unique Users:      {stats['unique_users']}")
            print(f"Unique Types:      {stats['unique_types']}")
            print(f"Time Range:        {stats['time_range']['start']} to {stats['time_range']['end']}")
            print(f"Database Size:     {stats['database_size_bytes']:,} bytes")
            print("\nEvents by Type:")
            for event_type, count in stats['events_by_type'].items():
                print(f"  {event_type}: {count:,}")

    finally:
        db.close()


if __name__ == '__main__':
    main()
