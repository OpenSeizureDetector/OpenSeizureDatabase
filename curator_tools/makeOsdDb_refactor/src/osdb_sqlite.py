#!/usr/bin/env python3
"""
osdb_sqlite.py - SQLite Working Database for OSDB

Phase 4 implementation: Replace direct JSON manipulation with a SQLite database
that provides fast queries, transactions, and efficient updates while maintaining
JSON export compatibility.

Features:
- Fast indexed queries by event ID, user, time, type
- Transactional safety (no corruption on crash)
- Import from existing JSON files
- Export to JSON (backward compatible)
- Efficient partial updates (no need to rewrite entire file)

Usage:
    from osdb_sqlite import OsdWorkingDb
    
    # Create/open database
    db = OsdWorkingDb('osdb_working.db')
    
    # Import from JSON
    db.import_from_json('osdb_3min_allSeizures.json')
    
    # Query events
    events = db.get_events(user_id=42, event_type='Seizure')
    
    # Add new events
    db.add_events([event1, event2])
    
    # Export to JSON
    db.export_to_json('output.json', event_type='Seizure')
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path


class OsdWorkingDb:
    """
    SQLite-backed working database for OSDB.
    Provides fast access and modification during curation.
    Can export to JSON for publication.
    """
    
    def __init__(self, db_path: str = 'osdb_working.db', debug: bool = False):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
            debug: Enable debug output
        """
        self.db_path = db_path
        self.debug = debug
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_schema()
        
        if self.debug:
            print(f"Opened database: {db_path}")
    
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_datapoints_time ON datapoints(dataTime)")
        
        self.conn.commit()
    
    def import_from_json(self, json_path: str, clear_existing: bool = False) -> int:
        """
        Import events from JSON file into working database.
        
        Args:
            json_path: Path to JSON file
            clear_existing: If True, clear database before importing
            
        Returns:
            Number of events imported
        """
        if self.debug:
            print(f"Importing from {json_path}...")
        
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle wrapped format
        if isinstance(data, dict) and 'events' in data:
            events = data['events']
        else:
            events = data
        
        if clear_existing:
            self.clear_all()
        
        # Import events
        imported = self.add_events(events)
        
        if self.debug:
            print(f"Imported {imported} events from {json_path}")
        
        return imported
    
    def add_events(self, events: List[Dict[str, Any]]) -> int:
        """
        Add events to database.
        
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
            event['hasHrData'] = int(any(dp.get('hr', 0) > 0 for dp in datapoints))
            event['hasO2SatData'] = int(any(dp.get('o2Sat', 0) > 0 for dp in datapoints))
            event['has3dData'] = int(any('rawData3D' in dp for dp in datapoints))
            
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
    
    def get_events(
        self,
        user_id: Optional[int] = None,
        event_type: Optional[str] = None,
        event_subtype: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_ids: Optional[List[int]] = None,
        include_datapoints: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query events with filters.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            event_subtype: Filter by event subtype
            start_time: Filter events after this time (ISO format)
            end_time: Filter events before this time (ISO format)
            event_ids: Filter by specific event IDs
            include_datapoints: Include datapoints in results
            
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        
        # Build query
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if user_id is not None:
            query += " AND userId = ?"
            params.append(user_id)
        
        if event_type is not None:
            query += " AND type = ?"
            params.append(event_type)
        
        if event_subtype is not None:
            query += " AND subType = ?"
            params.append(event_subtype)
        
        if start_time is not None:
            query += " AND dataTime >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND dataTime <= ?"
            params.append(end_time)
        
        if event_ids is not None:
            placeholders = ','.join('?' * len(event_ids))
            query += f" AND id IN ({placeholders})"
            params.extend(event_ids)
        
        query += " ORDER BY dataTime"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        events = []
        for row in rows:
            event = dict(row)
            
            # Restore metadata
            if event['metadata']:
                event.update(json.loads(event['metadata']))
            del event['metadata']
            del event['last_modified']
            
            # Parse merged_from_events
            if event['merged_from_events']:
                event['merged_from_events'] = json.loads(event['merged_from_events'])
            
            # Remove internal fields
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
                    
                    # Parse JSON fields
                    if dp['rawData']:
                        dp['rawData'] = json.loads(dp['rawData'])
                    else:
                        del dp['rawData']
                    
                    if dp['rawData3D']:
                        dp['rawData3D'] = json.loads(dp['rawData3D'])
                    else:
                        del dp['rawData3D']
                    
                    # Remove None fields
                    dp = {k: v for k, v in dp.items() if v is not None}
                    datapoints.append(dp)
                
                event['datapoints'] = datapoints
            
            events.append(event)
        
        return events
    
    def export_to_json(
        self,
        output_path: str,
        user_id: Optional[int] = None,
        event_type: Optional[str] = None,
        pretty: bool = True
    ) -> int:
        """
        Export events to JSON file (backward compatible format).
        
        Args:
            output_path: Output JSON file path
            user_id: Filter by user ID
            event_type: Filter by event type
            pretty: Pretty-print JSON
            
        Returns:
            Number of events exported
        """
        events = self.get_events(user_id=user_id, event_type=event_type)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(events, f, indent=2)
            else:
                json.dump(events, f)
        
        if self.debug:
            print(f"Exported {len(events)} events to {output_path}")
        
        return len(events)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM events")
        total_events = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM datapoints")
        total_datapoints = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT userId) FROM events")
        unique_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT type) FROM events")
        unique_types = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(dataTime), MAX(dataTime) FROM events")
        time_range = cursor.fetchone()
        
        cursor.execute("""
            SELECT type, COUNT(*) as count 
            FROM events 
            GROUP BY type 
            ORDER BY count DESC
        """)
        events_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            'total_events': total_events,
            'total_datapoints': total_datapoints,
            'unique_users': unique_users,
            'unique_types': unique_types,
            'time_range': {'start': time_range[0], 'end': time_range[1]},
            'events_by_type': events_by_type,
            'database_size_bytes': os.path.getsize(self.db_path)
        }
    
    def remove_events(self, event_ids: List[int]) -> int:
        """
        Remove events by ID.
        
        Args:
            event_ids: List of event IDs to remove
            
        Returns:
            Number of events removed
        """
        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(event_ids))
        cursor.execute(f"DELETE FROM events WHERE id IN ({placeholders})", event_ids)
        removed = cursor.rowcount
        self.conn.commit()
        return removed
    
    def clear_all(self):
        """Clear all data from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM datapoints")
        cursor.execute("DELETE FROM events")
        self.conn.commit()
    
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
        description='OSDB SQLite Working Database Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import JSON to database
  python3 osdb_sqlite.py import --input osdb_3min_allSeizures.json --db osdb.db
  
  # Export database to JSON
  python3 osdb_sqlite.py export --db osdb.db --output output.json
  
  # Show statistics
  python3 osdb_sqlite.py stats --db osdb.db
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
    
    db = OsdWorkingDb(args.db, debug=args.debug)
    
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
