"""
Database Manager for event_editor.

Handles all database operations for the OSDB SQLite database.
Separated from GUI code to allow testing without GUI dependencies.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any


class DatabaseManager:
    """Handles database operations for event editing."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def get_event_types(self) -> List[str]:
        """Get unique event types from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT type FROM events WHERE type IS NOT NULL ORDER BY type")
        return [row[0] for row in cursor.fetchall()]
    
    def get_event_subtypes(self, event_type: Optional[str] = None) -> List[str]:
        """Get unique event subtypes, optionally filtered by type."""
        cursor = self.conn.cursor()
        if event_type:
            cursor.execute(
                "SELECT DISTINCT subType FROM events WHERE subType IS NOT NULL AND type = ? ORDER BY subType",
                (event_type,)
            )
        else:
            cursor.execute("SELECT DISTINCT subType FROM events WHERE subType IS NOT NULL ORDER BY subType")
        return [row[0] for row in cursor.fetchall()]
    
    def get_user_ids(self, event_type: Optional[str] = None, event_subtype: Optional[str] = None) -> List[int]:
        """Get unique user IDs, optionally filtered by type and subtype."""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if event_subtype:
            query += " AND subType = ?"
            params.append(event_subtype)
        
        query += " ORDER BY userId"
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def get_filtered_events(
        self, 
        event_types: Optional[List[str]] = None, 
        event_subtypes: Optional[List[str]] = None,
        user_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        desc_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get events matching filters."""
        cursor = self.conn.cursor()
        query = "SELECT id, type, subType, userId, dataTime, desc, datapoint_count FROM events WHERE 1=1"
        params = []
        
        if event_types and len(event_types) > 0:
            placeholders = ','.join(['?'] * len(event_types))
            query += f" AND type IN ({placeholders})"
            params.extend(event_types)
        
        if event_subtypes and len(event_subtypes) > 0:
            placeholders = ','.join(['?'] * len(event_subtypes))
            query += f" AND subType IN ({placeholders})"
            params.extend(event_subtypes)
        
        if user_ids and len(user_ids) > 0:
            placeholders = ','.join(['?'] * len(user_ids))
            query += f" AND userId IN ({placeholders})"
            params.extend(user_ids)
        
        if start_date:
            query += " AND dataTime >= ?"
            params.append(start_date)
        
        if end_date:
            # Add one day to end_date to include events on that day
            query += " AND dataTime < ?"
            params.append(end_date)
        
        if desc_filter:
            query += " AND desc LIKE ? COLLATE NOCASE"
            params.append(desc_filter)
        
        query += " ORDER BY dataTime"
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_event_details(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get complete event details including metadata."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        event = dict(row)
        
        # Parse metadata JSON
        if event['metadata']:
            try:
                metadata = json.loads(event['metadata'])
                event.update(metadata)
            except json.JSONDecodeError:
                pass
        
        # Parse seizureTimes from dedicated column (takes precedence over metadata)
        if event.get('seizureTimes'):
            try:
                event['seizureTimes'] = json.loads(event['seizureTimes'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Get datapoints
        cursor.execute(
            "SELECT * FROM datapoints WHERE event_id = ? ORDER BY dataTime",
            (event_id,)
        )
        datapoints = []
        for dp_row in cursor.fetchall():
            dp = dict(dp_row)
            # Parse JSON fields
            for field in ['rawData', 'rawData3D']:
                if dp.get(field):
                    try:
                        dp[field] = json.loads(dp[field])
                    except json.JSONDecodeError:
                        dp[field] = None
            datapoints.append(dp)
        
        event['datapoints'] = datapoints
        return event
    
    def update_event(
        self, 
        event_id: str, 
        event_type: str, 
        subtype: str, 
        description: str,
        seizure_times: Optional[List[float]] = None
    ) -> bool:
        """Update event fields in database."""
        try:
            cursor = self.conn.cursor()
            
            # Get current metadata
            cursor.execute("SELECT metadata FROM events WHERE id = ?", (event_id,))
            row = cursor.fetchone()
            if not row:
                return False
            
            # Parse existing metadata
            metadata = {}
            if row['metadata']:
                try:
                    metadata = json.loads(row['metadata'])
                except json.JSONDecodeError:
                    pass
            
            # Update metadata with description
            metadata['desc'] = description
            
            # Prepare seizureTimes for dedicated column
            seizure_times_json = None
            if seizure_times is not None:
                seizure_times_json = json.dumps(seizure_times)
            
            # Update database (seizureTimes in dedicated column, not metadata)
            cursor.execute(
                """UPDATE events 
                   SET type = ?, subType = ?, desc = ?, metadata = ?, seizureTimes = ?
                   WHERE id = ?""",
                (event_type, subtype, description, json.dumps(metadata), seizure_times_json, event_id)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating event: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
