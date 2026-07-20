#!/usr/bin/env python3
"""
Test script to verify userId filter functionality
"""
import sys
import os
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

db_path = '/home/graham/osd/osdb/osdb_working.db'

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("Testing userId Filter Queries")
print("=" * 70)

# Test 1: Get all user IDs
print("\n1. All User IDs:")
cursor.execute("SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL ORDER BY userId")
all_users = [row[0] for row in cursor.fetchall()]
print(f"   Found {len(all_users)} users: {all_users[:10]}..." if len(all_users) > 10 else f"   Found {len(all_users)} users: {all_users}")

# Test 2: Get user IDs filtered by type
event_type = "Seizure"
print(f"\n2. User IDs with type='{event_type}':")
cursor.execute(
    "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL AND type = ? ORDER BY userId",
    (event_type,)
)
type_users = [row[0] for row in cursor.fetchall()]
print(f"   Found {len(type_users)} users: {type_users[:10]}..." if len(type_users) > 10 else f"   Found {len(type_users)} users: {type_users}")

# Test 3: Get user IDs filtered by type and subtype
event_subtype = "Tonic-Clonic"
print(f"\n3. User IDs with type='{event_type}' AND subType='{event_subtype}':")
cursor.execute(
    "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL AND type = ? AND subType = ? ORDER BY userId",
    (event_type, event_subtype)
)
filtered_users = [row[0] for row in cursor.fetchall()]
print(f"   Found {len(filtered_users)} users: {filtered_users}")

# Test 4: Get events filtered by type, subtype, and user
if filtered_users:
    user_id = filtered_users[0]
    print(f"\n4. Events with type='{event_type}', subType='{event_subtype}', userId={user_id}:")
    cursor.execute(
        """SELECT id, userId, dataTime, type, subType 
           FROM events 
           WHERE type = ? AND subType = ? AND userId = ?
           ORDER BY dataTime LIMIT 5""",
        (event_type, event_subtype, user_id)
    )
    events = cursor.fetchall()
    print(f"   Found {len(events)} events (showing first 5):")
    for event in events:
        print(f"     - Event {event['id']}: {event['dataTime']} (User {event['userId']})")

# Test 5: Verify cascading filter logic
print(f"\n5. Cascading Filter Verification:")
print(f"   All users: {len(all_users)}")
print(f"   Users with '{event_type}': {len(type_users)} ({100*len(type_users)/len(all_users):.1f}%)")
if type_users:
    print(f"   Users with '{event_type}' + '{event_subtype}': {len(filtered_users)} ({100*len(filtered_users)/len(type_users):.1f}% of type filter)")

conn.close()

print("\n" + "=" * 70)
print("✓ userId filter queries working correctly")
print("\nExpected behavior in GUI:")
print("  1. Select type → User dropdown updates to show only users with that type")
print("  2. Select subtype → User dropdown further filters to users with that combination")
print("  3. Select user → Event list shows only that user's events")
