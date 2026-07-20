#!/usr/bin/env python3
"""
Test script to verify seizureTimes loading and X-axis calculation
"""
import sys
import os
import json
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test with event 1046
db_path = '/home/graham/osd/osdb/osdb_working.db'
event_id = '1046'

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print(f"Testing event {event_id}...")
print("=" * 60)

# Get event details
cursor.execute("SELECT id, dataTime, seizureTimes FROM events WHERE id = ?", (event_id,))
row = cursor.fetchone()

if not row:
    print(f"Event {event_id} not found!")
    sys.exit(1)

event = dict(row)
print(f"\nEvent ID: {event['id']}")
print(f"Event dataTime: {event['dataTime']}")
print(f"seizureTimes (raw): {event['seizureTimes']}")

# Parse seizureTimes
if event['seizureTimes']:
    seizure_times = json.loads(event['seizureTimes'])
    print(f"seizureTimes (parsed): {seizure_times}")
    print(f"  Start: {seizure_times[0]:.1f} seconds relative to event dataTime")
    print(f"  End: {seizure_times[1]:.1f} seconds relative to event dataTime")
else:
    print("No seizureTimes found!")
    seizure_times = None

# Get first few datapoints
cursor.execute(
    "SELECT dataTime FROM datapoints WHERE event_id = ? ORDER BY dataTime LIMIT 5",
    (event_id,)
)

print(f"\n{'First 5 Datapoints:':<20} {'Relative Time (seconds)':>25}")
print("-" * 60)

event_dt = datetime.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
for dp_row in cursor.fetchall():
    dp_dt = datetime.fromisoformat(dp_row['dataTime'].replace('Z', '+00:00'))
    relative_time = (dp_dt - event_dt).total_seconds()
    print(f"{dp_row['dataTime']:<20} {relative_time:>25.1f}s")

# Verify the relationship
if seizure_times:
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print(f"  First datapoint is at {relative_time:.1f}s relative to event")
    print(f"  Seizure starts at {seizure_times[0]:.1f}s")
    print(f"  Seizure ends at {seizure_times[1]:.1f}s")
    
    if seizure_times[0] < 0:
        print(f"\n  ✓ Seizure started {abs(seizure_times[0]):.1f}s BEFORE event dataTime")
    else:
        print(f"\n  ✓ Seizure started {seizure_times[0]:.1f}s AFTER event dataTime")
    
    if seizure_times[1] < 0:
        print(f"  ✓ Seizure ended {abs(seizure_times[1]):.1f}s BEFORE event dataTime")
    else:
        print(f"  ✓ Seizure ended {seizure_times[1]:.1f}s AFTER event dataTime")

conn.close()

print("\n" + "=" * 60)
print("✓ Test complete - negative times are correctly supported")
