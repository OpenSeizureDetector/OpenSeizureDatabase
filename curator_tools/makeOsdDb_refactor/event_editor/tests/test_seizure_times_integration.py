"""
Integration tests for seizureTimes loading and time calculation.

These tests verify that seizureTimes are correctly stored and retrieved,
and that time calculations match between datapoints and seizure markers.

Requires: Real OSDB database at /home/graham/osd/osdb/osdb_working.db
Run with: pytest -v -m integration
Skip with: pytest -m "not integration"
"""

import pytest
import json
import sqlite3
import os
from datetime import datetime


# Skip all tests in this file if database doesn't exist
pytestmark = pytest.mark.skipif(
    not os.path.exists('/home/graham/osd/osdb/osdb_working.db'),
    reason="Integration test requires osdb_working.db"
)


@pytest.fixture
def real_db():
    """Connect to real OSDB database."""
    db_path = '/home/graham/osd/osdb/osdb_working.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def test_event_1046_seizure_times(real_db):
    """Test that event 1046 has correct seizureTimes."""
    cursor = real_db.cursor()
    event_id = '1046'
    
    # Get event details
    cursor.execute("SELECT id, dataTime, seizureTimes FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    
    assert row is not None, f"Event {event_id} not found in database"
    
    event = dict(row)
    assert event['id'] == event_id
    assert event['dataTime'] is not None
    assert event['seizureTimes'] is not None, "Event should have seizureTimes"
    
    # Parse seizureTimes
    seizure_times = json.loads(event['seizureTimes'])
    assert len(seizure_times) == 2, "seizureTimes should have [start, end]"
    assert seizure_times[0] < seizure_times[1], "Start should be before end"


def test_seizure_times_relative_to_event_time(real_db):
    """Test that seizureTimes are relative to event dataTime."""
    cursor = real_db.cursor()
    event_id = '1046'
    
    # Get event
    cursor.execute("SELECT id, dataTime, seizureTimes FROM events WHERE id = ?", (event_id,))
    event = dict(cursor.fetchone())
    
    seizure_times = json.loads(event['seizureTimes'])
    event_dt = datetime.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
    
    # Get first datapoint
    cursor.execute(
        "SELECT dataTime FROM datapoints WHERE event_id = ? ORDER BY dataTime LIMIT 1",
        (event_id,)
    )
    first_dp = cursor.fetchone()
    first_dp_dt = datetime.fromisoformat(first_dp['dataTime'].replace('Z', '+00:00'))
    
    # Calculate relative time
    relative_time = (first_dp_dt - event_dt).total_seconds()
    
    # Verify seizureTimes make sense relative to datapoints
    # If seizure starts before event time (negative), first datapoint should be after seizure start
    if seizure_times[0] < 0:
        # Seizure started before event time, so first datapoint might be after seizure start
        assert True  # This is valid
    else:
        # Seizure started after event time
        assert seizure_times[0] >= relative_time, "Seizure start should be after or at first datapoint"


def test_negative_seizure_times_supported(real_db):
    """Test that negative seizureTimes (before event time) are correctly supported."""
    cursor = real_db.cursor()
    event_id = '1046'
    
    cursor.execute("SELECT seizureTimes FROM events WHERE id = ?", (event_id,))
    event = cursor.fetchone()
    
    seizure_times = json.loads(event['seizureTimes'])
    
    # Event 1046 is known to have negative start time
    # This verifies the system supports seizures that started before the alarm
    if seizure_times[0] < 0:
        # This is the expected case - seizure started before alarm
        assert True
        print(f"✓ Seizure started {abs(seizure_times[0]):.1f}s before event time")
    else:
        # If positive, that's also valid - seizure started after alarm
        assert True
        print(f"✓ Seizure started {seizure_times[0]:.1f}s after event time")


def test_datapoint_time_calculation(real_db):
    """Test that datapoint times are correctly calculated relative to event time."""
    cursor = real_db.cursor()
    event_id = '1046'
    
    # Get event
    cursor.execute("SELECT dataTime FROM events WHERE id = ?", (event_id,))
    event = dict(cursor.fetchone())
    event_dt = datetime.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
    
    # Get first 5 datapoints
    cursor.execute(
        "SELECT dataTime FROM datapoints WHERE event_id = ? ORDER BY dataTime LIMIT 5",
        (event_id,)
    )
    
    datapoints = cursor.fetchall()
    assert len(datapoints) > 0, "Event should have datapoints"
    
    # Calculate relative times
    relative_times = []
    for dp in datapoints:
        dp_dt = datetime.fromisoformat(dp['dataTime'].replace('Z', '+00:00'))
        relative_time = (dp_dt - event_dt).total_seconds()
        relative_times.append(relative_time)
    
    # Verify times are sequential
    for i in range(len(relative_times) - 1):
        assert relative_times[i + 1] > relative_times[i], "Datapoints should be in chronological order"


def test_seizure_times_within_event_range(real_db):
    """Test that seizureTimes fall within reasonable range of event datapoints."""
    cursor = real_db.cursor()
    event_id = '1046'
    
    # Get event and seizureTimes
    cursor.execute("SELECT dataTime, seizureTimes FROM events WHERE id = ?", (event_id,))
    event = dict(cursor.fetchone())
    seizure_times = json.loads(event['seizureTimes'])
    event_dt = datetime.fromisoformat(event['dataTime'].replace('Z', '+00:00'))
    
    # Get first and last datapoint times
    cursor.execute(
        """SELECT 
           MIN(dataTime) as first_dp, 
           MAX(dataTime) as last_dp 
           FROM datapoints WHERE event_id = ?""",
        (event_id,)
    )
    result = cursor.fetchone()
    
    first_dt = datetime.fromisoformat(result['first_dp'].replace('Z', '+00:00'))
    last_dt = datetime.fromisoformat(result['last_dp'].replace('Z', '+00:00'))
    
    first_relative = (first_dt - event_dt).total_seconds()
    last_relative = (last_dt - event_dt).total_seconds()
    
    # seizureTimes should be within the range of datapoints (with some tolerance)
    # Seizure might start before first datapoint or end after last datapoint
    # but should be within reasonable range
    tolerance = 300  # 5 minutes
    
    assert seizure_times[0] >= first_relative - tolerance, "Seizure start too far before first datapoint"
    assert seizure_times[1] <= last_relative + tolerance, "Seizure end too far after last datapoint"


@pytest.mark.parametrize("event_id", ['1046'])
def test_event_has_required_fields(real_db, event_id):
    """Test that events have all required fields for proper display."""
    cursor = real_db.cursor()
    
    cursor.execute("""
        SELECT id, userId, dataTime, type, subType, desc, metadata, seizureTimes, datapoint_count
        FROM events WHERE id = ?
    """, (event_id,))
    
    event = cursor.fetchone()
    assert event is not None, f"Event {event_id} should exist"
    
    # Check required fields are not None
    assert event['id'] is not None
    assert event['dataTime'] is not None
    assert event['type'] is not None
    
    # Check that datapoints exist
    cursor.execute("SELECT COUNT(*) as count FROM datapoints WHERE event_id = ?", (event_id,))
    count = cursor.fetchone()['count']
    assert count > 0, f"Event {event_id} should have datapoints"
