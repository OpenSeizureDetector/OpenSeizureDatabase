"""
Tests for event update operations.

Tests that updating event metadata works correctly and doesn't corrupt other data.
"""

import pytest
import json
import sqlite3


def get_event_from_db(db_path: str, event_id: str):
    """Helper to get event directly from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_events(db_path: str):
    """Helper to get all events from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, type, subType, desc, userId, dataTime FROM events ORDER BY id")
    events = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return events


def test_update_basic_fields(db_manager, sample_events_db):
    """Test updating basic event fields (type, subType, desc)."""
    event_id = 'test_001'
    
    # Get original event
    original = get_event_from_db(sample_events_db, event_id)
    
    # Update event
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Focal',  # Changed from Tonic-Clonic
        description='Updated description',  # Changed
        seizure_times=[-20.0, 50.0]  # Unchanged
    )
    
    assert success, "Update should succeed"
    
    # Verify changes
    updated = get_event_from_db(sample_events_db, event_id)
    assert updated['type'] == 'Seizure'
    assert updated['subType'] == 'Focal'
    assert updated['desc'] == 'Updated description'
    assert updated['userId'] == original['userId']  # Unchanged
    assert updated['dataTime'] == original['dataTime']  # Unchanged
    
    # Verify seizureTimes
    seizure_times = json.loads(updated['seizureTimes'])
    assert seizure_times == [-20.0, 50.0]


def test_update_seizure_times(db_manager, sample_events_db):
    """Test updating seizureTimes field."""
    event_id = 'test_001'
    
    # Update with new seizureTimes
    new_times = [-15.0, 45.0]
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Tonic-Clonic',
        description='Test seizure event 1',
        seizure_times=new_times
    )
    
    assert success
    
    # Verify seizureTimes updated
    updated = get_event_from_db(sample_events_db, event_id)
    seizure_times = json.loads(updated['seizureTimes'])
    assert seizure_times == new_times


def test_update_clear_seizure_times(db_manager, sample_events_db):
    """Test clearing seizureTimes by passing None."""
    event_id = 'test_001'
    
    # Update with None to clear seizureTimes
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Tonic-Clonic',
        description='Test seizure event 1',
        seizure_times=None
    )
    
    assert success
    
    # Verify seizureTimes is None
    updated = get_event_from_db(sample_events_db, event_id)
    assert updated['seizureTimes'] is None


def test_update_preserves_metadata(db_manager, sample_events_db):
    """Test that update preserves existing metadata fields."""
    event_id = 'test_001'
    original = get_event_from_db(sample_events_db, event_id)
    original_metadata = json.loads(original['metadata'])
    
    # Update event
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Tonic-Clonic',
        description='Updated description',
        seizure_times=[-20.0, 50.0]
    )
    
    assert success
    
    # Verify metadata still has original fields
    updated = get_event_from_db(sample_events_db, event_id)
    updated_metadata = json.loads(updated['metadata'])
    
    # Original fields should be preserved
    assert updated_metadata.get('source') == original_metadata.get('source')
    assert updated_metadata.get('version') == original_metadata.get('version')
    # Description should be updated in metadata
    assert updated_metadata.get('desc') == 'Updated description'


def test_update_does_not_affect_other_events(db_manager, sample_events_db):
    """Test that updating one event doesn't affect others."""
    # Get all events before update
    events_before = get_all_events(sample_events_db)
    
    # Update one event
    success = db_manager.update_event(
        event_id='test_001',
        event_type='Seizure',
        subtype='Updated',
        description='Updated event',
        seizure_times=[-10.0, 20.0]
    )
    
    assert success
    
    # Get all events after update
    events_after = get_all_events(sample_events_db)
    
    # Verify same number of events
    assert len(events_before) == len(events_after)
    
    # Verify only test_001 changed
    for event_before, event_after in zip(events_before, events_after):
        if event_before['id'] == 'test_001':
            # This event should be different
            assert event_before['subType'] != event_after['subType']
            assert event_before['desc'] != event_after['desc']
        else:
            # Other events should be identical
            assert event_before['type'] == event_after['type']
            assert event_before['subType'] == event_after['subType']
            assert event_before['desc'] == event_after['desc']
            assert event_before['userId'] == event_after['userId']
            assert event_before['dataTime'] == event_after['dataTime']


def test_update_does_not_affect_datapoints(db_manager, sample_events_db):
    """Test that updating event metadata doesn't affect its datapoints."""
    import sqlite3
    
    event_id = 'test_001'
    
    # Get datapoints before update
    conn = sqlite3.connect(sample_events_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, dataTime, rawData, hr FROM datapoints WHERE event_id = ? ORDER BY id",
        (event_id,)
    )
    datapoints_before = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Update event
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Updated',
        description='Updated',
        seizure_times=[-5.0, 25.0]
    )
    
    assert success
    
    # Get datapoints after update
    conn = sqlite3.connect(sample_events_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, dataTime, rawData, hr FROM datapoints WHERE event_id = ? ORDER BY id",
        (event_id,)
    )
    datapoints_after = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    # Verify datapoints unchanged
    assert len(datapoints_before) == len(datapoints_after)
    for dp_before, dp_after in zip(datapoints_before, datapoints_after):
        assert dp_before['id'] == dp_after['id']
        assert dp_before['dataTime'] == dp_after['dataTime']
        assert dp_before['rawData'] == dp_after['rawData']
        assert dp_before['hr'] == dp_after['hr']


def test_update_nonexistent_event(db_manager):
    """Test updating an event that doesn't exist."""
    success = db_manager.update_event(
        event_id='nonexistent',
        event_type='Seizure',
        subtype='Test',
        description='Test',
        seizure_times=None
    )
    
    assert not success, "Update should fail for nonexistent event"


def test_database_integrity_after_multiple_operations(db_manager, sample_events_db):
    """Test database integrity after multiple update operations."""
    import sqlite3
    
    operations = [
        ('test_001', 'Seizure', 'Updated1', 'Desc1', [-10.0, 30.0]),
        ('test_002', 'Seizure', 'Absence', 'Desc2', None),
        ('test_003', 'False Alarm', 'Updated3', 'Desc3', None),
        ('test_004', 'Fall', 'Detected', 'Desc4', None),
    ]
    
    # Get expected total datapoints
    conn = sqlite3.connect(sample_events_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM datapoints")
    expected_count = cursor.fetchone()[0]
    conn.close()
    
    # Perform multiple updates
    for event_id, event_type, subtype, desc, seizure_times in operations:
        success = db_manager.update_event(event_id, event_type, subtype, desc, seizure_times)
        assert success, f"Update {event_id} should succeed"
    
    # Verify all events still exist
    all_events = get_all_events(sample_events_db)
    assert len(all_events) == 4, "All events should still exist"
    
    # Verify each event has correct updated values
    for event_id, event_type, subtype, desc, seizure_times in operations:
        event = get_event_from_db(sample_events_db, event_id)
        assert event is not None
        assert event['type'] == event_type
        assert event['subType'] == subtype
        assert event['desc'] == desc
    
    # Verify all datapoints still exist
    conn = sqlite3.connect(sample_events_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM datapoints")
    total_datapoints = cursor.fetchone()[0]
    conn.close()
    assert total_datapoints == expected_count, "All datapoints should be preserved"


def test_concurrent_update_safety(db_manager, sample_events_db):
    """Test that updates are properly committed/rolled back."""
    event_id = 'test_001'
    
    # Successful update
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Seizure',
        subtype='Updated',
        description='Test',
        seizure_times=None
    )
    assert success
    
    # Verify change persisted
    event = get_event_from_db(sample_events_db, event_id)
    assert event['subType'] == 'Updated'
    
    # Try to update nonexistent event (should fail without affecting database)
    events_before = get_all_events(sample_events_db)
    success = db_manager.update_event(
        event_id='nonexistent',
        event_type='Test',
        subtype='Test',
        description='Test',
        seizure_times=None
    )
    assert not success
    
    # Verify database unchanged after failed update
    events_after = get_all_events(sample_events_db)
    assert len(events_before) == len(events_after)
