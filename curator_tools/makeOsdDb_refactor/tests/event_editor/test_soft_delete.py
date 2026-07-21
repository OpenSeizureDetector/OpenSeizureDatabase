"""
Tests for soft delete operations.

Tests that marking events as 'Deleted' or 'Unknown' preserves data correctly.
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


def test_mark_as_deleted(db_manager, sample_events_db):
    """Test soft delete by changing type to 'Deleted'."""
    event_id = 'test_001'
    original = get_event_from_db(sample_events_db, event_id)
    
    # Simulate marking as deleted (type = 'Deleted')
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Deleted',  # Soft delete
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    assert success
    
    # Verify event still exists but marked as Deleted
    deleted = get_event_from_db(sample_events_db, event_id)
    assert deleted is not None, "Event should still exist in database"
    assert deleted['type'] == 'Deleted'
    
    # Verify other fields preserved
    assert deleted['subType'] == original['subType']
    assert deleted['desc'] == original['desc']
    assert deleted['userId'] == original['userId']
    assert deleted['dataTime'] == original['dataTime']
    assert deleted['seizureTimes'] == original['seizureTimes']


def test_mark_as_unknown(db_manager, sample_events_db):
    """Test soft delete by changing type to 'Unknown'."""
    event_id = 'test_002'
    original = get_event_from_db(sample_events_db, event_id)
    
    # Simulate marking as unknown (type = 'Unknown')
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Unknown',  # Soft delete
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    assert success
    
    # Verify event still exists but marked as Unknown
    unknown = get_event_from_db(sample_events_db, event_id)
    assert unknown is not None, "Event should still exist in database"
    assert unknown['type'] == 'Unknown'


def test_soft_delete_does_not_affect_other_events(db_manager, sample_events_db):
    """Test that marking one event as deleted doesn't affect others."""
    event_id = 'test_001'
    
    # Get snapshot of other events
    other_events_before = [e for e in get_all_events(sample_events_db) if e['id'] != event_id]
    
    # Mark event as deleted
    original = get_event_from_db(sample_events_db, event_id)
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Deleted',
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    assert success
    
    # Get snapshot of other events after
    other_events_after = [e for e in get_all_events(sample_events_db) if e['id'] != event_id]
    
    # Verify other events unchanged
    assert len(other_events_before) == len(other_events_after)
    for event_before, event_after in zip(other_events_before, other_events_after):
        assert event_before['type'] == event_after['type']
        assert event_before['subType'] == event_after['subType']
        assert event_before['desc'] == event_after['desc']
        assert event_before['userId'] == event_after['userId']


def test_soft_delete_preserves_datapoints(db_manager, sample_events_db):
    """Test that soft deleting event doesn't delete its datapoints."""
    event_id = 'test_001'
    
    # Get datapoint count before
    conn = sqlite3.connect(sample_events_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM datapoints WHERE event_id = ?", (event_id,))
    count_before = cursor.fetchone()['count']
    conn.close()
    
    assert count_before > 0, "Event should have datapoints"
    
    # Soft delete event
    original = get_event_from_db(sample_events_db, event_id)
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Deleted',
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    assert success
    
    # Verify datapoints still exist
    conn = sqlite3.connect(sample_events_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM datapoints WHERE event_id = ?", (event_id,))
    count_after = cursor.fetchone()['count']
    conn.close()
    
    assert count_before == count_after, "Datapoints should be preserved"


def test_filter_excludes_deleted_events(db_manager, sample_events_db):
    """Test that filtering can exclude Deleted events."""
    # Mark one event as deleted
    original = get_event_from_db(sample_events_db, 'test_001')
    db_manager.update_event(
        event_id='test_001',
        event_type='Deleted',
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    # Get all events including Deleted
    all_events = db_manager.get_filtered_events()
    assert len(all_events) == 4
    
    # Get only Seizure events (should exclude the Deleted one)
    seizure_events = db_manager.get_filtered_events(event_types=['Seizure'])
    assert len(seizure_events) == 1  # test_002 only
    assert seizure_events[0]['id'] == 'test_002'


def test_soft_delete_preserves_metadata(db_manager, sample_events_db):
    """Test that soft delete preserves all metadata."""
    event_id = 'test_001'
    original = get_event_from_db(sample_events_db, event_id)
    original_metadata = json.loads(original['metadata'])
    
    # Mark as deleted
    success = db_manager.update_event(
        event_id=event_id,
        event_type='Deleted',
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    assert success
    
    # Verify metadata preserved
    deleted = get_event_from_db(sample_events_db, event_id)
    deleted_metadata = json.loads(deleted['metadata'])
    assert deleted_metadata.get('source') == original_metadata.get('source')
    assert deleted_metadata.get('version') == original_metadata.get('version')


def test_soft_delete_can_be_undone(db_manager, sample_events_db):
    """Test that soft delete can be reversed by updating type again."""
    event_id = 'test_001'
    original = get_event_from_db(sample_events_db, event_id)
    original_type = original['type']
    
    # Mark as deleted
    db_manager.update_event(
        event_id=event_id,
        event_type='Deleted',
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    deleted = get_event_from_db(sample_events_db, event_id)
    assert deleted['type'] == 'Deleted'
    
    # Undo deletion by restoring original type
    db_manager.update_event(
        event_id=event_id,
        event_type=original_type,
        subtype=original['subType'],
        description=original['desc'],
        seizure_times=json.loads(original['seizureTimes']) if original['seizureTimes'] else None
    )
    
    restored = get_event_from_db(sample_events_db, event_id)
    assert restored['type'] == original_type
