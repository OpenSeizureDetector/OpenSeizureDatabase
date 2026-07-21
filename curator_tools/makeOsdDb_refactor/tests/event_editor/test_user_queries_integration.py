"""
Integration tests for user filtering queries.

Tests the cascading filter logic and user ID filtering functionality.

Requires: Real OSDB database at /home/graham/osd/osdb/osdb_working.db
Run with: pytest -v -m integration
Skip with: pytest -m "not integration"
"""

import pytest
import sqlite3
import os


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


def test_get_all_user_ids(real_db):
    """Test getting all user IDs from database."""
    cursor = real_db.cursor()
    cursor.execute("SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL ORDER BY userId")
    all_users = [row[0] for row in cursor.fetchall()]
    
    assert len(all_users) > 0, "Database should have users"
    # Verify they're unique and sorted
    assert all_users == sorted(set(all_users)), "User IDs should be unique and sorted"


def test_get_users_filtered_by_type(real_db):
    """Test getting user IDs filtered by event type."""
    cursor = real_db.cursor()
    event_type = "Seizure"
    
    # Get all users
    cursor.execute("SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL ORDER BY userId")
    all_users = [row[0] for row in cursor.fetchall()]
    
    # Get users with Seizure events
    cursor.execute(
        "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL AND type = ? ORDER BY userId",
        (event_type,)
    )
    type_users = [row[0] for row in cursor.fetchall()]
    
    assert len(type_users) > 0, f"Database should have users with {event_type} events"
    assert len(type_users) <= len(all_users), "Filtered users should be subset of all users"
    
    # Verify all returned users actually have seizure events
    for user_id in type_users:
        cursor.execute(
            "SELECT COUNT(*) as count FROM events WHERE userId = ? AND type = ?",
            (user_id, event_type)
        )
        count = cursor.fetchone()['count']
        assert count > 0, f"User {user_id} should have {event_type} events"


def test_get_users_filtered_by_type_and_subtype(real_db):
    """Test getting user IDs filtered by type and subtype."""
    cursor = real_db.cursor()
    event_type = "Seizure"
    event_subtype = "Tonic-Clonic"
    
    # Get users with type only
    cursor.execute(
        "SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL AND type = ? ORDER BY userId",
        (event_type,)
    )
    type_users = [row[0] for row in cursor.fetchall()]
    
    # Get users with type and subtype
    cursor.execute(
        """SELECT DISTINCT userId FROM events 
           WHERE userId IS NOT NULL AND type = ? AND subType = ? 
           ORDER BY userId""",
        (event_type, event_subtype)
    )
    filtered_users = [row[0] for row in cursor.fetchall()]
    
    assert len(filtered_users) > 0, f"Database should have users with {event_type}/{event_subtype}"
    assert len(filtered_users) <= len(type_users), "Subtype filter should narrow or maintain results"
    
    # All filtered users should be in type users
    for user_id in filtered_users:
        assert user_id in type_users, f"User {user_id} should be in type filter results"


def test_get_events_by_user(real_db):
    """Test getting events for a specific user."""
    cursor = real_db.cursor()
    
    # Get a user with seizure events
    cursor.execute(
        """SELECT DISTINCT userId FROM events 
           WHERE type = 'Seizure' AND subType = 'Tonic-Clonic' 
           LIMIT 1"""
    )
    result = cursor.fetchone()
    
    if not result:
        pytest.skip("No Tonic-Clonic seizure events in database")
    
    user_id = result[0]
    
    # Get events for this user
    cursor.execute(
        """SELECT id, userId, dataTime, type, subType 
           FROM events 
           WHERE type = 'Seizure' AND subType = 'Tonic-Clonic' AND userId = ?
           ORDER BY dataTime LIMIT 5""",
        (user_id,)
    )
    events = cursor.fetchall()
    
    assert len(events) > 0, f"User {user_id} should have Tonic-Clonic events"
    
    # Verify all events belong to the user
    for event in events:
        assert event['userId'] == user_id
        assert event['type'] == 'Seizure'
        assert event['subType'] == 'Tonic-Clonic'


def test_cascading_filter_reduces_results(real_db):
    """Test that cascading filters progressively narrow results."""
    cursor = real_db.cursor()
    
    # Get all users
    cursor.execute("SELECT COUNT(DISTINCT userId) as count FROM events WHERE userId IS NOT NULL")
    all_user_count = cursor.fetchone()['count']
    
    # Get users with Seizure events
    cursor.execute(
        """SELECT COUNT(DISTINCT userId) as count FROM events 
           WHERE userId IS NOT NULL AND type = 'Seizure'"""
    )
    seizure_user_count = cursor.fetchone()['count']
    
    assert seizure_user_count <= all_user_count, "Type filter should reduce or maintain count"
    
    # Get users with Seizure + Tonic-Clonic
    cursor.execute(
        """SELECT COUNT(DISTINCT userId) as count FROM events 
           WHERE userId IS NOT NULL AND type = 'Seizure' AND subType = 'Tonic-Clonic'"""
    )
    filtered_count = cursor.fetchone()['count']
    
    assert filtered_count <= seizure_user_count, "Subtype filter should reduce or maintain count"


def test_user_filter_with_no_results(real_db):
    """Test user filter that should return no results."""
    cursor = real_db.cursor()
    
    # Try to get users with a non-existent type
    cursor.execute(
        """SELECT DISTINCT userId FROM events 
           WHERE userId IS NOT NULL AND type = 'NonexistentType'"""
    )
    users = cursor.fetchall()
    
    assert len(users) == 0, "Non-existent type should return no users"


def test_multiple_subtypes_for_user(real_db):
    """Test that users can have multiple event subtypes."""
    cursor = real_db.cursor()
    
    # Find a user with multiple subtypes
    cursor.execute(
        """SELECT userId, COUNT(DISTINCT subType) as subtype_count
           FROM events
           WHERE userId IS NOT NULL AND type = 'Seizure'
           GROUP BY userId
           HAVING subtype_count > 1
           LIMIT 1"""
    )
    result = cursor.fetchone()
    
    if not result:
        pytest.skip("No users with multiple seizure subtypes in database")
    
    user_id = result['userId']
    
    # Get all seizure subtypes for this user
    cursor.execute(
        """SELECT DISTINCT subType FROM events
           WHERE userId = ? AND type = 'Seizure'
           ORDER BY subType""",
        (user_id,)
    )
    subtypes = [row['subType'] for row in cursor.fetchall()]
    
    assert len(subtypes) > 1, f"User {user_id} should have multiple seizure subtypes"


def test_events_ordered_by_time(real_db):
    """Test that events are returned in chronological order."""
    cursor = real_db.cursor()
    
    # Get events for first available user
    cursor.execute("SELECT DISTINCT userId FROM events WHERE userId IS NOT NULL LIMIT 1")
    result = cursor.fetchone()
    
    if not result:
        pytest.skip("No users in database")
    
    user_id = result[0]
    
    # Get events ordered by time
    cursor.execute(
        """SELECT id, dataTime FROM events 
           WHERE userId = ?
           ORDER BY dataTime
           LIMIT 10""",
        (user_id,)
    )
    events = cursor.fetchall()
    
    # Verify ordering
    for i in range(len(events) - 1):
        assert events[i]['dataTime'] <= events[i + 1]['dataTime'], \
            "Events should be ordered chronologically"


@pytest.mark.parametrize("event_type", ["Seizure", "False Alarm", "Fall"])
def test_type_specific_user_lists(real_db, event_type):
    """Test getting users for different event types."""
    cursor = real_db.cursor()
    
    cursor.execute(
        """SELECT DISTINCT userId FROM events 
           WHERE userId IS NOT NULL AND type = ?
           ORDER BY userId""",
        (event_type,)
    )
    users = [row[0] for row in cursor.fetchall()]
    
    # May return empty list if no events of this type, which is valid
    if len(users) > 0:
        # Verify each user has at least one event of this type
        for user_id in users:
            cursor.execute(
                "SELECT COUNT(*) as count FROM events WHERE userId = ? AND type = ?",
                (user_id, event_type)
            )
            count = cursor.fetchone()['count']
            assert count > 0, f"User {user_id} should have {event_type} events"
