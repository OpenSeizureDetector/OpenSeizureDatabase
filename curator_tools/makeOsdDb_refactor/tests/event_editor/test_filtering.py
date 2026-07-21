"""
Tests for database filtering operations.

Tests that event filtering works correctly with various combinations.
"""

import pytest


def test_filter_by_single_type(filter_db_manager):
    """Test filtering by single event type."""
    events = filter_db_manager.get_filtered_events(event_types=['Seizure'])
    assert len(events) == 4
    for event in events:
        assert event['type'] == 'Seizure'


def test_filter_by_multiple_types(filter_db_manager):
    """Test filtering by multiple event types."""
    events = filter_db_manager.get_filtered_events(event_types=['Seizure', 'Fall'])
    assert len(events) == 5
    for event in events:
        assert event['type'] in ['Seizure', 'Fall']


def test_filter_by_subtype(filter_db_manager):
    """Test filtering by event subtype."""
    events = filter_db_manager.get_filtered_events(event_subtypes=['Tonic-Clonic'])
    assert len(events) == 3
    for event in events:
        assert event['subType'] == 'Tonic-Clonic'


def test_filter_by_type_and_subtype(filter_db_manager):
    """Test filtering by both type and subtype."""
    events = filter_db_manager.get_filtered_events(
        event_types=['Seizure'],
        event_subtypes=['Tonic-Clonic']
    )
    assert len(events) == 2
    for event in events:
        assert event['type'] == 'Seizure'
        assert event['subType'] == 'Tonic-Clonic'


def test_filter_by_user(filter_db_manager):
    """Test filtering by user ID."""
    events = filter_db_manager.get_filtered_events(user_ids=[1])
    assert len(events) == 4
    for event in events:
        assert event['userId'] == 1


def test_filter_by_multiple_users(filter_db_manager):
    """Test filtering by multiple user IDs."""
    events = filter_db_manager.get_filtered_events(user_ids=[1, 2])
    assert len(events) == 6
    for event in events:
        assert event['userId'] in [1, 2]


def test_filter_by_date_range(filter_db_manager):
    """Test filtering by date range."""
    events = filter_db_manager.get_filtered_events(
        start_date='2024-01-03T00:00:00Z',
        end_date='2024-01-06T00:00:00Z'
    )
    # Should include filter_003, filter_004, filter_005
    assert len(events) == 3
    event_ids = [e['id'] for e in events]
    assert 'filter_003' in event_ids
    assert 'filter_004' in event_ids
    assert 'filter_005' in event_ids


def test_filter_by_description_wildcard(filter_db_manager):
    """Test filtering by description with wildcards."""
    # Test partial match with wildcard
    events = filter_db_manager.get_filtered_events(desc_filter='%seizure%')
    assert len(events) == 4
    
    # Test prefix match
    events = filter_db_manager.get_filtered_events(desc_filter='First%')
    assert len(events) == 1
    assert events[0]['id'] == 'filter_001'
    
    # Test suffix match
    events = filter_db_manager.get_filtered_events(desc_filter='%event')
    assert len(events) == 2  # Fall event and Deleted event


def test_filter_combined(filter_db_manager):
    """Test combining multiple filters."""
    events = filter_db_manager.get_filtered_events(
        event_types=['Seizure'],
        user_ids=[1],
        start_date='2024-01-01T00:00:00Z',
        end_date='2024-01-03T00:00:00Z'
    )
    # Should include filter_001 and filter_002
    assert len(events) == 2
    event_ids = [e['id'] for e in events]
    assert 'filter_001' in event_ids
    assert 'filter_002' in event_ids


def test_filter_no_results(filter_db_manager):
    """Test filter returning no results."""
    events = filter_db_manager.get_filtered_events(event_types=['NonexistentType'])
    assert len(events) == 0


def test_filter_all_events_no_criteria(filter_db_manager):
    """Test that no filter criteria returns all events."""
    events = filter_db_manager.get_filtered_events()
    assert len(events) == 7  # All test events


def test_filter_case_insensitive_description(filter_db_manager):
    """Test that description filter is case-insensitive."""
    # Search with uppercase
    events_upper = filter_db_manager.get_filtered_events(desc_filter='%SEIZURE%')
    # Search with lowercase
    events_lower = filter_db_manager.get_filtered_events(desc_filter='%seizure%')
    # Search with mixed case
    events_mixed = filter_db_manager.get_filtered_events(desc_filter='%SeIzUrE%')
    
    # All should return same results
    assert len(events_upper) == len(events_lower) == len(events_mixed) == 4


def test_filter_empty_lists_returns_all(filter_db_manager):
    """Test that empty filter lists return all events."""
    # Empty lists should be treated as no filter
    events = filter_db_manager.get_filtered_events(
        event_types=[],
        event_subtypes=[],
        user_ids=[]
    )
    assert len(events) == 7


def test_filter_by_date_start_only(filter_db_manager):
    """Test filtering by start date only."""
    events = filter_db_manager.get_filtered_events(start_date='2024-01-05T00:00:00Z')
    # Should include filter_005, filter_006, filter_007
    assert len(events) == 3
    for event in events:
        assert event['dataTime'] >= '2024-01-05T00:00:00Z'


def test_filter_by_date_end_only(filter_db_manager):
    """Test filtering by end date only."""
    events = filter_db_manager.get_filtered_events(end_date='2024-01-03T00:00:00Z')
    # Should include filter_001, filter_002 (before 2024-01-03)
    assert len(events) == 2
    for event in events:
        assert event['dataTime'] < '2024-01-03T00:00:00Z'


def test_filter_chaining_narrows_results(filter_db_manager):
    """Test that adding more filters narrows results."""
    # Get all seizures
    seizures = filter_db_manager.get_filtered_events(event_types=['Seizure'])
    seizure_count = len(seizures)
    
    # Filter by subtype - should be fewer
    tonic_clonic = filter_db_manager.get_filtered_events(
        event_types=['Seizure'],
        event_subtypes=['Tonic-Clonic']
    )
    assert len(tonic_clonic) < seizure_count
    
    # Filter by user - should be fewer or equal
    user_specific = filter_db_manager.get_filtered_events(
        event_types=['Seizure'],
        event_subtypes=['Tonic-Clonic'],
        user_ids=[1]
    )
    assert len(user_specific) <= len(tonic_clonic)


def test_get_event_types(filter_db_manager):
    """Test getting unique event types."""
    types = filter_db_manager.get_event_types()
    assert 'Seizure' in types
    assert 'False Alarm' in types
    assert 'Fall' in types
    assert 'Deleted' in types
    assert len(types) == 4


def test_get_event_subtypes(filter_db_manager):
    """Test getting unique event subtypes."""
    subtypes = filter_db_manager.get_event_subtypes()
    assert 'Tonic-Clonic' in subtypes
    assert 'Absence' in subtypes
    assert 'Focal' in subtypes
    assert 'Movement' in subtypes
    assert 'Detected' in subtypes


def test_get_event_subtypes_filtered_by_type(filter_db_manager):
    """Test getting subtypes filtered by event type."""
    subtypes = filter_db_manager.get_event_subtypes(event_type='Seizure')
    assert 'Tonic-Clonic' in subtypes
    assert 'Absence' in subtypes
    assert 'Focal' in subtypes
    # These shouldn't be in seizure subtypes
    assert 'Movement' not in subtypes
    assert 'Detected' not in subtypes


def test_get_user_ids(filter_db_manager):
    """Test getting unique user IDs."""
    user_ids = filter_db_manager.get_user_ids()
    assert 1 in user_ids
    assert 2 in user_ids
    assert 3 in user_ids
    assert len(user_ids) == 3


def test_get_user_ids_filtered_by_type(filter_db_manager):
    """Test getting user IDs filtered by event type."""
    user_ids = filter_db_manager.get_user_ids(event_type='Fall')
    assert 2 in user_ids
    assert 1 not in user_ids  # User 1 has no Fall events


def test_get_user_ids_filtered_by_type_and_subtype(filter_db_manager):
    """Test getting user IDs filtered by type and subtype."""
    user_ids = filter_db_manager.get_user_ids(
        event_type='Seizure',
        event_subtype='Tonic-Clonic'
    )
    assert 1 in user_ids
    assert 2 in user_ids
    assert 3 not in user_ids  # User 3 has Focal seizure, not Tonic-Clonic
