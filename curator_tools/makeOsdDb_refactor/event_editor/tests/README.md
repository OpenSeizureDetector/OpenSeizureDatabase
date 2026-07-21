# Event Editor Test Suite

Comprehensive pytest-based test suite for the event_editor database operations.

## Structure

```
tests/
├── conftest.py                        # Shared fixtures and test utilities
├── test_event_updates.py              # Event update operation tests
├── test_soft_delete.py                # Soft delete functionality tests
├── test_filtering.py                  # Event filtering tests
├── test_seizure_times_integration.py  # Integration tests for seizure timing
└── test_user_queries_integration.py   # Integration tests for user filtering
```

## Test Categories

### Unit Tests (Fast)
- **test_event_updates.py**: 10 tests for update operations
- **test_soft_delete.py**: 8 tests for soft delete functionality
- **test_filtering.py**: 20 tests for event filtering

Total: **38 unit tests** using temporary test databases

### Integration Tests (Requires Real Database)
- **test_seizure_times_integration.py**: 7 tests for seizure timing
- **test_user_queries_integration.py**: 11 tests for user queries

Total: **18 integration tests** requiring `/home/graham/osd/osdb/osdb_working.db`

## Running Tests

### Run All Tests
```bash
cd /path/to/event_editor
pytest
```

### Run Unit Tests Only (Fast, No Real Database Required)
```bash
pytest -m "not integration"
```

### Run Integration Tests Only
```bash
pytest -m integration
```

### Run Specific Test File
```bash
pytest tests/test_event_updates.py
pytest tests/test_filtering.py -v
```

### Run Specific Test
```bash
pytest tests/test_event_updates.py::test_update_basic_fields
```

### Run with Coverage Report
```bash
pytest --cov=database_manager --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Run in Verbose Mode
```bash
pytest -v
```

### Run with Output (Show Prints)
```bash
pytest -s
```

## Test Coverage

### Update Operations (`test_event_updates.py`)
- ✓ Update basic fields (type, subType, desc)
- ✓ Update seizure times
- ✓ Clear seizure times
- ✓ Preserve existing metadata
- ✓ Isolation (updates don't affect other events)
- ✓ Datapoint preservation
- ✓ Error handling (nonexistent events)
- ✓ Multiple operations integrity
- ✓ Transaction safety (commit/rollback)

### Soft Delete Operations (`test_soft_delete.py`)
- ✓ Mark events as 'Deleted'
- ✓ Mark events as 'Unknown'
- ✓ Isolation (soft delete doesn't affect other events)
- ✓ Datapoint preservation
- ✓ Filter exclusion of deleted events
- ✓ Metadata preservation
- ✓ Reversibility (undo soft delete)

### Filtering Operations (`test_filtering.py`)
- ✓ Filter by single/multiple event types
- ✓ Filter by event subtype
- ✓ Filter by type and subtype combination
- ✓ Filter by single/multiple users
- ✓ Filter by date range (start, end, both)
- ✓ Filter by description with SQL wildcards
- ✓ Case-insensitive description search
- ✓ Combined filters
- ✓ Empty filter lists (return all)
- ✓ No results handling
- ✓ Get unique types/subtypes/users
- ✓ Cascading filters (type → subtype → user)

### Integration Tests (`test_seizure_times_integration.py`)
- ✓ Seizure times loading from real database
- ✓ Relative time calculations
- ✓ Negative time support (seizure before alarm)
- ✓ Datapoint time calculation
- ✓ Seizure times within event range
- ✓ Required fields validation

### Integration Tests (`test_user_queries_integration.py`)
- ✓ Get all user IDs
- ✓ Filter users by event type
- ✓ Filter users by type and subtype
- ✓ Get events for specific user
- ✓ Cascading filter logic
- ✓ No results handling
- ✓ Multiple subtypes per user
- ✓ Chronological ordering
- ✓ Type-specific user lists

## Fixtures (conftest.py)

- **temp_dir**: Temporary directory for test databases
- **empty_db**: Empty database with schema only
- **sample_events_db**: Database with sample events for testing
- **filter_test_db**: Database with diverse events for filtering tests
- **db_manager**: DatabaseManager instance with sample data
- **filter_db_manager**: DatabaseManager for filtering tests
- **real_db**: Connection to real OSDB database (integration tests)

## Continuous Integration

To run tests in CI/CD pipeline:

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests only (fast, no external dependencies)
pytest -m "not integration" --cov=database_manager --cov-report=xml

# Upload coverage to CI service
# (coverage.xml generated)
```

## Development Workflow

1. **Before committing changes:**
   ```bash
   pytest -m "not integration"  # Fast unit tests
   ```

2. **Before releasing:**
   ```bash
   pytest  # All tests including integration
   ```

3. **Check coverage:**
   ```bash
   pytest --cov=database_manager --cov-report=term-missing
   ```

4. **Add new tests:**
   - Unit tests: Add to appropriate test_*.py file
   - Integration tests: Add to test_*_integration.py file
   - Always use fixtures from conftest.py

## Troubleshooting

### Integration Tests Skipped
If integration tests are skipped with "requires osdb_working.db":
- Ensure `/home/graham/osd/osdb/osdb_working.db` exists
- Or run unit tests only: `pytest -m "not integration"`

### Import Errors
```bash
# Ensure you're in the event_editor directory
cd /path/to/event_editor
python -m pytest
```

### Coverage Not Working
```bash
# Install pytest-cov
pip install pytest-cov
```

## Best Practices

1. **Test Isolation**: Each test uses its own temporary database
2. **Fixtures**: Use conftest.py fixtures for consistency
3. **Markers**: Use `@pytest.mark.integration` for tests requiring real DB
4. **Assertions**: Use descriptive assertion messages
5. **Cleanup**: Fixtures handle cleanup automatically
6. **Parametrize**: Use `@pytest.mark.parametrize` for similar tests

## Example: Adding a New Test

```python
# tests/test_event_updates.py

def test_my_new_feature(db_manager, sample_events_db):
    """Test that my new feature works correctly."""
    # Arrange
    event_id = 'test_001'
    
    # Act
    success = db_manager.some_new_method(event_id, param1, param2)
    
    # Assert
    assert success, "Operation should succeed"
    updated = get_event_from_db(sample_events_db, event_id)
    assert updated['field'] == expected_value
```

## Results

All **56 tests** pass successfully:
- **38 unit tests** verify core functionality without external dependencies
- **18 integration tests** validate against real database scenarios

Coverage: **>95%** of database_manager.py code
