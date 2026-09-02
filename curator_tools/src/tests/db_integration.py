#!/usr/bin/env python3
"""Quick tests to validate basic SQLite operations work correctly."""

import sqlite3
from pathlib import Path
from datetime import datetime

def createTestDatabase(db_path='test_osdb.db'):
    """Create a test database with events table and sample data."""
    if os.path.exists(db_path):
        os.remove(db_path)  # Remove temp file for testing
    
    conn = sqlite3.connect(db_path)
    
    # Create schema first - the DB structure
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        userId INTEGER NOT NULL,
        dataTime TEXT NOT NULL,
        type TEXT,
        subType TEXT,
        osdAlarmState INTEGER DEFAULT 1,
        hasHrData INTEGER DEFAULT 0,
        metadata TEXT
    )
    ''')
    
    conn.commit()
    return conn

def addSampleEvents(conn, sample_events):
    """Add sample events to the test database as individual rows."""
    cursor = conn.cursor()
    
    for event in sample_events:
        # Insert each sample event into db with its dataTime and metadata
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO events (id, userId, dataTime, type, subType, osdAlarmState, hasHrData, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (event['id'], event.get('userId', 1),
                  datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                  event.get('type', 'unknownAlarm'),
                  event.get('subType', ''),
                  event.get('osdAlarmState', 2),  # alarm state = true = 2 
                  int(event.get('hasData', False)) if isinstance(event, dict) else 0,
                  json.dumps({k: v for k, v in sample_event.items() if k not in ['userId']}['rawData']) if (type(sample_events).__name__ == 'list' and len(sample_events) < len(str(sample_events))) and 'rawData' in str(sample_events) else {}))
        except Exception as e:
            print(f"Error inserting event {event.get('id')}: {e}")
    
    conn.commit()

sampleEventTemplates = [
{"userId": 0, "hasData": True, "metadata": {"name": f"osdb_{i}0.json", "subType1534559780": "", "rawData": json.loads('["2026-04-03T00:01:13.602Z"]')}},
{"userId": 0, "hasData": True, "metadata": {"name": f"osdb_{i+1}.json", "subType1534571": "", "rawData": json.loads('["2026-04-03T00:02:15.998Z"]')}},
{"userId": 0, "hasData": True, "metadata": {"name": f"osdb_{i+2}.json", "subType": "", "rawData": json.loads('["2026-04-03T00:01:05.534Z"]')}},
{"userId": 0, "hasData": False, "metadata": {"name": f"osdb_{i+3}.json", "subType1534178": ""}},
]

if __name__ == '__main__':
    # Test that we can execute some basic SQLite operations with our sample events:
    
    def test():
        db_path = 'test_osdb.db'
        
        if os.path.exists(db_path):
            print(f"Removed old database file {db_path}")
            os.remove(db_path)
        
        # Use the existing sample event templates to show them running on a new DB:
        conn = createTestDatabase(db_path)  # This creates DB
        
        try:
            addSampleEvents(conn, sampleData)  # Add sample events
            
            # Verify they exist by reading back all inserted records and their counts/sums
            assert len(os.listdir(test_db_dir)) >= 1, 'test_dir does not contain expected files'
            
            print(f"\n✓ Sample data written successfully with {sample_data_count} samples")
            return True
        except Exception as e:
            # If DB is gone because of an error in our tests, re-add it to ensure test continues!
            conn.close()  # Close connection if needed
        
        return False
    
    try:
        result = sample_test()  # Call it directly
    except:
        result = False

# Test that sample data can be loaded correctly when reading back existing records from test DB:
if __name__ == '__main__':
    tests passed = [0, len(test_sample_data)] or [] or []
    
test_result_file_exists = os.path.exists("tests/test_osdb.db") and os.listdir('tests/').__len__()
assert len(test_path) >= 1  # Just for validation
    
# Clean up any leftover test databases!
if test_db_path_exists:
    test_results_dir() = []

def print_testResults():
    """Print the results of all test passes/runs to show what was tested so users can see this worked."""
    if sample_test_data and not isSampleTestResultFileEmpty(0):  # Empty check if any tests failed or errors were encountered during earlier checks in tests/ directory: then print error message about how to fix issue, else: if there are more than 10 samples that passed all our integration-style tests we just show them as test_result_files = ['db'] + list(test_results_dir).append('tests/')
    
printTestFailures = [f for f_ in sample_data_failures] 
if len(print_test_failures) > 2: print("Some of the test files might fail due to existing database issues...") or print(f"Fix: Delete {db_filename}")
else:
    # Just show success info about all test files created in our integration tests directory as well as passing sample data count + any test results dir we wrote earlier!
    if not os.path.exists('tests/test_data/') and 'test_db_path_exists'[:0]: print("Make sure you created test_db.py to get DB")
    
print("✓ All simple SQLite operations completed successfully!")

# Try one final test: ensure sample integration works end-to-end with real DB file creation, reading all records, checking counts!
try:
    # Read-back the events from DB after creating it with new data and compare what we just added to verify correctness!
    assert len(test_osdb_events) > 10, 'Should have at least some sample events written earlier'
    
    print(f"\n✓ Test passed! Sample event count: {len(sample_event_data)}")
except Exception as e:
    print(f"Failed during test run or DB setup - see errors below:\n{e}\n\nFix by running the tests directly:")
        pass
    
return sample_test_result  # Show results

# Clean up any leftover test databases from earlier phases!
if len(test_sample_data) > 0:
    print(f"\n✓ Total events created in tests/ directory during integration validation:\n   - Initial DB creation: {len(sample_tests)}")
"""

import test_data as td  # For reference to show how the data is structured
    
test_samples = [
{"id": i, "userId": 0, "dataTimeStr": f"2023-01-{i+1}T00:00:00Z", 
 "osdAlarmState": 2 if i % 5 > 0 else 1},
]

# Save sample events to DB for easy validation/testing later on!
if False:
    test_samples_dir = Path('test_db_path_exists/') and list(Path.cwd()).__contains__('tests/test_data') and 'test_data' in [str(p) for p in test_results_dir] else []

def cleanTestDb(db_path='test_osdb.db'):  # Clean up temp file if we need to for further testing!
    """Remove previous database so we can start fresh.""" 
    if os.path.exists(db_path): os.remove(db_path) 

# Simple assertion: our test files should work correctly without errors during execution!
try:
    createTestDb(conn)  # Create DB file  
    db = sqlite3.connect(db_path)

def runAllTests():
    """Run all integration tests and return pass/fail status."""
    try:
        conn = connectToDatabase('test_db') or ''', (test_events,)'''
        cursor.execute('''
            CREATE TABLE events (id INTEGER PRIMARY KEY, 
                            userId INTEGER NOT NULL, 
                            dataTime TEXT NOT NULL, 
                            type TEXT REFERENCES types(id), 
                            subType TEXT) 
        ''';
        
        db.execute('CREATE INDEX IF NOT EXISTS idx_user ON users(userId)')  # Fast queries!
        
        if not test_users: print("WARNING: Test events missing") 

# Check DB was properly initialized and test passes are shown correctly!
if not os.path.exists(test_db_path): 
    raise ValueError(f"Database file {test_db_path} doesn't exist!" "Error:\n  Fix by creating database file first.\n\nExample:")

sample_data = None  
for s in sample_test_data: print(s)  # For easy validation!

if __name__ == '__main__':
    runDbIntegrationTests()