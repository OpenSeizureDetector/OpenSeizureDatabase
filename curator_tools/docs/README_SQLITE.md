# makeOsdDb SQLite Refactor - README

## Overview

This directory contains the **refactored makeOsdDb implementation** that uses **SQLite as primary storage** with JSON export as a separate publication step.

**Status**: ✅ **Production Ready** (2026-07-20)

## Key Features

### 1. SQLite Primary Storage
- Events stored in SQLite database (not JSON)
- Fast queries on event attributes
- Efficient indexing and filtering
- ~25% smaller than equivalent JSON

### 2. Separate Publication Step
- **Update mode**: Download → Process → Save to SQLite
- **Publish mode**: Export SQLite → JSON files for distribution
- Flexibility to edit/review before publishing

### 3. Safety Features
- **Automatic backups**: Timestamped backups before destructive operations
- **CASCADE DELETE**: Datapoints automatically deleted with events
- **Transaction safety**: All operations atomic (no partial updates)
- **Schema versioning**: Future migrations supported

### 4. Event Management CLI
- **manage_events.py**: Command-line tool for event operations
- Show, list, edit, delete events
- Database statistics and validation
- See [QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md)

### 5. Data Integrity
- **No data loss**: All JSON fields mapped to SQLite
- **Foreign key constraints**: Referential integrity enforced
- **Validation tools**: Check database integrity
- **Comprehensive tests**: 29 tests passing

## Quick Start

### 1. Update Database (Download Events)

```bash
# Download events from server and save to SQLite
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /path/to/osdb
```

Default database location: `{osdb-dir}/osdb_working.db`

### 2. Manage Events

```bash
# Show database statistics
python3 manage_events.py stats --db /path/to/osdb_working.db

# List recent events
python3 manage_events.py list --db /path/to/osdb_working.db --limit 20

# Edit event metadata
python3 manage_events.py edit \
    --db /path/to/osdb_working.db \
    --event-id 12345 \
    --field type \
    --value "False Alarm"

# Delete event (with backup and confirmation)
python3 manage_events.py delete --db /path/to/osdb_working.db --event-id 12345
```

See [QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md) for detailed usage.

### 3. Publish to JSON

```bash
# Export database to JSON files
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /path/to/osdb \
    --publish

# With index files and graphs
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /path/to/osdb \
    --publish \
    --generate-index \
    --generate-graphs
```

## Directory Structure

```
makeOsdDb_refactor/
├── src/
│   ├── init_database.py          # Import JSON → SQLite
│   ├── osdb_sqlite.py             # SQLite working database class
│   ├── database_utils.py          # Utility functions (backup, delete, validate)
│   ├── event_validation.py        # Event validation
│   ├── event_grouping.py          # Event grouping/merging
│   ├── event_deduplication.py     # Duplicate removal
│   └── datetime_normalization.py  # Datetime parsing
├── tests/
│   ├── test_database_utils.py     # Database utility tests (18 tests)
│   └── test_wrapper_integration.py # Integration tests (11 tests)
├── manage_events.py               # Event management CLI
├── makeOsdDb_refactored_wrapper.py # Main wrapper with SQLite integration
├── SCHEMA_ANALYSIS.md             # JSON vs SQLite schema comparison
├── QUICKSTART_EVENT_MANAGEMENT.md # User guide for event management
├── IMPLEMENTATION_STATUS.md       # Implementation progress tracking
├── SQLITE_INTEGRATION_COMPLETE.md # Integration summary
└── README.md                      # This file
```

## Configuration

### Database Path

Add to `osdb.cfg`:

```json
{
    "osdbDir": "/path/to/osdb",
    "groupingPeriod": "3min",
    "databasePath": "/path/to/osdb_working.db",
    ...
}
```

**Precedence**: CLI `--database` > config file `databasePath` > default `{osdbDir}/osdb_working.db`

### Command-Line Arguments

```bash
# Core arguments
--config PATH          # Configuration file (default: ../osdb.cfg)
--osdb-dir PATH        # Output directory (required)
--database PATH        # SQLite database path (optional)

# Mode selection
--publish              # Publish mode (export SQLite → JSON)

# Optional processing
--generate-index       # Generate CSV index files
--generate-graphs      # Generate summary graphs
--start YYYY-MM-DD     # Start date for data
--end YYYY-MM-DD       # End date for data
--debug                # Enable debug output
```

## Testing

### Run All Tests

```bash
# Database utility tests (18 tests)
python3 tests/test_database_utils.py

# Integration tests (11 tests)
python3 tests/test_wrapper_integration.py
```

**Expected output**: All 29 tests passing ✅

### Test Coverage

- **Backup system**: Creation, preservation, listing
- **Safe deletion**: CASCADE DELETE, transactions
- **Metadata updates**: Field validation, type changes
- **Database validation**: Integrity checks, orphan detection
- **Statistics**: Count, date range, size
- **Integration**: Save/load, filtering, incremental updates, publish workflow

## Migration from JSON

Import existing JSON files:

```bash
cd src

# Import all JSON files from a directory
python3 init_database.py \
    --json-dir /path/to/existing/json/files \
    --db /path/to/osdb_working.db
```

## Documentation

- **[SCHEMA_ANALYSIS.md](SCHEMA_ANALYSIS.md)** - Comprehensive field-by-field comparison between JSON and SQLite schemas
- **[QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md)** - User guide with practical examples for event management
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Implementation progress and test results
- **[SQLITE_INTEGRATION_COMPLETE.md](SQLITE_INTEGRATION_COMPLETE.md)** - Detailed summary of SQLite integration

## Benefits

### Query Capabilities
```sql
-- Fast queries on event attributes
SELECT COUNT(*) FROM events WHERE type = 'Seizure';

-- Complex filtering
SELECT id, userId, dataTime FROM events 
WHERE type = 'Seizure' AND subType = 'Tonic-Clonic' 
  AND datapoint_count >= 10
ORDER BY dataTime DESC LIMIT 20;
```

### Data Integrity
- Foreign key constraints ensure referential integrity
- Transactions prevent partial updates
- Schema versioning supports future migrations

### Performance
- Indexing provides fast lookups
- Incremental updates (only download new events)
- Efficient storage (~25% smaller than JSON)

### Safety
- Automatic backups before destructive operations
- Rollback capability via backups
- Validation tools to check integrity

## Troubleshooting

### Database locked error

```bash
# Check what's using the database
lsof osdb_working.db

# Or wait longer
sqlite3 osdb_working.db -cmd ".timeout 10000"
```

### Restore from backup

```bash
# List backups
python3 src/database_utils.py list-backups --db osdb_working.db

# Restore (copy backup over current)
cp osdb_working.db.backup.20260720_143025 osdb_working.db

# Validate
python3 manage_events.py validate --db osdb_working.db
```

### Foreign key constraint failed

Foreign keys are now automatically enabled in all operations. This error indicates a data integrity issue that should be investigated.

## Known Limitations

1. **Event type filtering**: Currently loads all events and filters in Python (could add database-level filtering for large databases)
2. **Memory usage**: Publish mode loads all events into memory (may need batching for 10,000+ events)
3. **Migration script**: No automated migration from old workflow (manual import required)

## Future Enhancements

- Database-level event type filtering
- Batch export for large databases
- Automated migration script
- Web interface for event browsing/editing
- RESTful API for programmatic access

## Support

For issues or questions:
1. Check documentation in this directory
2. Review test files for usage examples
3. Run validation: `python3 manage_events.py validate --db your_database.db`
4. Check database statistics: `python3 manage_events.py stats --db your_database.db`

## License

See [LICENCE.txt](LICENCE.txt) for licensing information.

## Version History

- **Version 1.0** (2026-07-20) - Initial release with SQLite integration
  - SQLite as primary storage
  - Separate publish mode
  - Event management CLI
  - Automatic backups
  - CASCADE DELETE
  - Schema versioning
  - 29 tests passing
  - Full documentation
