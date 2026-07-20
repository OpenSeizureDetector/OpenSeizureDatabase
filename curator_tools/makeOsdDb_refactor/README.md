# makeOsdDb - OpenSeizureDatabase Builder

This tool creates an anonymized database of unique seizure-like events by downloading event data from an OpenSeizureDetector API server, processing and grouping nearby events, and storing them in a SQLite database with optional JSON export for publication.

## Overview

The makeOsdDb refactored tool uses **SQLite as primary storage** with a **separate publication step** for maximum flexibility:

**Update Mode (Default)**:
- Downloads seizure events from an OSD API server
- Validates and normalizes event data
- Groups events that occur close together in time (default: 3 minutes)
- Merges duplicate datapoints from overlapping events
- Stores processed events in SQLite database
- Preserves existing event IDs during incremental updates

**Publish Mode (--publish flag)**:
- Exports SQLite database to JSON files for distribution
- Generates separate files for different event types (allSeizures, tcSeizures, fallEvents, etc.)
- Applies data source filtering and invalid event exclusion
- Optionally generates CSV index files and summary graphs

**Key Benefits**:
- **Query capabilities**: Fast SQL queries on event attributes
- **Data integrity**: Foreign key constraints, CASCADE DELETE, transactions
- **Safety features**: Automatic backups before destructive operations
- **Flexible workflow**: Edit/review events before publishing
- **Event management**: CLI tool for viewing, editing, and deleting events

## Quick Start

### 0. Initialize from Existing JSON (First-Time Setup)

If you have existing JSON event files and want to migrate to SQLite:

```bash
# Create SQLite database from existing JSON files
python3 src/init_database.py --json-dir /path/to/json/files \
    --db /home/graham/osd/osdb/osdb_working.db
```

**What it does**:
- Reads all `*.json` files from the specified directory
- Creates a new SQLite database with proper schema
- Imports all events and datapoints
- Handles duplicate event IDs automatically
- Validates and computes statistics (datapoint counts, HR/O2Sat flags)

**Note**: After initialization, use the normal update workflow below to add new events.

### 1. Update Database (Download and Process Events)

```bash
cd /path/to/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor

# Download and process events, save to SQLite database
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb

# With date range filter
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb \
    --start 2025-01-01 --end 2025-12-31

# With custom database path
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb \
    --database /path/to/custom.db

# Enable debug output
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb --debug
```

Default database location: `{osdb-dir}/osdb_working.db`

### 2. Manage Events (Optional)

```bash
# View database statistics
python3 manage_events.py stats --db /home/graham/osd/osdb/osdb_working.db

# List recent events
python3 manage_events.py list --db /home/graham/osd/osdb/osdb_working.db --limit 20

# Show detailed event information
python3 manage_events.py show --db /home/graham/osd/osdb/osdb_working.db --event-id 12345

# Edit event metadata
python3 manage_events.py edit --db /home/graham/osd/osdb/osdb_working.db \
    --event-id 12345 --field type --value "False Alarm"

# Delete event (with automatic backup)
python3 manage_events.py delete --db /home/graham/osd/osdb/osdb_working.db --event-id 12345

# Validate database integrity
python3 manage_events.py validate --db /home/graham/osd/osdb/osdb_working.db
```

See [QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md) for detailed usage.

### 3. Publish to JSON Files

```bash
# Export database to JSON files for distribution
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb --publish

# Publish with index files and graphs
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb \
    --publish --generate-index --generate-graphs
```

## Command Line Options

### makeOsdDb_refactored_wrapper.py

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--osdb-dir` | **Yes** | - | Output directory for database and JSON files |
| `--config` | No | `../osdb.cfg` | Path to osdb.cfg configuration file |
| `--database` | No | `{osdb-dir}/osdb_working.db` | Path to SQLite database |
| `--publish` | No | False | Publish mode: export database to JSON files |
| `--start` | No | None | Start date for filtering events (yyyy-mm-dd) |
| `--end` | No | None | End date for filtering events (yyyy-mm-dd) |
| `--generate-index` | No | False | Generate CSV index files (use with --publish) |
| `--generate-graphs` | No | False | Generate summary graphs (use with --publish) |
| `--graph-output` | No | `{osdb-dir}/output` | Output directory for graphs |
| `--debug` | No | False | Enable debug output to console |

### manage_events.py

| Command | Description |
|---------|-------------|
| `show --event-id ID` | Display detailed event information |
| `list [--type TYPE] [--user-id ID] [--limit N]` | List events with optional filtering |
| `edit --event-id ID --field FIELD --value VALUE` | Modify event metadata |
| `delete --event-id ID [--force]` | Delete event (with confirmation) |
| `stats` | Show database statistics |
| `validate` | Check database integrity |

All commands require: `--db PATH` (path to SQLite database)

## Configuration Files

### osdb.cfg

Main configuration file specifying processing parameters:

```json
{
    "osdbDir": "/home/graham/osd/osdb",
    "groupingPeriod": "3min",
    "databasePath": "/path/to/osdb_working.db",
    "includeWarnings": 1,
    "excludeDataSources": ["Phone", "AndroidWear"],
    "includeDataSources": [],
    "invalidEvents": [12345, 67890],
    "skipElements": ["accMean", "accSd", "updated", "created"],
    "credentialsFname": "../client.cfg"
}
```

**Key Settings:**
- `groupingPeriod`: Time window (e.g., "3min") for grouping nearby events
- `databasePath`: Path to SQLite database (optional, CLI takes precedence)
- `excludeDataSources`: List of data sources to exclude
- `includeDataSources`: List of data sources to include exclusively (empty = all)
- `invalidEvents`: List of event IDs to exclude from publication
- `skipElements`: List of fields to omit from published JSON files
- `credentialsFname`: Path to API credentials file

### client.cfg

API server credentials (symlinked from parent directory):

```ini
[API]
username = your_username
password = your_password
server = https://osdb.example.com
```

**Note:** Use `client.cfg.template` as a starting point.

## Folder Structure

```
makeOsdDb_refactor/
├── README.md                           # This file - main documentation
├── README_SQLITE.md                    # Detailed SQLite integration guide
├── QUICKSTART_EVENT_MANAGEMENT.md      # Event management CLI guide
├── SCHEMA_ANALYSIS.md                  # JSON vs SQLite schema comparison
├── IMPLEMENTATION_STATUS.md            # Implementation progress tracking
├── SQLITE_INTEGRATION_COMPLETE.md      # Integration summary
│
├── makeOsdDb_refactored_wrapper.py     # Main tool - SQLite-based workflow
├── manage_events.py                    # Event management CLI tool
│
├── src/                                # Core implementation modules
│   ├── event_grouping.py               # Sliding window grouping
│   ├── event_validation.py             # Event validation
│   ├── event_deduplication.py          # Datapoint deduplication
│   ├── datetime_normalization.py       # Datetime normalization
│   ├── osdb_sqlite.py                  # SQLite database (primary storage)
│   ├── database_utils.py               # Database utilities (backup, validate, etc.)
│   └── init_database.py                # JSON to SQLite import tool
│
├── tests/                              # Unit and integration tests
│   ├── test_database_utils.py          # Database utility tests (18 tests)
│   └── test_wrapper_integration.py     # Integration tests (11 tests)
│
├── docs/                               # Development documentation
│   ├── README.md                       # Documentation index
│   ├── EVENT_ID_PRESERVATION_FIX.md    # Technical details
│   ├── DATAPOINT_MERGE_EXPLANATION.md  # Merge behavior explanation
│   └── [other documentation]
│
├── validation/                         # Validation scripts (one-off)
│   ├── README.md                       # Validation guide
│   ├── test_event_preservation.py      # Preservation test
│   └── comparison_results/             # Generated reports
│
└── archives/                           # Historical test data
    ├── README.md                       # Archive index
    └── [archived test results]
```

## Key Features

### 1. SQLite Primary Storage

**All events stored in SQLite database** with these benefits:
- **Fast queries**: SQL-based filtering on event attributes
- **Data integrity**: Foreign key constraints, CASCADE DELETE
- **Transactions**: Atomic operations, automatic rollback on errors
- **Schema versioning**: Tracked migrations for future updates
- **Efficient storage**: ~25% smaller than equivalent JSON

**Database Operations**:
```sql
-- Fast queries on event attributes
SELECT COUNT(*) FROM events WHERE type = 'Seizure';

-- Complex filtering
SELECT id, userId, dataTime FROM events 
WHERE type = 'Seizure' AND subType = 'Tonic-Clonic' 
  AND datapoint_count >= 10
ORDER BY dataTime DESC LIMIT 20;
```

### 2. Event Grouping (Sliding Window)

Events occurring within `groupingPeriod` minutes are grouped and merged:
- Uses sliding window algorithm (not fixed bins)
- Events can chain together if each consecutive pair is < 3 minutes apart
- Preserves existing published event IDs when updating the database
- Adds "Includes data from merged event(s): X, Y, Z" to description field

**Ex5mple:** Events at 10:00, 10:02, 10:05 → All grouped together
- 10:00 to 10:02 = 2 min (grouped)
- 10:02 to 10:05 = 3 min (grouped due to chaining)
- Total span: 5 minutes

### 3. Datapoint Deduplication

When merging events with overlapping time ranges:
- Identifies duplicate datapoints within 100ms time tolerance
- Removes duplicates, keeping unique datapoints
- Typically removes 10-15% duplicates in merged events
- Averages +26 datapoints per merge after deduplication

### 4. Event Type Handling

- **allSeizures**: All seizure events (grouped)
- **tcSeizures**: Tonic-clonic seizures (grouped)
- **fallEvents**: Fall events (grouped)
- **falseAlarms**: False alarm events (grouped)
- **ndaEvents**: NDA events (**not grouped** - contiguous data expected)

### 5. Event ID Preservation

When updating an existing database:
- Existing published events are prioritized during grouping
- If a new event is grouped with an existing event, the existing ID is preserved
- Metadata tracks which events were merged (`_merged_from_event_ids`)
- Description field updated to document merges

### 6. Data Exclusion

- Excludes events from specified data sources (e.g., Phone, AndroidWear)
- Configured via `excludeDataSources` in osdb.cfg
- Applied during publish step

### 7. Safety Features

**Automatic Backups**:
- Created before all destructive operations (edit, delete)
- Timestamped format: `database.db.backup.20260720_143025`
- Optional custom backup directory
- Can be disabled with `--no-backup` flag

**CASCADE DELETE**:
- Deleting an event automatically deletes all associated datapoints
- Enforced by foreign key constraints
- Prevents orphaned data

**Database Validation**:
- Check foreign key integrity
- Detect orphaned datapoints
- Verify required fields present
- Run: `manage_events.py validate --db database.db`

### 8. Event Management CLI

Comprehensive command-line interface for database operations:
- **View**: Show detailed event information, list with filtering
- **Edit**: Modify event metadata (type, subType, desc, seizureTimes, etc.)
- **Delete**: Remove events with confirmation and automatic backup
- **Stats**: Database statistics (event counts, date ranges, size)
- **Validate**: Check database integrity

**Editable Fields**: type, subType, desc, osdAlarmState, dataTime, dataTimeEnd, alarmPhrase, alarmRationale, seizureTimes, batteryPc

See [QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md) for detailed guide.

## Utilities

### init_database.py

One-time migration tool to initialize SQLite database from existing JSON files:

```bash
# Basic usage
python3 src/init_database.py --json-dir /path/to/json/files --db osdb.db

# With custom output directory
python3 src/init_database.py --json-dir /path/to/json \
    --db osdb.db --output-dir /path/to/output
```

**Features**:
- Imports all JSON files (`*.json`) from directory
- Creates complete SQLite schema with indexes
- Handles duplicate event IDs across files
- Preserves all event metadata and datapoints
- Validates data integrity during import

**When to use**: First-time migration from JSON-based workflow to SQLite.

### database_utils.py

Database utility functions accessible via command-line:

```bash
# Create manual backup
python3 src/database_utils.py backup --db osdb_working.db

# List all backups
python3 src/database_utils.py list-backups --db osdb_working.db

# Show database statistics
python3 src/database_utils.py stats --db osdb_working.db
```

See [QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md) for more information.

## Testing

### Run All Tests

```bash
# Database utility tests (18 tests)
python3 tests/test_database_utils.py

# Integration tests (11 tests)
python3 tests/test_wrapper_integration.py

# Or use pytest
pytest tests/
```

**Expected**: All 29 tests passing ✅

### Test Coverage

- **Backup system**: Creation, preservation, custom directories, listing
- **Safe deletion**: CASCADE DELETE, transactions, confirmation
- **Metadata updates**: Field validation, type changes, JSON fields
- **Database validation**: Integrity checks, orphan detection
- **SPrimary Storage**: The tool uses **SQLite as primary storage** (not JSON)
2. **Workflow**: Download → Process → SQLite → (optional) Publish to JSON
3. **Database Updates**: Existing event IDs are always preserved when updating the database
4. **NDA Events**: NDA events are NOT grouped (contiguous data expected)
5. **Datapoint Merging**: Duplicate datapoints are automatically removed during merging
6. **Safety Features**: Automatic backups before destructive operations, CASCADE DELETE
7. **Configuration**: Always verify osdb.cfg settings before running
8. **Credentials**: Ensure client.cfg contains valid API credentials
9. **Schema Version**: Database tracks schema version 1 for future migration
- **README.md** (this file) - Main tool documentation
- **[README_SQLITE.md](README_SQLITE.md)** - Detailed SQLite integration guide
- **[QUICKSTART_EVENT_MANAGEMENT.md](QUICKSTART_EVENT_MANAGEMENT.md)** - Event management CLI guide
- **[SCHEMA_ANALYSIS.md](SCHEMA_ANALYSIS.md)** - JSON vs SQLite schema comparison
- Verify database path is writable

**Problem:** Database locked error
- Check if another process is using the database: `lsof osdb_working.db`
- Close other connections to the database
- Increase timeout: `sqlite3 osdb_working.db -cmd ".timeout 10000"`

**Problem:** Unexpected grouping behavior
- Verify `groupingPeriod` in osdb.cfg
- Remember: sliding window allows chaining beyond groupingPeriod
- See [docs/EVENT_ID_PRESERVATION_FIX.md](docs/EVENT_ID_PRESERVATION_FIX.md)

**Problem:** Missing events
- Check `excludeDataSources` in osdb.cfg
- Check `invalidEvents` list in osdb.cfg
- Run validation: `python3 manage_events.py validate --db osdb_working.db`
- Check statistics: `python3 manage_events.py stats --db osdb_working.db`

**Problem:** Foreign key constraint failed
- Foreign keys are automatically enabled
- This indicates a data integrity issue
- Run validation: `python3 manage_events.py validate --db osdb_working.db`

**Problem:** Need to restore from backup
```bash
# List backups
python3 src/database_utils.py list-backups --db osdb_working.db

# Restore (copy backup over current)
cp osdb_working.db.backup.20260720_143025 osdb_working.db

# Validate restored database
python3 manage_events.py validate --db osdb_working.db
``

# Import JSON files from a directory
python3 init_database.py \
    --json-dir /path/to/existing/json/files \
    --db /path/to/osdb_working.db
```
python3 clean_existing_files.py /path/to/osdb
```

### publish_osdb.py (Experimental)

**Optional** format conversion tool for exporting JSON to other formats.

This experimental utility can:
- Convert JSON files to Parquet, CSV, or compressed JSON formats
- Optionally use SQLite database as intermediate format
- Verify data consistency during conversion

**Note:** The main production tool (`makeOsdDb_refactored_wrapper.py`) outputs JSON directly and does **not** use SQLite. This utility is for post-processing and format conversion only.

```bash
# Convert JSON to multiple formats
python3 publish_osdb.py --input osdb_3min_allSeizures.json --all-formats

# Convert to Parquet only
python3 publish_osdb.py --input osdb_3min_allSeizures.json --formats parquet

# Direct conversion without SQLite
python3 publish_osdb.py --input osdb_3min_allSeizures.json --no-database --formats csv
```

**Use Cases:**
- Converting JSON to more efficient formats (Parquet) for analysis
- Creating CSV exports for spreadsheet tools
- Experimenting with SQLite-based workflows
- Format validation and consistency checks

## Documentation

### For Users
- **README.md** (this file) - Main tool documentation
- **CLEAN_FILES_USAGE.md** - Clean utility guide
- **FOLDER_ORGANIZATION.md** - Complete folder structure

### For Developers
See the [docs/](docs/) directory for:
- Implementation details and rationale
- Bug fixes and solutions
- Testing methodology
- Performance characteristics
- Development history

### For Validation
See the [validation/](validation/) directory for:
- One-off validation scripts
- Comparison and analysis tools
- Test results and reports

## Important Notes

1. **First-Time Setup**: If migrating from JSON files, use `src/init_database.py` to create initial SQLite database
2. **SQLite Primary Storage**: This refactored tool uses SQLite as primary storage; publish to JSON for distribution
3. **Database Updates**: Existing published event IDs are always preserved when updating the database
4. **NDA Events**: NDA events are NOT grouped (contiguous data expected)
5. **Datapoint Merging**: Duplicate datapoints are automatically removed during merging
6. **Configuration**: Always verify osdb.cfg settings before running
7. **Credentials**: Ensure client.cfg contains valid API credentials

## Troubleshooting

**Problem:** No events downloaded
- Check client.cfg credentials
- Verify API server is accessible
- Check date range filter (--start/--end)

**Problem:** Unexpected grouping behavior
- Verify `groupingPeriod` in osdb.cfg
- Remember: sliding window allows chaining beyond groupingPeriod
- See [docs/EVENT_ID_PRESERVATION_FIX.md](docs/EVENT_ID_PRESERVATION_FIX.md)

**Problem:** Missing events
- Check `excludeDataSources` in osdb.cfg
- Verify event types in `eventTypes` configuration
- Review validation with `validation/test_event_preservation.py`

## Contact

See main project [README.md](../../README.md) for contact information and license details.
