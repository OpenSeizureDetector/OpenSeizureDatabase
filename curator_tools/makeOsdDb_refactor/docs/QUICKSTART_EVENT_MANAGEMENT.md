# Quick Start Guide: Event Management with SQLite

## Overview

The refactored makeOsdDb now uses SQLite as primary storage, with JSON export as a separate publication step. This provides:
- Fast queries and filtering
- Safe editing of event metadata
- Transactional safety (no corruption on crashes)
- Automatic backups before destructive operations

## Installation

No additional dependencies required beyond standard Python 3 libraries.

## Basic Workflow

### 1. Import Existing JSON Data to SQLite

```bash
cd curator_tools/makeOsdDb_refactor/src

# Import JSON files to create/update database
python3 init_database.py \
    --json-dir /path/to/existing/json/files \
    --db /path/to/osdb_working.db
```

Example output:
```
✓ Imported 1,234 total events from 5 JSON files
```

### 2. View Database Statistics

```bash
cd curator_tools/makeOsdDb_refactor

# Show database stats
python3 manage_events.py stats --db /path/to/osdb_working.db
```

Example output:
```
Database Statistics: /path/to/osdb_working.db
============================================================
Schema Version:       1
Total Events:         1,234
Total Datapoints:     15,678
Avg Datapoints/Event: 12.7
Date Range:           2020-01-01T00:00:00Z to 2023-12-31T23:59:59Z
Database Size:        45.2 MB

Events by Type:
  Seizure              856
  False Alarm          234
  Fall                  89
  Unknown               55
============================================================
```

### 3. List Events

```bash
# List all seizures (most recent first, limit 20)
python3 manage_events.py list --db osdb_working.db --type Seizure --limit 20

# List events for specific user
python3 manage_events.py list --db osdb_working.db --user-id 42 --limit 10
```

Example output:
```
ID       User   Date                 Type            SubType         DPs   Description
------------------------------------------------------------------------------------------------------------------------
12345    42     2023-12-15T14:30:00Z Seizure         Tonic-Clonic    15    User-reported seizure
12344    42     2023-12-10T09:15:00Z False Alarm     (none)          8     False positive
12343    43     2023-12-09T18:45:00Z Seizure         Other           12    Night-time event
```

### 4. View Event Details

```bash
# Show full details of a single event
python3 manage_events.py show --db osdb_working.db --event-id 12345
```

Example output:
```
============================================================
Event ID: 12345
============================================================
User ID:          42
Type:             Seizure
SubType:          Tonic-Clonic
Description:      User-reported seizure at home
Data Time:        2023-12-15T14:30:00Z
Data Time End:    2023-12-15T14:32:30Z
Duration:         150.0 seconds
Alarm State:      2
Alarm Phrase:     ALARM

Data Source:      Watch
Phone App Ver:    1.2.3
Watch SD Ver:     2.1.0
Watch FW Ver:     1.5.4
Watch Name:       Garmin Vivoactive
Battery:          85%

Data Availability:
  HR Data:        Yes
  O2Sat Data:     No
  3D Accel Data:  Yes

Datapoints:       15 (stored: 15)
Seizure Times:    10.5s - 140.2s
Merged From:      3 events: [12343, 12344, 12345]

Last Modified:    2023-12-16T10:00:00Z
============================================================
```

### 5. Edit Event Metadata

```bash
# Change event type
python3 manage_events.py edit \
    --db osdb_working.db \
    --event-id 12345 \
    --field type \
    --value "False Alarm"

# Update description
python3 manage_events.py edit \
    --db osdb_working.db \
    --event-id 12345 \
    --field desc \
    --value "Updated: Actually a false alarm after review"

# Set manual seizure times (in seconds from event start)
python3 manage_events.py edit \
    --db osdb_working.db \
    --event-id 12345 \
    --field seizureTimes \
    --value "[10.5, 140.2]"

# Skip automatic backup (faster, but risky)
python3 manage_events.py edit \
    --db osdb_working.db \
    --event-id 12345 \
    --field type \
    --value "Seizure" \
    --no-backup
```

Example output:
```
✓ Backup created: osdb_working.db.backup.20260720_143025
✓ Updated event 12345: type = False Alarm

============================================================
Event ID: 12345
============================================================
[... updated event details ...]
```

### 6. Delete Events

```bash
# Delete single event (with confirmation prompt)
python3 manage_events.py delete \
    --db osdb_working.db \
    --event-id 12345

# Delete multiple events
python3 manage_events.py delete \
    --db osdb_working.db \
    --event-ids 12345,12346,12347

# Force delete without confirmation (dangerous!)
python3 manage_events.py delete \
    --db osdb_working.db \
    --event-id 12345 \
    --force

# Delete without backup (not recommended)
python3 manage_events.py delete \
    --db osdb_working.db \
    --event-id 12345 \
    --no-backup
```

Example output:
```
⚠️  WARNING: About to delete 1 event(s)
   Event IDs: [12345]

Are you sure? (yes/no): yes

✓ Backup created: osdb_working.db.backup.20260720_143025
✓ Deleted 1 event(s) and 15 datapoint(s)
```

### 7. Validate Database

```bash
# Check database integrity
python3 manage_events.py validate --db osdb_working.db
```

Example output (healthy database):
```
Database: osdb_working.db
Valid: ✓ YES

✓ No issues found!
```

Example output (with issues):
```
Database: osdb_working.db
Valid: ✗ NO

Issues found:
  Found 3 orphaned datapoint(s)
  Foreign key violations found: 2
    - datapoints(event_id=999) references non-existent event
    - datapoints(event_id=1000) references non-existent event
  Warning: 5 event(s) have no datapoints
```

### 8. Manage Backups

```bash
# View all backups
python3 src/database_utils.py list-backups --db osdb_working.db
```

Example output:
```
Backups for osdb_working.db:
  2023-12-16 14:30:25 - 45.2 MB - osdb_working.db.backup.20231216_143025
  2023-12-16 10:15:42 - 45.1 MB - osdb_working.db.backup.20231216_101542
  2023-12-15 18:00:00 - 44.9 MB - osdb_working.db.backup.20231215_180000
```

```bash
# Create manual backup
python3 src/database_utils.py backup --db osdb_working.db

# Backup to custom directory
python3 src/database_utils.py backup \
    --db osdb_working.db \
    --backup-dir /path/to/backup/directory
```

### 9. Database Utilities

```bash
# Get statistics
python3 src/database_utils.py stats --db osdb_working.db

# Validate database
python3 src/database_utils.py validate --db osdb_working.db
```

## Common Tasks

### Correcting a Mis-classified Event

```bash
# View the event
python3 manage_events.py show --db osdb.db --event-id 12345

# Change from "Seizure" to "False Alarm"
python3 manage_events.py edit --db osdb.db --event-id 12345 \
    --field type --value "False Alarm"
```

### Adding Manual Seizure Time Boundaries

```bash
# After reviewing video/data, set precise seizure times
# Format: [start_seconds, end_seconds] from event start time
python3 manage_events.py edit --db osdb.db --event-id 12345 \
    --field seizureTimes --value "[15.5, 145.8]"
```

### Bulk Deletion of Test Events

```bash
# First, list events from test user
python3 manage_events.py list --db osdb.db --user-id 999 --limit 100

# Then delete them (copy IDs from list)
python3 manage_events.py delete --db osdb.db \
    --event-ids 10001,10002,10003,10004,10005
```

### Finding Events with Specific Characteristics

```bash
# SQLite makes this easy with direct queries
sqlite3 osdb_working.db <<EOF
SELECT id, userId, dataTime, type, subType, desc 
FROM events 
WHERE type = 'Seizure' 
  AND subType = 'Tonic-Clonic' 
  AND has3dData = 1 
  AND datapoint_count >= 10
ORDER BY dataTime DESC
LIMIT 20;
EOF
```

## Safety Features

### Automatic Backups

All destructive operations (`edit`, `delete`) automatically create timestamped backups:
```
osdb_working.db.backup.20260720_143025
osdb_working.db.backup.20260720_144532
osdb_working.db.backup.20260720_150001
```

Restore from backup:
```bash
cp osdb_working.db.backup.20260720_143025 osdb_working.db
```

### CASCADE DELETE

When you delete an event, all associated datapoints are automatically deleted:
- No orphaned datapoints left in database
- Referential integrity maintained
- Enforced by `PRAGMA foreign_keys = ON`

### Transaction Safety

All operations use SQLite transactions:
- Changes are atomic (all-or-nothing)
- Database remains consistent even if operation fails
- Automatic rollback on errors

### Validation

Regular validation checks:
```bash
python3 manage_events.py validate --db osdb.db
```

Checks for:
- Orphaned datapoints
- Foreign key violations
- Missing required fields
- Events without datapoints

## Editable Fields

Via `manage_events.py edit`:
- `type` - Event type (Seizure, False Alarm, Fall, etc.)
- `subType` - Event subtype (Tonic-Clonic, Other, etc.)
- `desc` - User description
- `osdAlarmState` - Alarm state (0=OK, 1=WARNING, 2=ALARM, 3=FALL)
- `dataTime` - Event start time (ISO 8601)
- `dataTimeEnd` - Event end time (ISO 8601)
- `alarmPhrase` - Alarm phrase text
- `alarmRationale` - Why alarm was raised
- `seizureTimes` - Manual seizure boundaries (JSON array)
- `batteryPc` - Battery percentage

**Non-editable fields** (for data integrity):
- `id` - Event ID (never changes)
- `userId` - User ID (fixed)
- `datapoints` - Use separate tools to edit sensor data
- Statistics (computed from datapoints)

## Troubleshooting

### "Foreign key constraint failed" error

Enable foreign keys:
```python
conn = sqlite3.connect('osdb.db')
conn.execute("PRAGMA foreign_keys = ON")
```

This is now done automatically in all tools.

### "Database is locked" error

Another process has the database open. Close it or:
```bash
# Check what's using the database
lsof osdb_working.db

# Or increase timeout
sqlite3 osdb_working.db -cmd ".timeout 10000"
```

### Restore from Backup

```bash
# List available backups
python3 src/database_utils.py list-backups --db osdb.db

# Copy backup over current database
cp osdb.db.backup.20260720_143025 osdb.db

# Validate restored database
python3 manage_events.py validate --db osdb.db
```

## Advanced: Direct SQL Queries

For power users, you can query the database directly:

```bash
# Open database
sqlite3 osdb_working.db

# Example queries:
sqlite> .schema events
sqlite> SELECT COUNT(*) FROM events WHERE type = 'Seizure';
sqlite> SELECT id, dataTime, desc FROM events WHERE userId = 42 LIMIT 10;
sqlite> .quit
```

See [SCHEMA_ANALYSIS.md](SCHEMA_ANALYSIS.md) for complete schema documentation.
