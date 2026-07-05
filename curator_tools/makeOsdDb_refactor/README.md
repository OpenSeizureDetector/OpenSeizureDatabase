# makeOsdDb - OpenSeizureDatabase Builder

This tool creates an anonymized database of unique seizure-like events by downloading event data from an OpenSeizureDetector API server, processing and grouping nearby events, and publishing the results as JSON files.

## Overview

The makeOsdDb tool:
- Downloads seizure events from an OSD API server
- Groups events that occur close together in time (default: 3 minutes)
- Merges duplicate datapoints from overlapping events
- Validates and normalizes event data
- Excludes specified data sources (e.g., Phone, AndroidWear)
- Generates separate files for different event types (allSeizures, tcSeizures, fallEvents, etc.)
- Preserves existing published event IDs during updates

## Quick Start

```bash
cd /path/to/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor

# Download and process events, output to /home/graham/osd/osdb
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb

# With date range filter
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb \
    --start 2025-01-01 --end 2025-12-31

# With custom config file
python3 makeOsdDb_refactored_wrapper.py --config /path/to/osdb.cfg \
    --osdb-dir /home/graham/osd/osdb

# Enable debug output
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb --debug
```

## Command Line Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--osdb-dir` | **Yes** | - | Output directory for OSDB JSON files |
| `--config` | No | `../osdb.cfg` | Path to osdb.cfg configuration file |
| `--start` | No | None | Start date for filtering events (yyyy-mm-dd) |
| `--end` | No | None | End date for filtering events (yyyy-mm-dd) |
| `--debug` | No | False | Enable debug output to console |

## Configuration Files

### osdb.cfg

Main configuration file specifying processing parameters:

```ini
[EventTypes]
eventTypes = allSeizures, tcSeizures, fallEvents, falseAlarms, ndaEvents

[Processing]
groupingPeriod = 3  # Minutes between events to consider them separate
excludeDataSources = Phone, AndroidWear  # Data sources to exclude

[Output]
# ... other settings ...
```

**Key Settings:**
- `groupingPeriod`: Time window (minutes) for grouping nearby events
- `excludeDataSources`: Comma-separated list of data sources to exclude
- `eventTypes`: Event types to process and publish

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
├── README.md                           # This file
├── FOLDER_ORGANIZATION.md              # Detailed structure guide
├── makeOsdDb_refactored_wrapper.py     # Main tool (production)
├── clean_existing_files.py             # Utility for cleaning old data
├── publish_osdb.py                     # Publishing utility
├── CLEAN_FILES_USAGE.md                # Usage guide for clean utility
│
├── src/                                # Core implementation modules
│   ├── event_grouping.py               # Sliding window grouping
│   ├── event_validation.py             # Event validation
│   ├── event_deduplication.py          # Datapoint deduplication
│   ├── datetime_normalization.py       # Datetime normalization
│   ├── osdb_sqlite.py                  # SQLite database (experimental)
│   └── osdb_publication.py             # Multi-format export (experimental)
│
├── tests/                              # Unit tests
│   └── [test files]
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
│   ├── generate_comparison_report.py   # Comparison generator
│   └── comparison_results/             # Generated reports
│
└── archives/                           # Historical test data
    ├── README.md                       # Archive index
    └── [archived test results]
```

## Key Features

### 1. Event Grouping (Sliding Window)

Events occurring within `groupingPeriod` minutes are grouped and merged:
- Uses sliding window algorithm (not fixed bins)
- Events can chain together if each consecutive pair is < 3 minutes apart
- Preserves existing published event IDs when updating the database
- Adds "Includes data from merged event(s): X, Y, Z" to description field

**Example:** Events at 10:00, 10:02, 10:05 → All grouped together
- 10:00 to 10:02 = 2 min (grouped)
- 10:02 to 10:05 = 3 min (grouped due to chaining)
- Total span: 5 minutes

### 2. Datapoint Deduplication

When merging events with overlapping time ranges:
- Identifies duplicate datapoints within 100ms time tolerance
- Removes duplicates, keeping unique datapoints
- Typically removes 10-15% duplicates in merged events
- Averages +26 datapoints per merge after deduplication

### 3. Event Type Handling

- **allSeizures**: All seizure events (grouped)
- **tcSeizures**: Tonic-clonic seizures (grouped)
- **fallEvents**: Fall events (grouped)
- **falseAlarms**: False alarm events (grouped)
- **ndaEvents**: NDA events (**not grouped** - contiguous data expected)

### 4. Event ID Preservation

When updating an existing database:
- Existing published events are prioritized during grouping
- If a new event is grouped with an existing event, the existing ID is preserved
- Metadata tracks which events were merged (`_merged_from_event_ids`)
- Description field updated to document merges

### 5. Data Exclusion

- Excludes events from specified data sources (e.g., Phone, AndroidWear)
- Configured via `excludeDataSources` in osdb.cfg
- Applied before grouping

## Output Files

The tool generates JSON files in the specified output directory:

```
osdb_3min_allSeizures.json      # All seizure events
osdb_3min_tcSeizures.json       # Tonic-clonic seizures
osdb_3min_fallEvents.json       # Fall events
osdb_3min_falseAlarms.json      # False alarms
osdb_3min_ndaEvents.json        # NDA events (not grouped)
```

Each file contains an array of event objects with:
- `id`: Unique event ID
- `dataTime`: Event timestamp
- `type`: Event type
- `subType`: Event subtype
- `userId`: Anonymized user ID
- `datapoints`: Array of accelerometer datapoints
- `desc`: Event description (includes merge information)
- Additional metadata fields

## Utilities

### clean_existing_files.py

Removes existing event files before processing to ensure clean regeneration.

See [CLEAN_FILES_USAGE.md](CLEAN_FILES_USAGE.md) for detailed usage.

```bash
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

1. **Output Format**: The production tool outputs JSON files directly (no SQLite database)
2. **Database Updates**: Existing published event IDs are always preserved when updating the database
3. **NDA Events**: NDA events are NOT grouped (contiguous data expected)
4. **Datapoint Merging**: Duplicate datapoints are automatically removed during merging
5. **Configuration**: Always verify osdb.cfg settings before running
6. **Credentials**: Ensure client.cfg contains valid API credentials

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
