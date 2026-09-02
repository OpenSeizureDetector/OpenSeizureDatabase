# makeOsdDb_refactored_wrapper.py - Feature Documentation

## Overview

The refactored wrapper script (`makeOsdDb_refactored_wrapper.py`) now includes full feature parity with the original `makeOsdDb.py`, including:

1. **Event Download and Processing** - Core functionality to fetch and process events
2. **CSV Index File Generation** - Generate index CSV files from JSON event files
3. **Summary Graph Generation** - Generate visual summaries of database content

## Usage Examples

### Basic Usage (Event Processing Only)

```bash
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /home/graham/osd/osdb
```

This will:
- Download events from the web API
- Process events using refactored modules
- Save JSON files to the specified directory

### With Index File Generation

```bash
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /home/graham/osd/osdb \
    --generate-index
```

This will additionally:
- Generate CSV index files for each JSON file (e.g., `osdb_3min_allSeizures.csv`)
- Index files contain event metadata in tabular format

### With Summary Graph Generation

```bash
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /home/graham/osd/osdb \
    --generate-graphs
```

This will additionally:
- Generate summary graphs in `osdb-dir/output/` directory
- Creates charts showing:
  - Summary statistics (total events by type)
  - Seizures per user (bar chart)
  - Cumulative seizures over time (line chart)

### Complete Usage (All Features)

```bash
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /home/graham/osd/osdb \
    --generate-index \
    --generate-graphs \
    --graph-output /home/graham/osd/osdb/graphs \
    --graph-threshold 10
```

This enables all post-processing features with custom options.

## Command-Line Options

### Core Options

- `--config <file>`: Path to osdb.cfg configuration file (default: `../osdb.cfg`)
- `--osdb-dir <dir>`: Output directory for OSDB files (required)
- `--start <date>`: Start date for data extraction (yyyy-mm-dd format)
- `--end <date>`: End date for data extraction (yyyy-mm-dd format)
- `--debug`: Enable debug output

### Post-Processing Options

- `--generate-index`: Generate CSV index files from JSON files
- `--generate-graphs`: Generate summary graphs from JSON files
- `--graph-output <dir>`: Custom output directory for graphs (default: `osdb-dir/output`)
- `--graph-threshold <n>`: Minimum events per user for individual display (default: 5)

## Generated Files

### JSON Event Files (always generated)

- `osdb_3min_allSeizures.json` - All seizure events
- `osdb_3min_tcSeizures.json` - Tonic-clonic seizures only
- `osdb_3min_fallEvents.json` - Fall events
- `osdb_3min_falseAlarms.json` - False alarm events
- `osdb_3min_ndaEvents.json` - Normal daily activity events

### CSV Index Files (with --generate-index)

- `osdb_3min_allSeizures.csv` - Index of all seizure events
- `osdb_3min_tcSeizures.csv` - Index of tonic-clonic seizures
- `osdb_3min_fallEvents.csv` - Index of fall events
- `osdb_3min_falseAlarms.csv` - Index of false alarms
- `osdb_3min_ndaEvents.csv` - Index of NDA events

Each CSV contains columns like:
- id, userId, dataTime, type, subType, osdAlarmState, dataSource, desc

### Summary Graphs (with --generate-graphs)

Generated in `osdb-dir/output/` (or custom directory):

1. `summary_statistics.png` - Bar chart of total events by type
2. `seizures_by_user.png` - Bar chart of seizure events per user
3. `cumulative_seizures_per_month.png` - Line chart showing cumulative seizures over time

## Comparison with Original makeOsdDb.py

### Feature Parity

| Feature | Original makeOsdDb.py | Refactored Wrapper | Notes |
|---------|----------------------|-------------------|-------|
| Event download | ✓ | ✓ | Same web API logic |
| Event filtering | ✓ | ✓ | Same filtering rules |
| Event grouping | ✓ | ✓ | Refactored modules |
| JSON output | ✓ | ✓ | Same format |
| CSV index files | ✓ | ✓ | Via --generate-index |
| Summary graphs | ✓ (via subcommand) | ✓ | Via --generate-graphs |

### Differences

1. **Command Structure**: Original uses subcommands (`update`, `graphs`), refactored uses flags
2. **Processing**: Refactored uses Phase 1-5 modules for validation, normalization, deduplication
3. **Create Mode**: Refactored doesn't have separate `--create` flag (always updates existing files)

## Integration Notes

The refactored wrapper can be used as a drop-in replacement for `makeOsdDb.py update` with the addition of post-processing flags:

```bash
# Original:
python3 makeOsdDb.py update --config osdb.cfg
python3 makeIndex.py osdb/osdb_3min_allSeizures.json
python3 makeOsdDb.py graphs osdb/*.json --output output

# Refactored (equivalent):
python3 makeOsdDb_refactored_wrapper.py \
    --config osdb.cfg \
    --osdb-dir osdb \
    --generate-index \
    --generate-graphs
```

## Troubleshooting

### Index files not generated

- Ensure JSON files exist in the osdb-dir before running with `--generate-index`
- Check that libosd.osdDbConnection module is properly installed

### Graphs not generated

- Ensure matplotlib is installed (`pip install matplotlib`)
- Check that JSON files contain valid event data
- Use `--debug` flag to see detailed error messages

### Missing events

- Verify osdb.cfg has correct API credentials
- Check date filters (--start/--end) if specified
- Review excludeDataSources/includeDataSources in config
