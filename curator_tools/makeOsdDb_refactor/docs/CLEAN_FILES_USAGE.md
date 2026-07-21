# Processing Existing Files with clean_existing_files.py

## Purpose

The `clean_existing_files.py` utility applies Phase 2 processing to existing OSDB JSON files:
- **Deduplication**: Removes duplicate events (same event downloaded multiple times)
- **Grouping & Merging**: Groups events within time threshold and merges their datapoints
- **Output**: Generates cleaned files with reduced event count and merged datapoints

## Quick Start

```bash
# Basic usage - process a file with defaults
python3 clean_existing_files.py osdb_3min_falseAlarms.json

# Dry run to see statistics without writing output
python3 clean_existing_files.py osdb_3min_falseAlarms.json --dry-run

# Custom output path
python3 clean_existing_files.py input.json -o cleaned_output.json

# Disable datapoint concatenation (just group, don't merge datapoints)
python3 clean_existing_files.py input.json --no-concatenate

# Skip deduplication step
python3 clean_existing_files.py input.json --no-dedup

# Use different time threshold (default: 3min)
python3 clean_existing_files.py input.json --time-threshold 5min
```

## Default Behavior

- **Deduplication Method**: `hash` (based on id, userId, dataTime, type)
- **Keep Strategy**: `most_datapoints` (keeps event with most data)
- **Time Threshold**: `3min` (180 seconds)
- **Datapoint Concatenation**: `enabled` (merges datapoints from grouped events)
- **Output File**: `<input>_cleaned.json` (e.g., `input_cleaned.json`)

## Example Output

```
============================================================
Processing: test_data/time_boundary_cases.json
Output: test_data/time_boundary_cases_cleaned.json
============================================================

Loading events...
Loaded 18 events

=== Step 1: Deduplication (method: hash) ===
  Input: 18 events
  Found: 0 duplicates
  Removed: 0 events
  Output: 18 unique events

=== Step 2: Grouping & Merging (threshold: 3min) ===
Selecting from groups: 100%|███████████████| 11/11 [00:00<00:00, 1004.36group/s]
  Input: 18 events
  Groups: 11 groups identified
  Output: 11 merged events
  Datapoints: 504 → 427

============================================================
SUMMARY:
  Input events:        18
  After deduplication: 18 (-0)
  After grouping:      11 (-7 total)
  Datapoints:          504 → 427
  Reduction:           38.9%
============================================================
```

## File Format Support

The utility handles two JSON formats:

1. **Plain array** (standard OSDB files):
   ```json
   [
     {"id": 1, "userId": 100, "dataTime": "...", ...},
     {"id": 2, "userId": 100, "dataTime": "...", ...}
   ]
   ```

2. **Wrapped format** (test data files):
   ```json
   {
     "description": "...",
     "events": [
       {"id": 1, "userId": 100, "dataTime": "...", ...},
       {"id": 2, "userId": 100, "dataTime": "...", ...}
     ]
   }
   ```

## CLI Options

```
positional arguments:
  input                 Input JSON file path

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output JSON file path (default: input_cleaned.json)
  --time-threshold TIME_THRESHOLD
                        Time threshold for grouping (e.g., "3min", "180s") (default: 3min)
  --no-concatenate      Disable datapoint concatenation
  --no-dedup            Skip deduplication step
  --dedup-method {hash,id}
                        Deduplication method (default: hash)
  --keep {first,last,most_datapoints}
                        Which duplicate to keep (default: most_datapoints)
  --dry-run             Show statistics without writing output file
  --no-progress         Hide progress bars
```

## Use Cases

### 1. Clean Up Existing Database Files

Remove duplicates and merge close events in production OSDB files:

```bash
python3 clean_existing_files.py ../output/osdb_3min_falseAlarms.json
python3 clean_existing_files.py ../output/osdb_3min_allSeizures.json
python3 clean_existing_files.py ../output/osdb_3min_tcSeizures.json
```

### 2. Test Different Time Thresholds

Experiment with different grouping thresholds:

```bash
python3 clean_existing_files.py data.json --time-threshold 1min --dry-run
python3 clean_existing_files.py data.json --time-threshold 3min --dry-run
python3 clean_existing_files.py data.json --time-threshold 5min --dry-run
```

### 3. ID-Based Deduplication Only

If you only want to remove events with identical IDs (not hash-based):

```bash
python3 clean_existing_files.py data.json --dedup-method id --no-concatenate
```

### 4. Batch Processing

Process multiple files:

```bash
for file in ../output/osdb_3min_*.json; do
    echo "Processing $file..."
    python3 clean_existing_files.py "$file" --dry-run
done
```

## Implementation Details

The utility uses the Phase 2 modules:
- `event_deduplication.remove_duplicate_events()` for deduplication
- `event_grouping.apply_sliding_window_grouping()` for grouping and merging
- `event_grouping.concatenate_datapoints()` for datapoint merging

It handles string datetime values (e.g., "02-10-2022 13:44:56") by parsing them to timestamps for comparison.

## Safety

- **Non-destructive**: Creates new `_cleaned.json` files, never overwrites input
- **Dry-run mode**: Test processing without writing output
- **Progress bars**: Visual feedback during processing
- **Statistics**: Detailed report of changes made

## Next Steps

After cleaning existing files, you can:
1. Compare cleaned vs original files
2. Validate the cleaned data
3. Replace original files with cleaned versions
4. Integrate into main workflow for new downloads
