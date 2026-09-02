# Datetime Normalization for OSDB Files

## Issue

OSDB database contains events with two different datetime formats:
- **Old format**: `"DD-MM-YYYY HH:MM:SS"` (e.g., `"02-10-2022 13:44:56"`)
- **New format**: `"YYYY-MM-DDTHH:MM:SSZ"` (e.g., `"2024-07-12T05:58:24Z"`) - ISO 8601 UTC

This inconsistency makes it difficult to:
- Parse dates correctly
- Sort events chronologically
- Compare timestamps across events
- Process data in other tools

## Solution

The `datetime_normalization.py` module standardizes all datetime fields to ISO 8601 format.

### Features

- **Automatic format detection**: Recognizes both old and new formats
- **Event-level normalization**: Converts `dataTime` field in events
- **Datapoint-level normalization**: Converts `dataTime`, `time`, and `t` fields in datapoints
- **Batch processing**: Handles lists of events with progress bars
- **Format reporting**: Shows statistics on formats found and converted

### Usage

```python
from datetime_normalization import normalize_event_datetimes, normalize_events_batch

# Single event
event = {'id': 123, 'dataTime': '02-10-2022 13:44:56', 'datapoints': [...]}
normalized = normalize_event_datetimes(event, normalize_datapoints=True)
# Result: {'id': 123, 'dataTime': '2022-10-02T13:44:56Z', 'datapoints': [...]}

# Batch of events
events = [...]  # List of events
normalized_events, stats = normalize_events_batch(events, show_progress=True)
print(f"Normalized {stats['events_normalized']} events")
print(f"Normalized {stats['datapoints_normalized']} datapoints")
```

### Format Detection

```python
from datetime_normalization import detect_datetime_formats

formats = detect_datetime_formats(events)
# Returns: {'iso_8601': 25, 'old_format': 12, 'other': 0, 'missing': 0}
```

### Test Results

Test data file (`time_boundary_cases.json`) with 18 events:
- **Before**: 12 events in old format (`DD-MM-YYYY HH:MM:SS`)
- **After**: All 18 events in ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`)
- **Datapoints**: 504 datapoint timestamps normalized

Sample conversion:
```
'02-10-2022 13:44:56' → '2022-10-02T13:44:56Z'
'04-05-2022 15:33:56' → '2022-05-04T15:33:56Z'
'2024-07-12T05:58:24Z' → '2024-07-12T05:58:24Z' (unchanged)
```

## Integration with clean_existing_files.py

The datetime normalization will be integrated into the `clean_existing_files.py` utility as a processing step:

1. **Load events** from JSON file
2. **Detect formats** (report old vs new)
3. **Remove duplicates** (Phase 2 deduplication)
4. **Normalize datetimes** → **NEW STEP**
5. **Group and merge** events (Phase 2 grouping)
6. **Save cleaned** output

This ensures all output files have consistent, ISO 8601-compliant datetime formats.

## Command Line Usage (Future)

```bash
# With datetime normalization (default)
python3 clean_existing_files.py input.json

# Skip datetime normalization  
python3 clean_existing_files.py input.json --no-normalize-datetimes

# Dry run to see format statistics
python3 clean_existing_files.py input.json --dry-run
```

## Benefits

1. **Consistency**: All datetime fields use the same format
2. **Standards compliance**: ISO 8601 is an international standard
3. **Better sorting**: ISO 8601 sorts correctly lexicographically
4. **Tool compatibility**: Most tools expect ISO 8601 format
5. **Timezone clarity**: Explicit UTC timezone (`Z` suffix)
6. **Future-proof**: New data already uses this format

## Next Steps

1. ✅ Datetime normalization module created and tested
2. ⏭️ Integrate into `clean_existing_files.py` utility
3. ⏭️ Add unit tests for edge cases (timezone handling, malformed dates)
4. ⏭️ Process existing OSDB files to normalize all timestamps
5. ⏭️ Update makeOsdDb.py to ensure new downloads use ISO 8601
