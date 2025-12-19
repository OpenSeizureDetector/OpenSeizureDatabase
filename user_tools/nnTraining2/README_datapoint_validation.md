# Datapoint Validation Feature

## Overview

The `flattenData.py` script now includes optional datapoint validation to ensure temporal continuity in event data. This feature detects and handles common data quality issues that can occur due to connectivity problems, software bugs, or transmission errors.

## Usage

## Configuration

### In runSequence Pipeline

Add the following to your `nnConfig.json`:

```json
{
    "dataProcessing": {
        ...
        "simpleMagnitudeOnly": true,
        "validateDatapoints": true,
        ...
    }
}
```

**Files Updated:**
- `nnConfig_deep_pytorch.json` ✓
- `nnConfig_deep.json` ✓
- `nnConfig_deep_run.json` ✓

When running `runSequence.py`, the validation will automatically:
- Read the `validateDatapoints` setting from config
- Pass it to `flattenData.flattenOsdb()` 
- Report gaps and overlaps during the flattening stage
- Fill gaps with zero-filled datapoints
- Skip overlapping datapoints

### Standalone Usage

Enable datapoint validation by adding the `--validate-datapoints` flag:

```bash
python flattenData.py -i input.json -o output.csv --validate-datapoints
```

Without this flag or config setting, the script operates in backward-compatible mode, processing all datapoints without validation.

## Features

### 1. Gap Detection and Filling

**Problem**: Missing datapoints due to connectivity issues or data loss.

**Solution**: Automatically detects gaps in the temporal sequence and fills them with zero-filled datapoints.

**Example**:
```
Original datapoints:
  - 02:37:25 (sample 0-124)
  - 02:37:45 (sample 0-124)  ← 15 second gap!

After validation:
  - 02:37:25 (original data)
  - 02:37:30 (zero-filled)
  - 02:37:35 (zero-filled)
  - 02:37:40 (zero-filled)
  - 02:37:45 (original data)
```

**Output**:
```
Event 100 (user 1) has data issues:
  Gap #1: 15040ms (3 missing datapoints)
```

### 2. Overlap Detection

**Problem**: Duplicate or overlapping datapoints from retransmissions or buffering issues.

**Solution**: Detects when datapoints overlap in time and skips the duplicates.

**Example**:
```
Original datapoints:
  - 03:00:10 (covers 03:00:05.04 to 03:00:10)
  - 03:00:12 (covers 03:00:07.04 to 03:00:12)  ← Overlaps!
  - 03:00:17 (covers 03:00:12.04 to 03:00:17)

After validation:
  - 03:00:10 (kept)
  - 03:00:12 (skipped - overlaps previous)
  - 03:00:17 (kept)
```

**Output**:
```
Event 101 (user 2) has data issues:
  Overlap #1: 2960ms - skipping datapoint
```

### 3. Robust Date/Time Parsing

**Problem**: Data may contain different date/time formats from various sources.

**Solution**: Automatically detects and parses multiple common date/time formats:

- `DD-MM-YYYY HH:MM:SS` (e.g., "09-05-2022 02:37:25")
- `YYYY-MM-DD HH:MM:SS` (e.g., "2022-05-09 02:37:25")
- `DD/MM/YYYY HH:MM:SS` (e.g., "09/05/2022 02:37:25")
- `YYYY/MM/DD HH:MM:SS` (e.g., "2022/05/09 02:37:25")
- All formats with milliseconds (`.%f` suffix)

Invalid timestamps are skipped with a warning.

## Technical Details

### Datapoint Timing

Each datapoint contains 125 samples at 25Hz, spanning exactly 5 seconds:
- Sample frequency: 25Hz
- Samples per datapoint: 125
- Sample interval: 40ms
- Datapoint duration: 5000ms (5 seconds)

**Important**: The `dataTime` field represents the timestamp of the **last sample** (sample 124), not the first sample.

### Gap Tolerance

The validation uses a 100ms tolerance to account for minor timing jitter:
- Gaps < 100ms: Treated as continuous
- Gaps ≥ 100ms: Filled with zero-filled datapoints
- Overlaps > 100ms: Datapoint is skipped

### Zero-Filled Datapoint Structure

Gap-filling creates complete datapoint records with:
- `rawData`: Array of 125 zeros
- `rawData3D`: Array of 125 `[0, 0, 0]` arrays
- `hr`: None
- `maxVal`, `minVal`, `maxFreq`, `specPower`, `roiPower`: 0
- `alarmState`: 0
- `alarmPhrase`: ""
- `dataTime`: Calculated timestamp for the gap position

## Testing

Run the comprehensive test suite:

```bash
python tests/test_flattenData_validation.py
```

This tests:
- ✓ Gap detection and filling
- ✓ Overlap detection and skipping
- ✓ Multiple date/time format parsing
- ✓ Backward compatibility (no validation)

## Example Output

### With Validation Enabled

```bash
$ python flattenData.py -i data.json -o output.csv --validate-datapoints

Event 100 (user 1) has data issues:
  Gap #1: 15040ms (3 missing datapoints)

Event 101 (user 2) has data issues:
  Overlap #1: 2960ms - skipping datapoint
  Gap #1: 2040ms (0 missing datapoints)
```

### Without Validation (Default)

```bash
$ python flattenData.py -i data.json -o output.csv

# No validation messages, processes all datapoints as-is
```

## When to Use Validation

**Use validation when**:
- Processing real-world data from wearable devices
- Data may have connectivity issues
- You need clean, continuous time series for training
- Temporal integrity is important for your analysis

**Don't use validation when**:
- Processing already-validated test data
- You want to preserve original data exactly as-is
- You need to analyze data quality issues themselves
- Backward compatibility with existing pipelines is required

## Performance Impact

Validation adds minimal overhead:
- Parses timestamps for each datapoint
- Sorts datapoints by time (O(n log n))
- Single-pass validation and gap filling

For typical event sizes (10-100 datapoints), the impact is negligible.
