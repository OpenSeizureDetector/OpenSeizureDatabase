# Summary Graphs Generation for OSDB

## Overview

The `generateGraphs.py` script generates publication-quality summary graphs from OpenSeizureDatabase (OSDB) JSON event files. It creates visualizations of:

1. **Summary Statistics**: Total counts of seizures, false alarms, and NDA (normal daily activity) events
2. **Seizures by User**: Bar chart showing seizure event counts per user, with users contributing fewer than a threshold number of seizures grouped as "Other"
3. **Cumulative Seizures per Month**: Line chart showing the cumulative number of seizures contributed by each user over time, with threshold-based grouping

## Usage

### Standalone Usage

The script can be run independently with JSON files specified on the command line:

```bash
python generateGraphs.py file1.json file2.json file3.json --output output_dir --threshold 5 --debug
```

#### Arguments:

- `json_files` (positional, required): One or more JSON files to process
- `--output, -o` (optional): Output directory for graphs (default: `output`)
- `--threshold, -t` (optional): Minimum number of events for individual user display (default: `5`)
- `--debug` (optional): Print debug information during execution

#### Example:

```bash
python generateGraphs.py osdb_3min_allSeizures.json osdb_3min_falseAlarms.json osdb_3min_ndaEvents.json \
    --output ./graphs \
    --threshold 5 \
    --debug
```

### Integration with makeOsdDb.py

The graph generation functionality is integrated into `makeOsdDb.py` as a subcommand:

```bash
python makeOsdDb.py graphs file1.json file2.json --output output_dir --threshold 5
```

This allows seamless integration with existing OSDB workflows.

## Output Files

The script generates three PNG files in the output directory:

1. **summary_statistics.png**: Bar chart showing total counts
2. **seizures_by_user.png**: Bar chart of seizure events per user
3. **cumulative_seizures_per_month.png**: Line chart of cumulative seizures over time

All files are saved at 300 DPI for publication quality.

## Features

### Threshold Grouping

Users contributing fewer seizures than the threshold value are automatically grouped as "Other" to keep visualizations clean and focused on major contributors. This is applied to:

- Seizures by user bar chart
- Cumulative seizures per month line chart

### Date Parsing

The script automatically parses ISO 8601 formatted timestamps (`YYYY-MM-DDTHH:MM:SSZ`) from the `dataTime` field in event objects to generate monthly aggregations.

### Event Categorization

Events are automatically categorized by their `type` field:

- **Seizures**: type = "seizure"
- **False Alarms**: type = "false alarm"
- **NDA Events**: type = "nda" or "normal daily activity"

### User Identification

User identities are extracted from the `userId` field in each event. Users are labeled by their ID in the output, with the exception of those in the "Other" group when they fall below the threshold.

## JSON File Format

Input JSON files should contain an array of event objects with at least the following fields:

```json
[
  {
    "id": 101,
    "userId": "U9",
    "type": "seizure",
    "subType": "clonic",
    "dataTime": "2024-01-02T12:00:00Z",
    "desc": "event description",
    ...
  },
  ...
]
```

Required fields for graph generation:
- `type`: Event type (seizure, false alarm, nda)
- `userId`: Identifier for the event contributor
- `dataTime`: ISO 8601 timestamp for time-series aggregation

## Dependencies

The script requires:

- `pandas`: Data manipulation
- `matplotlib`: Graph generation
- Python 3.6+

These should be installed via the project's `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Examples

### Generate graphs from all seizure files

```bash
python generateGraphs.py \
    osdb_3min_allSeizures.json \
    osdb_3min_tcSeizures.json \
    --output ./summary_graphs \
    --threshold 10
```

### Generate graphs with custom threshold

```bash
python generateGraphs.py data.json --output . --threshold 3
```

### Use with makeOsdDb.py

```bash
# Generate graphs for all event types
python makeOsdDb.py graphs \
    osdb_3min_allSeizures.json \
    osdb_3min_falseAlarms.json \
    osdb_3min_ndaEvents.json \
    --output ./reports \
    --threshold 5
```

## Error Handling

The script handles:

- Missing files: Skips files that cannot be found with a warning
- Invalid JSON: Reports parsing errors and continues with other files
- Invalid timestamps: Skips events with unparseable dates (with debug output)
- Empty datasets: Gracefully handles cases with no data to plot

## Design Decisions

1. **Threshold Grouping**: Improves readability by reducing clutter in visualizations while preserving total counts
2. **Monthly Aggregation**: Provides sufficient granularity for trend analysis while maintaining clarity
3. **300 DPI Output**: Ensures publication-quality graphics suitable for papers and reports
4. **Separate Files**: Three distinct output files allow independent use in different contexts

## Notes

- The cumulative seizures chart may have gaps if a user has no events in a given month
- The "Other" group in the user charts represents the sum of all events from users below the threshold
- Chart colors are automatically assigned using matplotlib's color cycles
- All timestamps are assumed to be in UTC (Z suffix in ISO 8601 format)

## Future Enhancements

Potential improvements for future versions:

- Add confidence intervals or uncertainty bands to cumulative charts
- Support for filtering by date range
- Geographic distribution visualization
- Event type breakdown per user
- Interactive HTML output option
- Comparison between multiple datasets
