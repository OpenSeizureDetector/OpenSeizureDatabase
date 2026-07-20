# Quick Start Guide - OSDB Graph Generation

## What Was Implemented

A new script for generating summary graphs from OSDB JSON event files. The script creates three publication-quality visualizations:

1. **Summary Statistics** - Total counts of seizures, false alarms, and NDA events
2. **Seizures by User** - Bar chart of seizure contributions per user
3. **Cumulative Seizures per Month** - Time-series trend showing user contributions over time

Users contributing fewer seizures than a configurable threshold are automatically grouped as "Other" for cleaner visualizations.

## Quick Start

### Option 1: Standalone Usage (Recommended for most users)

```bash
# Activate environment
source /home/graham/pyEnvs/osdb/bin/activate
cd /home/graham/osd/OpenSeizureDatabase/curator_tools

# Generate graphs from JSON files
python generateGraphs.py your_file.json --output output_dir --threshold 5
```

### Option 2: Integrated with makeOsdDb.py

```bash
# Use as a subcommand
python makeOsdDb.py graphs your_file.json --output output_dir --threshold 5
```

## Common Commands

```bash
# Basic - use defaults
python generateGraphs.py data.json

# Custom output directory
python generateGraphs.py data.json --output reports

# Custom threshold (group users with <3 seizures as "Other")
python generateGraphs.py data.json --threshold 3

# Multiple files
python generateGraphs.py file1.json file2.json file3.json

# Debug mode (verbose output)
python generateGraphs.py data.json --debug

# All options together
python generateGraphs.py data.json --output reports --threshold 10 --debug
```

## Output Files

The script creates three PNG files in the output directory:

- `summary_statistics.png` - Bar chart of event type totals
- `seizures_by_user.png` - Bar chart of seizures per user
- `cumulative_seizures_per_month.png` - Line chart showing trends

All files are 300 DPI and suitable for publication.

## Key Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--output` | `-o` | `output` | Directory for output graphs |
| `--threshold` | `-t` | `5` | Min events per user before grouping as "Other" |
| `--debug` | | False | Print detailed processing information |

## Examples

```bash
# Process seizure events only, group users with <10 seizures
python generateGraphs.py osdb_3min_allSeizures.json --threshold 10

# Process all event types together
python generateGraphs.py osdb_3min_*.json --output summary_graphs

# Create graphs in current directory with low threshold
python generateGraphs.py events.json --output . --threshold 2
```

## Help

For detailed information:

```bash
python generateGraphs.py --help
python makeOsdDb.py graphs --help
```

For comprehensive documentation:

See [GRAPHS_README.md](GRAPHS_README.md)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

Make sure you've activated the correct environment:
```bash
source /home/graham/pyEnvs/osdb/bin/activate
```

### "No events loaded from files"

Check that:
- File paths are correct
- JSON files contain event objects with required fields:
  - `type` - event type (seizure, false alarm, nda)
  - `userId` - user identifier
  - `dataTime` - timestamp in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)

### Graphs look empty or have no data

This typically happens if:
- The threshold is too high for your dataset
- Events are categorized with non-standard type values
- Try lowering the threshold: `--threshold 1`

## Next Steps

1. Review [GRAPHS_README.md](GRAPHS_README.md) for detailed documentation
2. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
3. See [GRAPH_EXAMPLES.sh](GRAPH_EXAMPLES.sh) for more usage examples

## Integration

The `generateGraphs` module can also be imported and used programmatically:

```python
import generateGraphs

success = generateGraphs.generate_all_graphs(
    json_files=['file1.json', 'file2.json'],
    output_dir='output',
    threshold=5,
    debug=False
)
```

## Features

✓ Works with both individual and multiple JSON files
✓ Automatic user grouping for cleaner visualizations
✓ Publication-quality 300 DPI output
✓ Robust error handling (skips invalid files/events)
✓ Standalone or integrated CLI usage
✓ Configurable thresholds
✓ Monthly time-series aggregation
✓ Comprehensive help documentation
