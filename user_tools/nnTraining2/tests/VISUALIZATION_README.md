# Augmented Data Visualization Tool

## Overview

`visualizeAugmentedData.py` is a Python script that reads CSV files produced by `augmentData.py` and generates visualization plots for accelerometer data. It creates comprehensive graphs showing both acceleration magnitude and x, y, z components for each event.

## Features

- **Seizure-focused plotting**: By default, only generates plots for seizure events
- **All events option**: Can plot all events (including non-seizures) with the `--all-events` flag
- **Augmented data overlay**: Optionally overlays augmented event data with base events for comparison
- **Descriptive filenames**: Saves plots with clear naming (includes 'seizure' tag for easy filtering)
- **Configurable output**: Specify custom output directory for organized plot storage

## Requirements

All required libraries are available in the project's virtual environment:
- pandas
- matplotlib
- numpy

## Usage

### Basic Usage

```bash
# Activate the virtual environment first
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate

# Plot only seizure events from a CSV file
./visualizeAugmentedData.py data.csv

# Or with full path
python visualizeAugmentedData.py /path/to/data.csv
```

### Command Line Options

```
positional arguments:
  csv_file              Path to the CSV file produced by augmentData.py

optional arguments:
  -h, --help            Show help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save output plots (default: visualization_output)
  -a, --all-events      Plot all events, not just seizure events
  -i, --include-augmented
                        Include augmented data on the same axes as base events for comparison
```

### Examples

#### 1. Basic seizure event plotting
```bash
./visualizeAugmentedData.py after_underample.csv
```
Creates plots for all seizure events in the default `visualization_output` directory.

#### 2. Plot all events (including non-seizures)
```bash
./visualizeAugmentedData.py output_allSeizures.csv --all-events
```

#### 3. Include augmented data overlays
```bash
./visualizeAugmentedData.py output_allSeizures.csv --include-augmented
```
This overlays augmented variants on the same axes as their base events for comparison.

#### 4. Custom output directory
```bash
./visualizeAugmentedData.py data.csv --output-dir ./my_plots
```

#### 5. Combine all options
```bash
./visualizeAugmentedData.py data.csv \
  --all-events \
  --include-augmented \
  --output-dir ./comprehensive_plots
```

## Understanding the Data

### Event IDs

The script recognizes different event ID formats:

- **Base events**: Simple numeric IDs (e.g., `24077`)
- **Augmented events**: Base ID with suffix (e.g., `24077-1`, `24077-2`, or `24077-1-2` for multiple augmentation types)

### Data Columns

The script expects CSV files with the following columns:

- `eventId`: Event identifier (can be numeric like `407` or include augmentation suffix like `407-1`)
- `type`: Event type (1 = seizure, 0 = false alarm/NDA, 2 = other)
- `dataTime`: Timestamp for each datapoint (used for ordering)
- `M000` to `M124`: Acceleration magnitude data (125 samples per datapoint)
- `X000` to `X124`: X-axis acceleration (125 samples per datapoint, optional)
- `Y000` to `Y124`: Y-axis acceleration (125 samples per datapoint, optional)
- `Z000` to `Z124`: Z-axis acceleration (125 samples per datapoint, optional)

**Important**: Each event typically consists of multiple rows (datapoints) in the CSV file. The script:
1. Groups all rows by `eventId`
2. Sorts them by `dataTime` to ensure proper temporal order
3. Concatenates the acceleration data from all datapoints to create a complete time series

For example, an event with 30 datapoints will have:
- 30 rows in the CSV
- 3,750 total samples (30 × 125 samples per datapoint)
- 150 seconds of data (30 × 5 seconds per datapoint)

## Output

### Plot Structure

Each event generates a single PNG file with two subplots:

1. **Top plot**: Acceleration magnitude vs. time
2. **Bottom plot**: X, Y, Z components vs. time (all on same axes)

### Filename Convention

- **Seizure events**: `event_<eventId>_seizure.png`
- **Non-seizure events**: `event_<eventId>.png`

This naming convention makes it easy to filter seizure-only plots using shell commands:
```bash
ls visualization_output/*_seizure.png
```

### Plot Details

- **Resolution**: 150 DPI (high quality for analysis)
- **Size**: 12x10 inches (suitable for presentations/reports)
- **Colors**:
  - Base event magnitude: Blue
  - Base event X/Y/Z: Red/Green/Blue
  - Augmented overlays: Rainbow colors (semi-transparent)
- **Features**: Grid lines, legends, labeled axes, clear titles

## Example Workflow

```bash
# 1. Activate virtual environment
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate

# 2. Navigate to the script directory
cd /home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2

# 3. Run the script on augmented data
./visualizeAugmentedData.py ../../after_underample.csv \
  --include-augmented \
  --output-dir ../../plots_with_augmentation

# 4. View results
ls -lh ../../plots_with_augmentation/*.png

# 5. Filter to see only seizure plots
ls ../../plots_with_augmentation/*_seizure.png
```

## Performance Notes

- Processing time depends on the number of events in the CSV file
- Progress updates are shown every 10 events
- Each plot takes approximately 0.5-1 second to generate
- Memory usage is minimal as events are processed sequentially

## Troubleshooting

### Missing columns error
If you see "Required column 'X' not found", ensure your CSV file was generated by `augmentData.py` and contains the full acceleration data.

### No plots generated
Check that your CSV file contains events of the requested type (seizures by default, or use `--all-events`).

### Memory issues
If processing very large files, consider splitting the CSV into smaller chunks first.

## Integration with augmentData.py

This visualization tool is designed to work with CSV files produced by `augmentData.py`. Typical augmentation workflow:

```bash
# 1. Run augmentation
python augmentData.py --input original_data.csv --output augmented_data.csv

# 2. Visualize results
./visualizeAugmentedData.py augmented_data.csv --include-augmented
```

## Additional Suggestions

While the current implementation uses matplotlib (already in your environment), you could consider adding **seaborn** for enhanced styling:

```bash
pip install seaborn
```

Seaborn would provide:
- Better default color palettes
- Enhanced statistical visualizations
- Improved plot aesthetics

However, the current implementation works perfectly with the existing requirements.

## License

This tool is part of the OpenSeizureDatabase project. See the main project LICENSE for details.
