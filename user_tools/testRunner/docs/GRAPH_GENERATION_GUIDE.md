# Graph Generation Guide for testRunner

## Overview

testRunner automatically generates per-event PNG graphs showing seizure probability and algorithm behavior over time. These graphs provide visual analysis of:
- **Raw sensor data**: Acceleration magnitude during each event
- **Algorithm metrics**: Computed seizure probability or alarm ratio over time  
- **Alarm state evolution**: Comparison between algorithm-predicted and device-reported alarms

This is the primary visualization tool for understanding how well your seizure detection algorithm performs on individual events, especially useful for analyzing both seizure detections and false positive events.

## Automatic Graph Generation

### During Normal Test Run

When you run testRunner.py, graphs are automatically generated for all processed events:

```bash
cd user_tools/testRunner
./testRunner.py --config testConfig_deviceML.json
```

**Output location**: `output/testRun/N/report/`

Where `N` is an incrementing run number (1, 2, 3, ...).

### What Gets Generated

For each test run, three folders are created under `report/`:

```
output/testRun/N/report/
├── index.html                          # Main HTML summary (open in browser)
├── falseNegatives/
│   ├── event_12345.png
│   ├── event_12346.png
│   └── ...
├── truePositives/
│   ├── event_20001.png
│   ├── event_20002.png
│   └── ...
└── falsePositives/
    ├── event_30001.png
    ├── event_30002.png
    └── ...
```

The false positive folder shows a balanced random sample (up to the number of FN + TP events).

## Understanding the Graphs

Each event graph contains 3 panels arranged vertically:

### Panel 1: Acceleration Vector Magnitude

**Y-axis**: Acceleration magnitude in milligravities (mg)  
**X-axis**: Time in seconds from event start  
**What it shows**: Raw sensor movement data (25 Hz samples)

**Interpretation**:
- Seizures typically show **high acceleration peaks** (sudden rapid movements)
- False alarms often show **gradual or sustained acceleration** (normal activities)
- Useful for identifying motion artifacts or distinguishing seizure types

**Example patterns**:
- **Tonic-clonic seizure**: Multiple high-amplitude peaks (100-500+ mg)
- **Walking/exercise**: Steady rhythmic acceleration
- **Falls**: Single or few sharp peaks, then stabilization

### Panel 2: Algorithm Metric vs Time

**Y-axis**: Algorithm-specific metric value  
**X-axis**: Time in seconds from event start  
**What it shows**: Continuous output of seizure probability or alarm ratio for each datapoint

**Metric names by algorithm** (shown in legend):
- **Seizure probability (pSeizure)**: 0.0 to 1.0, higher = more likely to be seizure
- **Alarm Ratio (roiRatio)**: Device algorithm metric
- **Other metrics**: Algorithm-dependent (check algorithm documentation)

**Color coding**: Each algorithm gets a different color (blue, red, green, orange, purple, brown)

**Interpretation**:
- Look for **sharp increases** in metric during actual seizures
- High metrics that don't trigger alarms may indicate algorithm tuning opportunities
- Low metrics for actual seizures indicate missed detections (False Negatives)
- Sustained metrics in non-seizure events indicate false positive risk

### Panel 3: Alarm State Evolution

**Y-axis**: Alarm state (0=OK, 1=WARNING, 2=ALARM)  
**X-axis**: Time in seconds from event start  
**What it shows**: Step plot comparing algorithm predictions vs device-reported alarm state

**Line types**:
- **Solid black line**: Device-reported alarm state (ground truth)
- **Dashed colored lines**: Algorithm outputs (one per algorithm)

**Alarm state meanings**:
- **0 (OK)**: No alert
- **1 (WARNING)**: Preliminary alert (device-dependent behavior)
- **2 (ALARM)**: Full alert triggered

**Interpretation**:
- Ideally, algorithm lines should **match the black line**
- Algorithm line above black line = **false positive** (predicted alarm when none occurred)
- Algorithm line below black line = **false negative** (missed detection)
- Timing mismatches show **detection latency**

## Viewing Results

### HTML Summary Report

The easiest way to browse results:

```bash
# Open the report in your default browser
open output/testRun/N/report/index.html

# Or for Linux:
xdg-open output/testRun/N/report/index.html

# Or view in VS Code browser:
# Click on the file in Explorer, then "Open Preview"
```

**Report contents**:
- Summary statistics (FN, TP, FP counts)
- Algorithm names used in the test
- Categorized event listings with embedded PNG graphs
- Click through each section to view all events

### Direct File Access

Individual PNG files can be viewed directly:

```bash
# View a specific event graph
open output/testRun/1/report/falsePositives/event_12345.png

# Or list all seizure events
ls output/testRun/1/report/truePositives/
```

## Regenerating Reports from Saved Results

If you already have test results and want to regenerate the graphs (e.g., with different visualization settings), use the `--analyze` flag:

```bash
./testRunner.py --config testConfig.json --rerun N --analyze
```

**Requirements**:
- `testConfig.json`: Your configuration file (same as original run)
- `--rerun N`: The run number to analyze (must exist)
- Original data files must still be accessible

**What it regenerates**:
- All PNG graphs (Panel 1-3 for each event)
- HTML summary report with categorized events
- Preserves all original CSV results

**Use cases**:
- Adjust graph appearance/settings without re-running algorithms
- Debug visualization issues
- Create reports for presentation/publication
- Share results folder without needing original run folder

**Example workflow**:
```bash
# Run tests (takes 30 minutes)
./testRunner.py --config testConfig.json

# Later, regenerate just the graphs (takes 2 minutes)
./testRunner.py --config testConfig.json --rerun 1 --analyze
```

## Customizing Graphs

The `report.py` module controls graph generation. Key customization points:

### Changing Graph Layout

Edit `report.py`, function `generateEventGraph()`:
- Line ~60-70: Modify figure size with `figsize=(14, 10)`
- Line ~100+: Adjust panel titles and labels
- Line ~130-140: Modify color scheme via `_GRAPH_COLORS` list

### Adjusting Panel Height Ratios

Modify `plt.subplots(3, 1, figsize=...)` to `plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 2, 1]})` for different proportions.

### Export Format

Currently exports as PNG at 100 DPI. To change:

Edit line in `generateEventGraph()`:
```python
# Change DPI for higher quality (slower, larger files)
fig.savefig(outFname, dpi=150, bbox_inches='tight')
```

## Data Files Referenced in Graphs

The graphs are based on data stored during algorithm execution:

**File**: `output/testRun/N/perDpData.json`

Contains for each event:
- `timestamps`: Raw sample timestamps (25 Hz)
- `accelMag`: Acceleration vector magnitudes
- `dpTimestamps`: Datapoint timestamps
- `reportedAlarmStates`: Device-reported alarm states
- `algOutputs`: Per-algorithm metrics and predictions

This JSON file is automatically generated during test run and used to create graphs.

## Example Workflow

### Step 1: Run Tests
```bash
./testRunner.py --config testConfig_deviceML.json --debug
# Output: output/testRun/1/
```

### Step 2: View HTML Report
```bash
open output/testRun/1/report/index.html
```

### Step 3: Analyze False Negatives
1. Click "False Negatives" section in HTML
2. Examine each graph to understand why algorithm missed detection
3. Note patterns in Panel 2 (low metric values indicate algorithm needs tuning)

### Step 4: Analyze False Positives
1. Click "False Positives" section
2. Look for high Panel 2 metrics in non-seizure events
3. Check if Panel 1 (acceleration) resembles seizures
4. Use findings to adjust algorithm thresholds

### Step 5: Compare Algorithms
1. Look at legend in Panel 2 for multiple algorithm lines
2. Compare which algorithm performs better for different event types
3. Make tuning decisions based on FN/TP/FP patterns

## Troubleshooting

### No graphs generated

**Symptom**: `report/` folder is empty after test run

**Causes**:
1. Test run crashed before graph generation
   - Check for errors in testRunner output
   - Use `--analyze` to regenerate from checkpoint

2. perDpData.json is missing
   - Check `output/testRun/N/perDpData.json` exists
   - Re-run test: `./testRunner.py --config testConfig.json`

3. Data files not found during --analyze
   - Ensure `dataFiles` in config are still accessible
   - Use absolute paths if needed

### Graphs show no data in panels

**Symptom**: Panels are blank or show "No metric data available"

**Causes**:
1. Algorithm didn't return expected output fields
   - Check algorithm returns `metrics` and `alarmStates`
   - See algorithm documentation

2. Data format mismatch
   - Verify CSV/JSON data has correct acceleration and timestamp fields

### Graph titles don't show complete information

**Symptom**: User ID or event time cut off in title

**Causes**: Event metadata missing from data file
- Ensure data includes `userId`, `type`, `dataTime` fields

## Related Documentation

- [EVENT_LEVEL_METRICS_README.md](EVENT_LEVEL_METRICS_README.md) - Event classification and statistics
- [QUICKSTART_EVENT_METRICS.md](QUICKSTART_EVENT_METRICS.md) - Quick start guide
- [README.md](README.md) - Original testRunner documentation
- [report.py](report.py) - Source code for graph generation

## Questions or Feedback?

For issues with graph generation or suggestions for improvements, check the code in `report.py` or contact the development team.

---

**Last updated**: 2026-09-05  
**Applies to**: testRunner with report.py graph generation module
