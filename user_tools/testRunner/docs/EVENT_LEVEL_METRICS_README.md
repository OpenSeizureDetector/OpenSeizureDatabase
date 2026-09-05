# testRunner with Event-Level Metrics

## Overview

testRunner now includes event-level metrics calculation similar to nnTester, with support for sensitivity analysis.

## New Features

### Event-Level Metrics

testRunner aggregates datapoint-level alarm states to event-level predictions and calculates comprehensive statistics:

- **Standard Mode**: Only `alarmState=2` (ALARM) counts as seizure detection
- **Sensitive Mode**: `alarmState>=1` (WARNING or ALARM) counts as seizure detection

### Output Files

The following new files are generated in the output folder:

1. **eventLevel_standard.csv** - Event-level results using standard mode
   - Columns: EventID, UserID, Type, SubType, DataTime, TrueLabel
   - Algorithm predictions and alarm state counts for each algorithm
   - Description field

2. **eventLevel_sensitive.csv** - Event-level results using sensitive mode
   - Same format as standard, but with warnings counting as detections

3. **eventLevel_comparison.txt** - Side-by-side comparison of both modes
   - TP/FP/TN/FN counts for each mode
   - Sensitivity (TPR), Specificity (TNR), False Alarm Rate (FPR)
   - Additional seizures detected and false alarms in sensitive mode

### Configuration

Add the following section to your `testConfig.json`:

```json
{
  "eventLevelMetrics": {
    "enabled": true,
    "treatWarningsAsSeizures": false,
    "compareSensitivityModes": true
  }
}
```

**Configuration Options:**

- `enabled` (boolean, default: true): Enable event-level metrics generation
- `treatWarningsAsSeizures` (boolean, default: false): 
  - `false`: Standard mode (alarm=2 only)
  - `true`: Sensitive mode (alarm>=1)
- `compareSensitivityModes` (boolean, default: true):
  - `true`: Generate both modes and comparison
  - `false`: Generate only the mode specified by `treatWarningsAsSeizures`

### CSV File Support

testRunner fully supports CSV files from `flattenData.py`. Simply specify the CSV file in `dataFiles`:

```json
{
  "dataFiles": ["testData.csv"]
}
```

Or use the command-line override:

```bash
./testRunner.py --config testConfig.json --testData /path/to/testData.csv
```

### Example: On-Device Testing with CSV

**testConfig_device.json:**
```json
{
  "dbDir": "/home/user/osdb",
  "dataFiles": ["testData.csv"],
  
  "eventFilters": {
    "includeUserIds": [],
    "excludeUserIds": [],
    "includeTypes": [],
    "excludeTypes": [],
    "includeSubTypes": [],
    "excludeSubTypes": [],
    "includeDataSources": [],
    "excludeDataSources": [],
    "includeText": [],
    "excludeText": [],
    "requireHrData": false,
    "requireO2SatData": false,
    "require3dData": false,
    "excludeTrainingEvents": "trainData.csv"
  },
  
  "algorithms": [
    {
      "name": "PhysicalDevice",
      "alg": "deviceAlg.DeviceAlg",
      "enabled": true,
      "settings": {
        "ipAddr": "192.168.1.100",
        "delayMs": 200
      }
    }
  ],
  
  "invalidEvents": [],
  
  "eventLevelMetrics": {
    "enabled": true,
    "treatWarningsAsSeizures": false,
    "compareSensitivityModes": true
  }
}
```

**Run the test:**
```bash
cd user_tools/testRunner
./testRunner.py --config testConfig_device.json --debug
```

**Output:**
- Standard testRunner outputs (output_allSeizures.csv, output_falseAlarms.csv, etc.)
- **NEW:** eventLevel_standard.csv
- **NEW:** eventLevel_sensitive.csv  
- **NEW:** eventLevel_comparison.txt
- Per-event graphs and HTML summary report

### Understanding the Metrics

**Standard Mode (alarmState=2):**
- Conservative: Only counts definite alarms as seizure detections
- Lower sensitivity, higher specificity
- Typical for production deployment

**Sensitive Mode (alarmState>=1):**
- Aggressive: Counts warnings and alarms as detections
- Higher sensitivity, lower specificity  
- Useful for analyzing how many additional seizures could be caught

**Comparison Analysis:**
The comparison file shows the trade-off:
- How many additional seizures are detected in sensitive mode
- How many additional false alarms occur
- The change in TPR (sensitivity) and FPR (false alarm rate)

### Example Output

**eventLevel_comparison.txt excerpt:**
```
================================================================================
EVENT-LEVEL METRICS - SENSITIVITY MODE COMPARISON
================================================================================

Standard Mode: alarmState=2 (ALARM) counts as seizure detection
Sensitive Mode: alarmState>=1 (WARNING or ALARM) counts as seizure detection

--------------------------------------------------------------------------------
Algorithm: PhysicalDevice
--------------------------------------------------------------------------------

STANDARD MODE (alarmState=2 only):
  True Positives (TP):  45
  False Positives (FP): 3
  True Negatives (TN):  97
  False Negatives (FN): 5
  Sensitivity (TPR):    0.900 (90.0%)
  Specificity (TNR):    0.970 (97.0%)
  False Alarm Rate:     0.030 (3.0%)

SENSITIVE MODE (alarmState>=1, warnings count):
  True Positives (TP):  48
  False Positives (FP): 8
  True Negatives (TN):  92
  False Negatives (FN): 2
  Sensitivity (TPR):    0.960 (96.0%)
  Specificity (TNR):    0.920 (92.0%)
  False Alarm Rate:     0.080 (8.0%)

SENSITIVITY COMPARISON:
  TPR Increase: +0.060 (+6.0%)
  FPR Increase: +0.050 (+5.0%)
  Additional Seizures Detected: 3
  Additional False Alarms: 5
```

### Integration with Existing Workflow

The event-level metrics integrate seamlessly with the existing testRunner workflow:

1. **Data loading**: Works with both JSON and CSV files
2. **Algorithm testing**: Uses existing algorithm infrastructure (deviceAlg, osdAlg, nnAlg, etc.)
3. **Results**: Extends existing outputs with event-level analysis
4. **Backward compatible**: Existing configs work without modification (defaults to enabled)

### Differences from nnTester

While the output format is similar to nnTester's event-level results, there are key differences:

**testRunner (alarm-state based):**
- Works with physical devices and streaming algorithms
- Alarm states: 0=OK, 1=WARNING, 2=ALARM (discrete)
- No ROC/PR curves (not probability-based)
- Event classification based on threshold (alarmState >= threshold)

**nnTester (probability-based):**
- Works with trained models and inference
- Outputs seizure probabilities (continuous 0-1)
- Supports ROC/PR curves and threshold optimization
- Event classification based on probability threshold

### Use Cases

**1. On-Device Validation**
Test a .pte model deployed on physical hardware:
```bash
./testRunner.py --config testConfig_device.json --testData testData.csv
```

**2. Algorithm Comparison**
Compare multiple algorithms (device, OSD, ML models):
```json
{
  "algorithms": [
    {"name": "Device", "alg": "deviceAlg.DeviceAlg", "enabled": true, ...},
    {"name": "OSD_Local", "alg": "osdAlg.OsdAlg", "enabled": true, ...},
    {"name": "SpecCnn", "alg": "specAlg.SpecAlg", "enabled": true, ...}
  ]
}
```

**3. Sensitivity Analysis**
Understand the impact of warning-level detections:
```json
{
  "eventLevelMetrics": {
    "enabled": true,
    "compareSensitivityModes": true
  }
}
```

## Migration Notes

If you have existing testRunner configurations:
- No changes required
- Event-level metrics enabled by default
- To disable: Add `"eventLevelMetrics": {"enabled": false}` to config

## Visual Analysis: Per-Event Graphs

In addition to event-level metrics, testRunner automatically generates per-event graphs showing:

- **Panel 1**: Acceleration magnitude vs time (raw sensor data)
- **Panel 2**: Algorithm metric (seizure probability) vs time  
- **Panel 3**: Alarm state evolution (algorithm vs device-reported)

These graphs help you understand:
- Why algorithms missed seizures (False Negatives)
- Which false alarms have motion patterns similar to seizures
- Timing differences between algorithm and device detection

**To view graphs:**
```bash
# Open the HTML summary report
open output/testRun/1/report/index.html
```

**To regenerate graphs from existing results:**
```bash
./testRunner.py --config testConfig.json --rerun 1 --analyze
```

For detailed interpretation guide, see [GRAPH_GENERATION_GUIDE.md](GRAPH_GENERATION_GUIDE.md).

## See Also

- [testRunner README](README.md) - General testRunner documentation
- [GRAPH_GENERATION_GUIDE.md](GRAPH_GENERATION_GUIDE.md) - Per-event graph interpretation
- [deviceAlg.py](deviceAlg.py) - Physical device testing implementation
- [io_utils.py](io_utils.py) - CSV/JSON data loading
- [eventLevelMetrics.py](eventLevelMetrics.py) - Event-level metrics implementation
