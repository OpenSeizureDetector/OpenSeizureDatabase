# Quick Start: On-Device Testing with Event-Level Metrics

## What You Can Do Now

Test your .pte model on a physical device using CSV test data from nnTraining2, with automated event-level statistics matching nnTester's output format.

## 1. Prepare Test Data

From your nnTraining2 output folder, you should have:
```
user_tools/nnTraining2/output/testData.csv
```

## 2. Create Device Test Configuration

Create `testConfig_device.json`:

```json
{
  "dbDir": null,
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

**Key settings:**
- `ipAddr`: Your device's IP address
- `delayMs`: Delay between datapoints (ms), optional (null = no delay)
- `excludeTrainingEvents`: Path to trainData.csv to avoid train/test contamination

## 3. Run Test

```bash
cd user_tools/testRunner
./testRunner.py --config testConfig_device.json --debug
```

## 4. Review Results

Check the output folder (e.g., `output/testRun/1/`):

**Standard testRunner outputs:**
- `output_allSeizures.csv` - All seizure events with algorithm results
- `output_falseAlarms.csv` - False alarm events
- `output_nda.csv` - NDA events
- `testRunner_Summary.txt` - Summary statistics

**NEW: Event-level metrics:**
- `eventLevel_standard.csv` - Event predictions (alarm=2 only)
- `eventLevel_sensitive.csv` - Event predictions (alarm>=1)
- `eventLevel_comparison.txt` - Side-by-side comparison

**Visual reports:**
- `summary_report.html` - Interactive HTML report
- Per-event PNG graphs

## 5. Interpret Results

### Standard Mode (alarmState=2)
Conservative detection - only counts definite alarms as seizures.

**Example output:**
```
STANDARD MODE (alarmState=2 only):
  True Positives (TP):  45
  False Positives (FP): 3
  True Negatives (TN):  97
  False Negatives (FN): 5
  Sensitivity (TPR):    0.900 (90.0%)
  Specificity (TNR):    0.970 (97.0%)
  False Alarm Rate:     0.030 (3.0%)
```

### Sensitive Mode (alarmState>=1)
Aggressive detection - warnings and alarms both count as seizures.

**Example output:**
```
SENSITIVE MODE (alarmState>=1, warnings count):
  True Positives (TP):  48
  False Positives (FP): 8
  True Negatives (TN):  92
  False Negatives (FN): 2
  Sensitivity (TPR):    0.960 (96.0%)
  Specificity (TNR):    0.920 (92.0%)
  False Alarm Rate:     0.080 (8.0%)
```

### Comparison
```
SENSITIVITY COMPARISON:
  TPR Increase: +0.060 (+6.0%)
  FPR Increase: +0.050 (+5.0%)
  Additional Seizures Detected: 3
  Additional False Alarms: 5
```

**Interpretation:**
- Standard mode missed 5 seizures → 90% sensitivity
- Sensitive mode missed only 2 seizures → 96% sensitivity
- Cost: 5 additional false alarms (3% → 8% false alarm rate)

## Configuration Options

### Minimal (just test with defaults)
```json
{
  "dataFiles": ["testData.csv"],
  "algorithms": [{
    "name": "Device",
    "alg": "deviceAlg.DeviceAlg",
    "enabled": true,
    "settings": {"ipAddr": "192.168.1.100"}
  }],
  "invalidEvents": []
}
```

### Standard Mode Only
```json
{
  "eventLevelMetrics": {
    "enabled": true,
    "treatWarningsAsSeizures": false,
    "compareSensitivityModes": false
  }
}
```

### Sensitive Mode Only
```json
{
  "eventLevelMetrics": {
    "enabled": true,
    "treatWarningsAsSeizures": true,
    "compareSensitivityModes": false
  }
}
```

### Disable Event-Level Metrics
```json
{
  "eventLevelMetrics": {
    "enabled": false
  }
}
```

## Command-Line Options

### Override test data file
```bash
./testRunner.py --config testConfig.json --testData /path/to/testData.csv
```

### Test seizures only (faster)
```bash
./testRunner.py --config testConfig.json --seizuresOnly
```

### Specify output directory
```bash
./testRunner.py --config testConfig.json --outDir ./my_results
```

### Re-analyze existing results
```bash
./testRunner.py --config testConfig.json --rerun 1 --analyze
```

### Debug mode
```bash
./testRunner.py --config testConfig.json --debug
```

## Troubleshooting

### Device not connecting
- Verify IP address: `ping 192.168.1.100`
- Check device is running and OSD server is active
- Try telnet: `telnet 192.168.1.100 8080`

### CSV file not found
- Use absolute path or place CSV in testRunner directory
- Check filename in config matches actual file
- Try --testData command-line override

### No event-level metrics generated
- Check eventLevelMetrics.enabled is true
- Look for error messages in output
- Run with --debug for detailed logging

### ImportError for libosd
- Ensure you're running from user_tools/testRunner directory
- Check PYTHONPATH includes repository root

## Next Steps

1. **Compare with nnTester**: Run nnTester on the same test data and compare event-level results
2. **Optimize sensitivity**: Use the comparison to choose optimal sensitivity threshold
3. **Analyze failures**: Review false negatives and false positives for patterns
4. **Iterate model**: Use insights to improve training data or model architecture

## Full Documentation

- [EVENT_LEVEL_METRICS_README.md](EVENT_LEVEL_METRICS_README.md) - Complete documentation
- [GRAPH_GENERATION_GUIDE.md](GRAPH_GENERATION_GUIDE.md) - How to interpret per-event graphs
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical details
- [testConfig_example_device.json](testConfig_example_device.json) - Full config example

## Understanding Per-Event Graphs

After running tests, browse the results with:
```bash
open output/testRun/1/report/index.html
```

Each event graph has 3 panels:
1. **Acceleration magnitude** - Raw sensor motion (25 Hz samples)
2. **Algorithm metric** - Seizure probability or alarm ratio over time
3. **Alarm states** - Algorithm predictions vs device-reported alarms

Use graphs to understand:
- **False Negatives**: Why algorithm missed seizure (check panel 2 for low metrics)
- **False Positives**: Whether motion resembles seizure (check panels 1-2)
- **Differences**: How algorithm detection timing compares to device

See [GRAPH_GENERATION_GUIDE.md](GRAPH_GENERATION_GUIDE.md) for detailed interpretation examples and troubleshooting.

## Support

If you encounter issues:
1. Check [EVENT_LEVEL_METRICS_README.md](EVENT_LEVEL_METRICS_README.md)
2. Run with --debug for detailed output
3. Review test_eventLevelMetrics.py for usage examples
4. Check that CSV format matches flattenData.py output
