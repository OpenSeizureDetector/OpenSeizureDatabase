# TestRunner Crash Recovery Guide

## Quick Start

### First Run
```bash
./testRunner.py --config testConfig.json
```
This creates a new output folder and saves progress periodically.

### After a Crash - Resume from Checkpoint
```bash
./testRunner.py --config testConfig.json --resume --rerun 1
```
Continues from where the previous run crashed.

### Quick Seizure Test (TPR Assessment)
```bash
./testRunner.py --config testConfig.json --seizuresOnly
```
Tests only seizure events for faster validation (~3-5x faster).

---

## What Was Changed?

### 1. **Retry Logic with Backoff**
   - HTTP timeout errors now retry up to 3 times (default)
   - Waits 1s, 2s, 4s between retries
   - Prevents crashes from transient network issues

### 2. **Checkpoint/Resume System**
   - Progress saved after each event
   - Checkpoint file: `output/testRun/N/checkpoint.json`
   - Automatically resumes without manual state management

### 3. **Crash Recovery**
   - If runner crashes, partial results are saved
   - Files marked with `_PARTIAL` suffix
   - `PARTIAL_RUN.txt` marker explains the situation

### 4. **Seizure-Only Option**
   - New `--seizuresOnly` flag for quick testing
   - Tests only seizure events
   - Produces results in standard format

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **Retry with backoff** | Handles network delays & busy devices |
| **Automatic checkpoint** | No data loss, resume seamlessly |
| **Partial result saving** | See what was tested before crash |
| **Seizure-only option** | Quick TPR assessment (~3-5x faster) |
| **Exception handling** | Graceful degradation on errors |

---

## Output Files

### When Run Completes Successfully
```
output/testRun/1/
├── testConfig.json                 # Run configuration
├── cmdArgs.json                    # CLI arguments used
├── output_allSeizures.csv          # Seizure test results
├── output_falseAlarms.csv          # False alarm results
├── output_nda.csv                  # NDA results
├── output_otherEvents.csv          # Other event results
├── testRunner_Summary.txt          # Summary stats (TPR, TNR)
├── perDpData.json                  # Per-datapoint data for reports
└── checkpoint.json                 # Final checkpoint (can be deleted)
```

### When Run Crashes
```
output/testRun/1/
├── [above files for processed events]
├── output_allSeizures_PARTIAL.csv  # Partial seizure results
├── output_falseAlarms_PARTIAL.csv  # Partial false alarm results
├── PARTIAL_RUN.txt                 # Marker file
├── perDpData_PARTIAL.json          # Partial per-datapoint data
└── checkpoint.json                 # Resume checkpoint (required for --resume)
```

---

## Typical Workflow

### Scenario: Device timeout after 35 minutes
```bash
# Start the run
$ ./testRunner.py --config testConfig.json
# ... runs for 35 minutes ...
# ... device timeout, crash ...
# ERROR: Test runner crashed: HTTPConnectionPool(host='192.168.0.104', port=8080)

# Partial results automatically saved
# output/testRun/1/output_allSeizures_PARTIAL.csv created
# output/testRun/1/checkpoint.json created

# Resume from checkpoint
$ ./testRunner.py --config testConfig.json --resume --rerun 1
# ... continues from event 152/200 ...
# ... remaining 48 events processed ...
# Completes successfully

# Final results now in:
# output/testRun/1/output_allSeizures.csv (all 200 events)
# output/testRun/1/testRunner_Summary.txt (final stats)
```

### Scenario: Quick seizure validation
```bash
# Standard run: 100 seizures + 300 non-seizures = 400 events (~30 min)
# Seizure-only: 100 seizures = 100 events (~5 min)

$ ./testRunner.py --config testConfig.json --seizuresOnly
# ... processes only seizure events ...
# Results in: output/testRun/1/output_allSeizures.csv

# Quick TPR assessment without waiting for full test
```

---

## Troubleshooting

### "ERROR: Checkpoint file not found"
**Problem**: You ran with `--resume --rerun 1` but there's no checkpoint.

**Solution**: 
- Check the output folder exists: `ls output/testRun/1/`
- If folder exists but no checkpoint, the previous run completed successfully
- Run a new test normally (no `--resume` flag)

### "ERROR: Run folder not found"
**Problem**: Output folder doesn't exist.

**Solution**:
- Run a new test (without `--resume` flag)
- Use `--rerun 1` to create testRun/1/

### Checkpoint file is huge
**Problem**: `checkpoint.json` is very large (100+ MB).

**Solution**:
- This is normal for large test runs
- File will be deleted automatically after successful completion
- Safe to delete manually if needed (but can't resume after deletion)

### Network still timing out
**Problem**: Still getting timeout errors even with retries.

**Solutions**:
1. Increase retry count:
   - Edit osdAppConnection.py
   - Change `max_retries=3` to `max_retries=5`
   
2. Increase wait times:
   - Edit osdAppConnection.py
   - Change `backoff_factor=2.0` to `backoff_factor=3.0`
   
3. Check device connectivity:
   - Verify device is responding: ping 192.168.0.104
   - Check device CPU usage (may be processing heavy algorithm)
   - Try increasing device timeout in config

---

## Advanced Usage

### Resume with Modified Configuration
```bash
# If you need to change config for resumed run:
1. Edit testConfig.json
2. Delete old testConfig.json from output/testRun/1/
3. Run with --resume (new config will be used)
```

### Analyze Results from Partial Run
```bash
# Generate summary report from incomplete run
./testRunner.py --config testConfig.json --analyze --rerun 1
# Creates graphs and report based on events processed so far
```

### Dry Run Checkpoint
```bash
# Start run, stop after first few events
./testRunner.py --config testConfig.json
# ... wait a few minutes ...
# Press Ctrl+C to interrupt
# checkpoint.json created with first N events
# Resume later with --resume
```

---

## Technical Details

### Checkpoint Contents
```json
{
  "eventNo": 152,              # Events processed (0-based)
  "nEvents": 200,              # Total events
  "slot_names": [...],         # Algorithm names
  "results_counts": [...],     # Result matrix
  "resultsStrArr": [...],      # Result strings
  "perDpDataLst": [...]        # Per-datapoint data
}
```

### Retry Strategy
```
Attempt 1: Fail → Wait 1 second
Attempt 2: Fail → Wait 2 seconds
Attempt 3: Fail → Wait 4 seconds
All failed → Raise exception, save partial results
```

### Result Format
- `_` = No data processed yet
- `0` = OK alarm state
- `1` = WARNING alarm state
- `2` = ALARM alarm state
- `.` = Invalid/skipped datapoint

---

## Questions?

Refer to the main testRunner documentation or check the implementation:
- [testRunner.py](./testRunner.py) - Main orchestrator
- [alg_runner.py](./alg_runner.py) - Event processing & checkpointing
- [results.py](./results.py) - Result saving
- [osdAppConnection.py](../../libosd/osdAppConnection.py) - Network communication
