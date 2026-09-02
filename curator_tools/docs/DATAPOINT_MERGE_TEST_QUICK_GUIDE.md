# Datapoint Merge Validation - Quick Reference Guide

## Overview
This guide provides quick commands for running and understanding the datapoint merge validation tests that ensure events are combined correctly without data loss or overlaps.

---

## Quick Start

### Run All Merge Validation Tests
```bash
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor

# Activate virtual environment
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate

# Run all tests
pytest tests/test_datapoint_merge_validation.py -v

# Or run with detailed output
pytest tests/test_datapoint_merge_validation.py -v -s
```

### Run Unit Grouping Tests (28 tests)
```bash
pytest tests/test_unit_grouping.py -v
```

### Run All Merge-Related Tests (42 total)
```bash
pytest tests/test_unit_grouping.py tests/test_datapoint_merge_validation.py -v
```

---

## Test Files

### File 1: test_datapoint_merge_validation.py (NEW - 14 tests)
**Location:** `tests/test_datapoint_merge_validation.py`  
**Purpose:** Comprehensive validation of datapoint merge operations

**Test Classes:**
1. **TestDatapointTimeOrdering** (2 tests)
   - Validates chronological ordering of merged datapoints
   - Checks monotonic increasing time intervals

2. **TestDatapointTimeOverlapDetection** (2 tests)
   - Ensures no overlapping time intervals in merged data
   - Validates 100ms duplicate detection tolerance

3. **TestAccelerationDataIntegrity** (3 tests)
   - Verifies rawData arrays preserved
   - Verifies rawData3D (3D acceleration) preserved
   - Tests mixed acceleration formats

4. **TestDatapointCountAndStats** (2 tests)
   - Tracks _merged_event_count
   - Tracks _merged_datapoint_count

5. **TestEdgeCases** (3 tests)
   - Handles zero-datapoint events
   - Handles missing time fields
   - Processes large acceleration arrays (1000+ elements)

6. **TestComplexMergingScenarios** (2 tests)
   - Realistic 3-event seizure sequence merge
   - Data type preservation validation

### File 2: test_unit_grouping.py (UPDATED - 28 tests)
**Location:** `tests/test_unit_grouping.py`  
**Purpose:** Unit tests for grouping functions + NEW datapoint concatenation tests

**New Test Classes:**
- **TestConcatenateDatapoints** (6 tests)
  - Simple concatenation
  - Time ordering after concat
  - Duplicate removal with tolerance
  - No duplicate removal option
  - Empty datapoint handling
  - Acceleration data preservation

---

## Running Specific Tests

### Run Only Time Ordering Tests
```bash
pytest tests/test_datapoint_merge_validation.py::TestDatapointTimeOrdering -v
```

### Run Only Overlap Detection Tests
```bash
pytest tests/test_datapoint_merge_validation.py::TestDatapointTimeOverlapDetection -v
```

### Run Only Acceleration Data Tests
```bash
pytest tests/test_datapoint_merge_validation.py::TestAccelerationDataIntegrity -v
```

### Run Only Complex Scenarios
```bash
pytest tests/test_datapoint_merge_validation.py::TestComplexMergingScenarios -v
```

### Run With Detailed Output (Print Statements)
```bash
pytest tests/test_datapoint_merge_validation.py -v -s
```

### Run With Coverage Report
```bash
pytest tests/test_datapoint_merge_validation.py --cov=src/event_grouping --cov-report=html
```

---

## Test Results Format

### When All Tests Pass (Expected)
```
======================== 42 passed in 0.21s ========================
```

### Check Individual Test Output
```bash
pytest tests/test_datapoint_merge_validation.py::TestComplexMergingScenarios::test_three_event_merge_with_realistic_data -v -s
```

**Expected Output:**
```
✓ Complex 3-event merge successful: 19 datapoints, properly ordered
  Event IDs: ['101', '102', '103']
  Time span: 1705329000.0 to 1705329135.0 (135.0s)
PASSED
```

---

## What Each Test Validates

| Test Name | Purpose | Validation |
|-----------|---------|-----------|
| test_datapoints_ordered_by_time | Time ordering | All 6 datapoints in chronological order |
| test_datapoints_monotonic_increasing_time | No time gaps | Timestamps strictly increasing |
| test_no_overlapping_time_intervals | No overlaps | 4 distinct time values |
| test_duplicate_timestamp_handling | Deduplication | 100ms tolerance applied correctly |
| test_raw_acceleration_data_preserved | Data integrity | Both rawData arrays preserved |
| test_3d_acceleration_data_preserved | Data integrity | Both rawData3D arrays preserved |
| test_mixed_acceleration_formats | Format mixing | rawData + rawData3D coexist |
| test_merged_event_count_tracking | Metadata | Count = 4 for 4 merged events |
| test_merged_datapoint_count_tracking | Metadata | Count matches actual datapoints |
| test_merge_with_zero_datapoints | Edge case | Handles empty datapoint events |
| test_merge_events_with_missing_time_field | Edge case | Graceful handling of missing fields |
| test_large_number_of_datapoints | Performance | 3000 elements processed correctly |
| test_three_event_merge_with_realistic_data | Real-world | Seizure sequence merge validated |
| test_merge_preserves_sensor_data_types | Type safety | All data types preserved |

---

## Understanding Test Output

### Successful Test Run
```bash
$ pytest tests/test_datapoint_merge_validation.py -v

tests/test_datapoint_merge_validation.py::TestDatapointTimeOrdering::test_datapoints_ordered_by_time PASSED
  ✓ 6 datapoints properly ordered with monotonic increasing time

tests/test_datapoint_merge_validation.py::TestAccelerationDataIntegrity::test_raw_acceleration_data_preserved PASSED
  ✓ RawData arrays preserved: 2 arrays

tests/test_datapoint_merge_validation.py::TestComplexMergingScenarios::test_three_event_merge_with_realistic_data PASSED
  ✓ Complex 3-event merge successful: 19 datapoints, properly ordered
  Event IDs: ['101', '102', '103']
  Time span: 1705329000.0 to 1705329135.0 (135.0s)
```

### If a Test Fails (Troubleshooting)
```bash
# Get more details
pytest tests/test_datapoint_merge_validation.py::FAILED_TEST_NAME -vv

# Run with full traceback
pytest tests/test_datapoint_merge_validation.py -v --tb=long

# Run with output capture disabled
pytest tests/test_datapoint_merge_validation.py -v -s --tb=short
```

---

## Integration with Development Workflow

### Pre-Commit Validation
```bash
# Run before committing changes to event_grouping.py
pytest tests/test_datapoint_merge_validation.py tests/test_unit_grouping.py -v --tb=short
```

### Continuous Integration
```bash
# Complete validation suite
pytest tests/test_datapoint_merge_validation.py \
       tests/test_unit_grouping.py \
       tests/test_datapoint_transfer.py \
       -v --tb=short --json-report --json-report-file=report.json
```

---

## Key Test Scenarios

### Scenario 1: Basic Two-Event Merge
**Events:** 2 seizure detections 10 seconds apart  
**Expected:** 4-6 datapoints, properly ordered, no overlaps

### Scenario 2: Duplicate Prevention
**Events:** 2 events with same timestamp (within 100ms)  
**Expected:** Deduplication removes near-duplicates

### Scenario 3: Acceleration Data Preservation
**Events:** Mixed rawData and rawData3D formats  
**Expected:** Both acceleration formats preserved, data intact

### Scenario 4: Realistic Seizure Sequence
**Events:** 3 events (initial detection → continued → post-ictal)  
**Time:** 135 seconds total  
**Expected:** 19 datapoints, chronological, metadata accurate

---

## Validation Checklist

Before deployment, verify:
- [ ] All 42 tests pass
- [ ] No performance regressions
- [ ] Acceleration data preserved completely
- [ ] Time ordering correct for realistic scenarios
- [ ] Metadata tracking accurate
- [ ] Edge cases handled gracefully

---

## Related Documentation

- **[DATAPOINT_MERGE_TEST_RESULTS.md](DATAPOINT_MERGE_TEST_RESULTS.md)** - Comprehensive test results and findings
- **[src/event_grouping.py](src/event_grouping.py)** - Implementation under test
- **[tests/test_unit_grouping.py](tests/test_unit_grouping.py)** - Unit tests
- **[tests/test_datapoint_merge_validation.py](tests/test_datapoint_merge_validation.py)** - Comprehensive validation tests
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Overall system status

---

## Test Statistics

```
Total Tests:           42
  - Unit tests:        28 (test_unit_grouping.py)
  - Validation tests:  14 (test_datapoint_merge_validation.py)

Execution Time:        ~0.2 seconds
Pass Rate:             100%

Test Coverage:
  ✓ Time ordering
  ✓ Overlap detection
  ✓ Data preservation
  ✓ Metadata tracking
  ✓ Edge cases
  ✓ Real-world scenarios
```

---

## Troubleshooting

### ImportError: No module named 'event_grouping'
```bash
# Ensure working directory is correct
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor

# Activate venv
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate

# Run tests
pytest tests/test_datapoint_merge_validation.py -v
```

### Tests Timeout
```bash
# Increase timeout (especially for large array tests)
pytest tests/test_datapoint_merge_validation.py -v --timeout=30
```

### Missing Dependencies
```bash
# Install required packages
pip install pytest pandas dateutil

# Verify installation
pytest --version
```

### Verbose Debugging
```bash
# Full traceback and output
pytest tests/test_datapoint_merge_validation.py::SPECIFIC_TEST -vv -s --tb=long --capture=no
```

---

## Performance Expectations

| Test Category | Time |
|---------------|------|
| Time Ordering | ~5ms |
| Overlap Detection | ~5ms |
| Acceleration Data | ~5ms |
| Edge Cases | ~10ms |
| Complex Scenarios | ~15ms |
| **Total (42 tests)** | **~210ms** |

---

## Success Criteria

✅ **Test Suite is Passing Successfully When:**
- All 42 tests show PASSED status
- Execution completes in < 1 second
- No error messages in stderr
- Output includes "✓" check marks for validation messages

**Current Status:** ✅ All tests passing (42/42)

---

For additional questions or issues, refer to the comprehensive test results documentation in `DATAPOINT_MERGE_TEST_RESULTS.md`.
