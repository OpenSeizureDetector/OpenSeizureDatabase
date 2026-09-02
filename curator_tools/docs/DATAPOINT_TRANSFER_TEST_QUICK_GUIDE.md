# Datapoint Transfer Test - Quick Reference Guide

## Overview
Comprehensive test suite validating that datapoints downloaded from the remote server are correctly imported into the SQLite database and exported to JSON files.

**Status**: ✅ All 9 tests passing

---

## Running the Tests

### Quick Test (Local Data Only - No Server Required)
```bash
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferWithLocalData -v
```

**Time**: ~0.1 seconds  
**Results**: 3 tests - all passing ✅

---

### Full Test Suite (Includes Remote Server Validation)
```bash
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate
python3 -m pytest tests/test_datapoint_transfer.py -v
```

**Time**: ~30 seconds (includes server connection)  
**Results**: 9 tests - all passing ✅

**Requirements**:
- Valid credentials in `../client.cfg`
- Network connectivity to OpenSeizureDatabase server
- Python venv with pytest installed

---

### Individual Test Groups

**Test Server Connection & Event Retrieval**:
```bash
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_01_server_connection -v -s
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_02_retrieve_single_event_with_datapoints -v -s
```

**Test Database Import & Roundtrip**:
```bash
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_03_import_server_event_to_database -v -s
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_06_roundtrip_json_export -v -s
```

**Test Format & Value Validation**:
```bash
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_04_validate_datapoint_format_consistency -v -s
python3 -m pytest tests/test_datapoint_transfer.py::TestDatapointTransferFromServer::test_05_validate_datapoint_value_preservation -v -s
```

---

## Test Coverage

### Local Tests (No Server Required)
| Test | Purpose | Status |
|------|---------|--------|
| `test_datapoint_import_with_comprehensive_data` | Full datapoint structure with hr, o2Sat, rawData, rawData3D | ✅ |
| `test_datapoint_with_missing_fields` | Handle events with partial datapoint data | ✅ |
| `test_zero_datapoints` | Handle events with no datapoints gracefully | ✅ |

### Remote Server Tests
| Test | Purpose | Status |
|------|---------|--------|
| `test_01_server_connection` | Verify server connectivity and event availability | ✅ |
| `test_02_retrieve_single_event_with_datapoints` | Download event with datapoints from API | ✅ |
| `test_03_import_server_event_to_database` | Verify datapoint import and preservation | ✅ |
| `test_04_validate_datapoint_format_consistency` | Check event structure compatibility | ✅ |
| `test_05_validate_datapoint_value_preservation` | Verify numeric accuracy of datapoint values | ✅ |
| `test_06_roundtrip_json_export` | Test complete export pipeline | ✅ |

---

## Test Results Summary

### Key Findings ✅
- **69,150+ events** available on server
- **35 datapoints** successfully transferred per test event
- **100% datapoint preservation** through import/export cycle
- **Numeric accuracy** fully maintained (HR, O2Sat, accelerometer data)
- **Database normalization** working as designed

### Data Flow Validation
```
Server API (69,150 events)
  └─ Download with includeDatapoints=True
     └─ 35 datapoints per event
        └─ Import to SQLite database
           └─ Normalize and store in separate table
              └─ Export to JSON
                 └─ All 35 datapoints present
                    └─ Numeric values accurate ✅
```

---

## Test Classes

### DatapointTransferValidator Utility Class
Helper class for comparing event formats and validating data:
- `get_datapoint_stats()` - Extract field statistics
- `compare_events_format()` - Check field compatibility
- `validate_datapoint_values()` - Verify numeric accuracy

### TestDatapointTransferFromServer
Remote server integration tests requiring:
- Active server connection
- Valid credentials in client.cfg
- Network connectivity

**Setup**: Fetches 69,150+ events, selects test IDs, runs 6 tests

### TestDatapointTransferWithLocalData
Standalone tests using synthetic event data:
- No server required
- No network dependency
- Fast execution (<0.1 seconds)

---

## Expected Output

```
============================= test session starts ==============================
collected 9 items

test_datapoint_transfer.py::TestDatapointTransferFromServer::test_01_server_connection PASSED
test_datapoint_transfer.py::TestDatapointTransferFromServer::test_02_retrieve_single_event_with_datapoints PASSED
test_datapoint_transfer.py::TestDatapointTransferFromServer::test_03_import_server_event_to_database PASSED
test_datapoint_transfer.py::TestDatapointTransferFromServer::test_04_validate_datapoint_format_consistency PASSED
test_datapoint_transfer.py::TestDatapointTransferFromServer::test_05_validate_datapoint_value_preservation PASSED
test_datapoint_transfer.py::TestDatapointTransferFromServer::test_06_roundtrip_json_export PASSED
test_datapoint_transfer.py::TestDatapointTransferWithLocalData::test_datapoint_import_with_comprehensive_data PASSED
test_datapoint_transfer.py::TestDatapointTransferWithLocalData::test_datapoint_with_missing_fields PASSED
test_datapoint_transfer.py::TestDatapointTransferWithLocalData::test_zero_datapoints PASSED

============================== 9 passed in 30.90s ==============================
```

---

## Detailed Results

For comprehensive analysis, see: [DATAPOINT_TRANSFER_TEST_RESULTS.md](DATAPOINT_TRANSFER_TEST_RESULTS.md)

Key sections:
- Test Summary
- Key Findings (5 major discoveries)
- Root Cause Analysis
- Recommendations
- Data Flow Validation
- Conclusion

---

## Troubleshooting

### "Server not configured" Error
**Cause**: Missing or invalid credentials in `../client.cfg`

**Solution**:
```bash
# Check config file exists
ls -la ../client.cfg

# Ensure format is correct:
cat ../client.cfg
# Should contain:
# {
#   "baseurl": "https://osdapi.ddns.net/api",
#   "uname": "username",
#   "passwd": "password"
# }
```

### Connection Timeout
**Cause**: Server unreachable or network issue

**Solution**:
- Test connectivity: `ping osdapi.ddns.net`
- Check credentials are valid
- Run local tests only: `pytest tests/test_datapoint_transfer.py::TestDatapointTransferWithLocalData -v`

### Import Errors
**Cause**: Missing dependencies

**Solution**:
```bash
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor
source /home/graham/osd/OpenSeizureDatabase/venv/bin/activate
pip install -e ../../
```

---

## Conclusion

The test suite comprehensively validates that:
1. ✅ Datapoints are successfully transferred from remote server
2. ✅ All datapoint fields are preserved during import
3. ✅ Numeric values maintain full accuracy
4. ✅ Database normalization works correctly
5. ✅ Round-trip export maintains data integrity

**Status**: READY FOR PRODUCTION ✅
