# Analysis: Missing SubType Filtering in CNN-LSTM Training

## Issue Summary

The training configuration specifies that "Suspected" and "Check" seizure events should be excluded, but they are appearing in the training dataset and analysis results. Investigation reveals a configuration file error.

## Root Cause

### Configuration File Mismatch

**Expected Config (nnConfig_cnn.json):**
```json
"eventFilters": {
    "excludeSubTypes": [ "Check", "check",  "Suspected", "suspected" ],
    "_excludeSubTypes": [ "Aura", "aura", "Check", "check", "unknown", "Unknown", "null", "Null", "Suspected", "suspected"]
}
```

**Actual Config Used in Runs 5-7 (nnConfig_cnn_lstm_pytorch.json):**
```json
"eventFilters": {
    "excludeSubTypes": [ ],
    "_excludeSubTypes": [ "Aura", "aura", "Check", "check", "unknown", "Unknown", "null", "Null"]
}
```

### Filtering Logic

The filtering is implemented in `libosd/osdDbConnection.py` (lines 506-509):

```python
if (excludeSubTypes is not None):
    # Exclude Event subTypes
    matchingSubTypesLst = self.getMatchingElementsLst('subType', excludeSubTypes, debug)
    nAdded = libosd.osdUtils.removeEntriesFromLst(eventsLst, matchingSubTypesLst)
```

When `excludeSubTypes` is an empty list `[]`:
- The condition `excludeSubTypes is not None` is TRUE (empty list is not None)
- `getMatchingElementsLst('subType', [])` finds no matching subtypes (empty filter list)
- No events are removed (filtering against an empty list excludes nothing)

### Evidence from Training Log

```
selectData: filterCfg= {...'excludeSubTypes': [], '_excludeSubTypes': ['Aura', 'aura', 'Check', 'check', 'unknown', 'Unknown', 'null', 'Null'], ...}
selectData: %d events remaining after applying filters
nnTester.testModel(): Kept 135642 of 155215 rows after filtering (19573 removed)
Unique seizure subtypes: ['Tonic-Clonic' 'Other' nan 'Aura' 'Suspected']
```

The log shows:
- `excludeSubTypes` is EMPTY
- "Suspected" appears in unique seizure subtypes (2 events)
- "Check" events should also be missing but may have been removed by other filters

## Impact on Dataset

| Dataset Element | Actual Count | Expected (if filtered) |
|-----------------|--------------|------------------------|
| Seized events with subType="Suspected" | 2 | 0 |
| Seized events with subType="Check" | Unknown | 0 |

From the event analysis report for Run 5:
```
Seizure SubType Analysis:
     SubType  Count  TP  FN      TPR
Tonic-Clonic     61  58   3 0.950820
        Aura     19  17   2 0.894737
       Other     21  17   4 0.809524
   Suspected      2   1   1 0.500000
```

The 2 "Suspected" seizures should not be in the training or test data.

## Root Cause Analysis

When `nnConfig_cnn_lstm_pytorch.json` was created from `nnConfig_cnn.json`:
1. The active `excludeSubTypes` field was cleared (set to `[]`)
2. The original value was moved to a commented field `_excludeSubTypes` (underscore prefix indicates comment/documentation only)
3. The code only uses the active field, not the underscore-prefixed version
4. This was likely an oversight during CNN-LSTM configuration setup

## Recommendations for Fix

To restore proper filtering, the configuration file `nnConfig_cnn_lstm_pytorch.json` should be updated to:

```json
"eventFilters": {
    "excludeSubTypes": [ "Check", "check", "Suspected", "suspected" ],
    "_excludeSubTypes": [ "Aura", "aura", "Check", "check", "unknown", "Unknown", "null", "Null", "Suspected", "suspected"]
}
```

### Files Affected
- `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/nnConfig_cnn_lstm_pytorch.json` (primary config)
- Any future CNN-LSTM training runs using this configuration

### Re-training Needed
After fixing the configuration, the model should be re-trained to ensure Suspected and Check events are properly excluded.

### Impact Assessment
- **Minor:** Only 2 "Suspected" seizures were in the dataset (~1.9% of 104 test seizures in Run 5)
- **But:** These represent data quality issues that should be excluded
- **Recommendation:** Fix and re-train to ensure clean dataset

## Files Reviewed
- `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/output/cnnLstmModel_pytorch/5/runSequence_20260827_203723.log`
- `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/output/cnnLstmModel_pytorch/5/nnConfig_cnn_lstm_pytorch.json`
- `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/nnConfig_cnn.json` (reference)
- `/home/graham/osd/OpenSeizureDatabase/libosd/osdDbConnection.py` (lines 506-509)
- `/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2/selectData.py` (lines 74-82)

---

**Analysis Date:** August 29, 2026  
**Status:** Waiting for approval before making changes
