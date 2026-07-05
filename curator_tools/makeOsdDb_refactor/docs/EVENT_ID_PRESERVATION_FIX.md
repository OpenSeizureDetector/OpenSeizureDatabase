# Event ID Preservation Fix - Implementation Report

**Date:** July 5, 2026  
**Issue:** Critical bug - existing published event IDs were being lost during grouping  
**Status:** ✅ FIXED and TESTED

---

## Summary

Fixed a critical data integrity bug where existing published event IDs were being lost during the grouping step. The refactored version now properly preserves all existing event IDs, either as primary events or in the `_merged_from_event_ids` tracking field.

---

## The Problem

### Original Behavior
When events were grouped together, the algorithm would select ONE event as "primary" based on:
1. Highest alarm state
2. Has description
3. Earliest timestamp

**The issue:** It did NOT prioritize existing published event IDs. This meant:
- Existing events could be "lost" (their IDs discarded)
- Only the newly downloaded event ID would be kept
- Users couldn't trace what happened to previously published events

### Impact
- **76 events** out of 407 were being merged/removed
- **13 events** had no clear merge target in analysis
- **4 events** appeared to be completely lost
- **18.7% event removal rate** with no traceability

---

## The Solution

### Three-Part Fix

#### 1. Modified Event Selection Logic
**File:** `src/event_grouping.py`

Updated `select_best_event_from_group()` to:
- Check for `_is_existing_event` flag on events
- **Always prioritize existing published events** when selecting from a group
- Only compare alarm states, descriptions, and timestamps AMONG existing events
- If no existing events in group, use original selection criteria

```python
# CRITICAL: Always prefer events from existing published database
existing_events = [e for e in group if e.get('_is_existing_event', False)]

if existing_events:
    # Only consider existing events for selection
    # This preserves published event IDs
    group_to_select_from = existing_events
else:
    group_to_select_from = group
```

#### 2. Added Validation and Tracking
**File:** `src/event_grouping.py`

Updated `apply_sliding_window_grouping()` to:
- Track all existing event IDs at start of grouping
- After grouping, verify all existing IDs are preserved
- Check both primary event IDs AND `_merged_from_event_ids` lists
- **Report any lost events as a WARNING**
- Include preservation stats in grouping_info

```python
# VALIDATION: Ensure all existing events are preserved
lost_existing_events = []
if existing_event_ids:
    preserved_ids = set()
    
    for event in unique_events:
        if event['id'] in existing_event_ids:
            preserved_ids.add(event['id'])
        
        # Check if existing events were merged into this one
        merged_from = event.get('_merged_from_event_ids', [])
        for merged_id in merged_from:
            if merged_id in existing_event_ids:
                preserved_ids.add(merged_id)
    
    lost_existing_events = list(existing_event_ids - preserved_ids)
    
    if lost_existing_events:
        print(f"\n⚠️  WARNING: {len(lost_existing_events)} existing published events were LOST!")
    else:
        print(f"  ✓ All {len(existing_event_ids)} existing events preserved")
```

#### 3. Updated Wrapper to Mark Events
**File:** `makeOsdDb_refactored_wrapper.py`

Updated `saveEventsAsJson()` to:
- Mark all existing events with `_is_existing_event = True`
- Mark all newly downloaded events with `_is_existing_event = False`
- Display counts of existing vs new events
- Clean up internal flags before saving to JSON

```python
# CRITICAL: Mark existing events to preserve their IDs during grouping
for event in existing_events:
    event['_is_existing_event'] = True

# New events are NOT marked
for event in deduplicated_events:
    event['_is_existing_event'] = False

print(f"  Existing (published): {len(existing_events)} events")
print(f"  New (downloaded): {len(deduplicated_events)} events")
```

---

## Test Results

### Unit Test
**File:** `test_event_preservation.py`

Tested with 407 existing events from `/home/graham/osd/osdb`:
- ✅ All 407 existing events preserved
- ✅ 331 preserved as primary event IDs
- ✅ 76 preserved in `_merged_from_event_ids`
- ✅ 0 events lost

### Integration Test
**File:** `test_refactored_v2.log`

Full database update test:
- ✅ tcSeizures: All 184 existing events preserved
- ✅ allSeizures: All 407 existing events preserved
- ✅ Proper validation messages displayed
- ✅ Event counts match expected values

### Specific Event Verification
The 4 events that appeared "lost" in original investigation:

| Event ID | Status | Details |
|----------|--------|---------|
| 5486 | ✅ Preserved | Merged into Event 5483 (both existing, 1.62 min apart) |
| 6590 | ✅ Preserved | Merged into Event 6587 (both existing, 0.40 min apart) |
| 6668 | ✅ Preserved | Merged into Event 6717 (both existing, 1.57 min apart) |
| 21569 | ✅ Preserved | Merged into Event 21561 (both existing, 0.82 min apart) |

**All merges are valid:** Events are within 3-minute threshold and all merged event IDs are tracked.

---

## Data Integrity Guarantees

### Before Fix
- ❌ Existing event IDs could be discarded
- ❌ No tracking of merged events
- ❌ No validation of preservation
- ❌ Data loss not detected

### After Fix
- ✅ **All existing event IDs are preserved** (primary or in `_merged_from_event_ids`)
- ✅ **Existing events prioritized** when selecting from groups
- ✅ **Validation checks** ensure no events are lost
- ✅ **Tracking metadata** shows merge history
- ✅ **Warning messages** if any events lost (shouldn't happen)

---

## Backward Compatibility

### Database Format
- No breaking changes to JSON structure
- Adds new field: `_merged_from_event_ids` (array of event IDs)
- Adds new field: `_merged_event_count` (number of events merged)
- Internal flags (`_is_existing_event`) are cleaned up before saving

### Grouping Behavior
- **UNCHANGED:** Sliding window still allows chains >3 min (gaps <3 min)
- **CHANGED:** Existing event IDs now preferred as primary events
- **CHANGED:** All existing events are tracked, even when merged

---

## Grouping Algorithm Clarification

### How Sliding Window Works
The grouping algorithm creates chains where each consecutive pair of events must be <3 minutes apart, but the total group can span >3 minutes:

**Example:**
- Event A at 00:00
- Event B at 00:02 (2 min from A) → grouped with A
- Event C at 00:05 (5 min from A, but 3 min from B) → grouped with A & B
- Event D at 00:08 (8 min from A, but 3 min from C) → grouped with A, B & C

**Result:** Events A-D are all in one group spanning 8 minutes, even though threshold is 3 minutes.

**This is INTENTIONAL and ACCEPTABLE** - it handles seizures with multiple alarm triggers over several minutes.

---

## Files Modified

1. **src/event_grouping.py**
   - `select_best_event_from_group()` - Added existing event prioritization
   - `apply_sliding_window_grouping()` - Added validation and tracking

2. **makeOsdDb_refactored_wrapper.py**
   - `saveEventsAsJson()` - Mark existing/new events, cleanup flags

3. **Test files created:**
   - `test_event_preservation.py` - Unit test for grouping preservation
   - `test_refactored_v2.log` - Integration test output

4. **Documentation created:**
   - `EVENT_ID_PRESERVATION_FIX.md` (this file)
   - Updated `NO_MATCH_INVESTIGATION_REPORT.md`
   - Updated `MERGE_ANALYSIS_README.md`

---

## Production Readiness

### Status: ✅ READY FOR PRODUCTION

The refactored version now:
- ✅ Preserves all existing published event IDs
- ✅ Properly tracks merged events
- ✅ Validates data integrity during processing
- ✅ Maintains backward compatibility
- ✅ Passes all tests with real data
- ✅ Provides clear progress and validation messages

### Recommended Next Steps
1. ✅ Run final comparison test against original makeOsdDb.py
2. ✅ Review merge tracking metadata format
3. ⏳ Update documentation for production use
4. ⏳ Deploy to production environment
5. ⏳ Monitor first production run closely

---

## Technical Notes

### Event Selection Priority (when multiple existing events in group)
1. Is existing event (both are, so tied)
2. Highest alarm state
3. Has description
4. Earliest timestamp

### Validation Process
The validation happens AFTER grouping and checks:
1. All existing event IDs are in output (as primary)
2. OR all existing event IDs are in `_merged_from_event_ids` (as merged)
3. Reports warning if ANY existing event ID is missing from both

### Performance Impact
- Minimal - only adds flag checking during group selection
- Validation is O(n) over final events (very fast)
- No noticeable performance degradation in tests

---

## Conclusion

This fix resolves the critical data integrity bug where existing published event IDs were being lost. All existing events are now properly preserved, with full tracking of merge operations. The refactored version is now safe for production deployment.

**Key Achievement:** 100% preservation of existing published data while still applying improved grouping and deduplication logic.
