# seizureTimes Fix Summary

## Issues Identified

1. **seizureTimes showing as zero**: Event 1046 has `seizureTimes: [-80.0, -55.0]` but they were displaying as `[0.0, 0.0]` in the UI
2. **Graph X-axis incorrect**: X-axis was showing seconds from start of first datapoint instead of seconds relative to event dataTime

## Root Causes

### Issue 1: seizureTimes Not Loading
- **Problem**: Code was looking for `seizureTimes` in the `metadata` JSON column
- **Reality**: `seizureTimes` are stored in a dedicated `seizureTimes` TEXT column (as JSON array)
- **Additionally**: QDoubleValidator had minimum=0.0, preventing negative values from being entered

### Issue 2: X-axis Calculation Wrong  
- **Problem**: Time axis was calculated as `i * 5` (index × 5 seconds), starting from 0
- **Reality**: Each datapoint has its own `dataTime` timestamp, which must be compared to the event's `dataTime` to get relative time
- **Example**: 
  - Event dataTime: `2022-03-21 23:23:56`
  - First datapoint: `2022-03-21 23:22:46`
  - Relative time: -70 seconds (not 0)

## Fixes Applied

### Fix 1: Load seizureTimes from Correct Column
**File**: `event_editor.py`

#### A. Parse seizureTimes in `get_event_details()` method
```python
# Parse seizureTimes from dedicated column (takes precedence over metadata)
if event.get('seizureTimes'):
    try:
        event['seizureTimes'] = json.loads(event['seizureTimes'])
    except (json.JSONDecodeError, TypeError):
        pass
```

#### B. Save seizureTimes to correct column in `update_event()` method
```python
# Prepare seizureTimes for dedicated column
seizure_times_json = None
if seizure_times is not None:
    seizure_times_json = json.dumps(seizure_times)

# Update database (seizureTimes in dedicated column, not metadata)
cursor.execute(
    """UPDATE events 
       SET type = ?, subType = ?, desc = ?, metadata = ?, seizureTimes = ?
       WHERE id = ?""",
    (event_type, subtype, description, json.dumps(metadata), seizure_times_json, event_id)
)
```

### Fix 2: Allow Negative Values
**File**: `event_editor.py`

Changed QDoubleValidator minimum from `0.0` to `-999999.0`:

```python
# Start time validator
self.seizure_start_edit.setValidator(QDoubleValidator(-999999.0, 999999.0, 1))

# End time validator  
self.seizure_end_edit.setValidator(QDoubleValidator(-999999.0, 999999.0, 1))
```

Removed `max(0.0, ...)` constraints in `adjust_seizure_time()`:

```python
# Before: new_val = max(0.0, current + adjustment)
# After:  new_val = current + adjustment
```

### Fix 3: Calculate X-axis Relative to Event dataTime
**File**: `event_editor.py`

Changed `plot_event_data()` to calculate relative time:

```python
# Get event dataTime as reference point
from datetime import datetime
event_dt = datetime.fromisoformat(self.current_event['dataTime'].replace('Z', '+00:00'))

for i, dp in enumerate(datapoints):
    # Calculate time relative to event dataTime
    dp_dt = datetime.fromisoformat(dp['dataTime'].replace('Z', '+00:00'))
    time_sec = (dp_dt - event_dt).total_seconds()
    time_points.append(time_sec)
    # ... rest of processing

# Create time array for raw data samples (125 samples per 5-second datapoint)
start_time = time_points[0]
end_time = time_points[-1] + 5
raw_time = np.linspace(start_time, end_time, len(raw_data_list))
```

**Result**: X-axis now correctly shows negative times for datapoints that occurred before the event dataTime.

## Documentation Updates

Updated `event_editor/README.md` to clarify:

1. **Quick Start section**: Added note that seizure times "can be negative" and example
2. **Data Visualization section**: Added "X-axis: Time in seconds relative to event dataTime (negative = before event, positive = after)"
3. **Database Schema section**: Clarified that seizureTimes are in a dedicated column, not metadata

## Verification

### Test with Event 1046
```bash
python3 test_seizure_times.py
```

**Results**:
- ✓ Event dataTime: `2022-03-21 23:23:56`
- ✓ seizureTimes correctly loaded: `[-80.0, -55.0]`
- ✓ First datapoint at `-70.0s` relative to event (correctly between seizure start and end)
- ✓ Seizure started 80 seconds BEFORE event dataTime
- ✓ Seizure ended 55 seconds BEFORE event dataTime

### Expected Behavior in GUI

When opening event 1046:
1. **Start Time field**: Shows `-80.0`
2. **End Time field**: Shows `-55.0`
3. **Acceleration graph**:
   - X-axis extends from approximately -70s to +30s
   - Red shaded region from -80s to -55s
   - Vertical markers at -80s (start) and -55s (end)
4. **Can adjust times**: +/-5s buttons work with negative values

## Files Modified

1. `/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/event_editor/event_editor.py`
   - `get_event_details()`: Parse seizureTimes from dedicated column
   - `update_event()`: Save seizureTimes to dedicated column
   - `QDoubleValidator` creation (2 places): Allow negative values
   - `adjust_seizure_time()`: Remove max(0.0, ...) constraint
   - `plot_event_data()`: Calculate X-axis relative to event dataTime

2. `/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor/event_editor/README.md`
   - Quick Start: Clarify negative times supported
   - Data Visualization: Document X-axis is relative to event dataTime
   - Database Schema: Update seizureTimes location (dedicated column vs metadata)

## Testing Commands

```bash
# Syntax check
python3 -m py_compile event_editor.py

# Verify seizureTimes loading
python3 test_seizure_times.py

# Launch GUI
python3 event_editor.py --db /home/graham/osd/osdb/osdb_working.db
```

## Migration Notes

**No database migration needed** - the `seizureTimes` column already exists in the schema and contains the correct data. The issue was purely in the UI code not reading from the correct column.

**Backward compatibility**: If any seizureTimes were saved to `metadata` JSON (unlikely due to the bug), they won't be displayed, but no data will be lost. Users can re-enter them and they'll be saved to the correct column.

## Summary

All issues resolved:
- ✅ seizureTimes now load from correct database column
- ✅ Negative values are properly supported in UI validators
- ✅ Graph X-axis shows time relative to event dataTime
- ✅ Seizure period correctly displayed as shaded region with markers
- ✅ Documentation updated with examples and clarifications
- ✅ Test script confirms correct behavior

The event editor now correctly handles seizure times that occur before the event dataTime, which is common when the event timestamp is set to when an alarm triggered rather than when the seizure actually started.
