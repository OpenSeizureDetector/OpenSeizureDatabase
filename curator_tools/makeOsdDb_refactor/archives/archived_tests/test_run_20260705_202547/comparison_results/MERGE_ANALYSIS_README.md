# Merge Analysis Spreadsheets - README

## Overview

This directory contains detailed analysis of the 76 events that were merged by the refactored version's sliding window grouping algorithm compared to the original version's time-bin grouping.

## Files

### 1. **merge_comparison_detailed.csv** ⭐ RECOMMENDED
**Best file for manual review**

Side-by-side comparison showing:
- **REMOVED_*** columns: Event that was removed/merged
  - id, userId, dataTime, type, subType, desc, alarmState, datapoints count
- **TIME_DIFF_minutes**: How far apart the events are (within 3 min window)
- **MERGED_INTO_*** columns: Event it was likely merged into
  - id, userId, dataTime, type, subType, desc, alarmState, datapoints count
- **REVIEW_STATUS**: Column for you to mark (PENDING/APPROVED/REJECTED)
- **REVIEW_NOTES**: Column for your notes

**Total:** 76 rows (63 with clear merge target, 13 need investigation)

**To open:**
```bash
libreoffice --calc merge_comparison_detailed.csv
# or
xdg-open merge_comparison_detailed.csv
```

### 2. **merged_events_detail.csv**
Alternative format with single-row per removed event.

Contains: removed event details, likely merge target ID and time, time difference, number of nearby events.

## Key Findings

### Merge Statistics
- **Total events merged:** 76
- **Events with clear merge target:** 63 (83%)
- **Events needing investigation:** 13 (17%)

### By User
- User 39: 32 events merged
- User 1643: 28 events merged  
- User 45: 12 events merged
- User 8: 2 events merged
- User 1246: 1 event
- User 1422: 1 event

### Typical Merge Scenarios

**Example 1: Aura followed by Tonic-Clonic**
- Event 26077 (Tonic-Clonic at 00:06:08) merged into
- Event 26071 (Aura at 00:03:16)
- Time difference: 2.87 minutes
- This is likely correct - aura typically precedes tonic-clonic

**Example 2: Close temporal events**
- Event 26992 (Tonic-Clonic at 16:30:04) merged into  
- Event 26988 (Aura at 16:28:20)
- Time difference: 1.73 minutes

### Events Needing Investigation ⚠️

13 events have **NO_MATCH** - no nearby event found within 3 minutes. 

**CRITICAL FINDING:** After investigation, these events appear to have been **incorrectly removed**:

- **4 events** (5486, 6590, 6668, 21569) have NO similar events within 10 minutes
  - These were completely lost, not merged
  - **This is likely a BUG in the refactored version**
  
- **9 events** were merged into events 3-6 minutes away
  - Beyond the stated 3-minute threshold
  - Due to sliding window "chaining" effect
  - Example: Event 1328552 merged into 1328546 (4.0 min away)

**See [NO_MATCH_INVESTIGATION_REPORT.md](NO_MATCH_INVESTIGATION_REPORT.md) for full details.**

**Action needed:** 
1. Review these 13 events to determine if data loss is acceptable
2. Consider fixing the grouping algorithm to prevent loss
3. Possibly add event ID preservation to merged events

## How to Review

### Recommended Process

1. **Open the spreadsheet:**
   ```bash
   libreoffice --calc merge_comparison_detailed.csv
   ```

2. **For each row, verify:**
   - Are both events from the same user? ✓
   - Are they close in time? (< 3 minutes) ✓
   - Are they related seizure types? (e.g., Aura → Tonic-Clonic)
   - Does merging make clinical sense?

3. **Mark your review:**
   - Set `REVIEW_STATUS` to:
     - `APPROVED` - merge is correct
     - `REJECTED` - merge should not have happened
     - `UNCERTAIN` - needs clinical review
   - Add notes in `REVIEW_NOTES` column

4. **Focus on NO_MATCH events:**
   These need special attention to understand why they were removed.

### Questions to Ask

- **For Aura + Tonic-Clonic merges:** Is the aura typically a precursor? (Usually yes → APPROVED)
- **For same subType merges:** Are these likely continuation of same event? (Check descriptions)
- **For different subTypes:** Does the sequence make clinical sense?
- **For NO_MATCH:** Why was this event removed? Check original JSON for details.

## Clinical Context

### Typical Seizure Progressions
Common sequences that should be merged:
- Aura → Tonic-Clonic (most common)
- Partial → Generalized
- Multiple close events of same type (likely single event with breaks)

### Sequences That Shouldn't Merge
- Two distinct tonic-clonic seizures 2+ minutes apart
- Different event types with different alarm states
- Events with detailed descriptions indicating they are separate

## Next Steps

After review:
1. Save your marked-up spreadsheet
2. If you find merges that should be rejected, document the event IDs
3. Consider whether the 3-minute window needs adjustment
4. Decide if the sliding window algorithm needs refinement

## Technical Details

**Grouping Algorithm (Refactored):**
- Groups events by: userId, type, dataTime proximity
- Window: 3 minutes
- Method: Sliding window (events < 3min from each neighbor)
- More aggressive than original time-bin method

**Original Algorithm:**
- Groups events by: userId, type, 3-minute time bins
- Method: Fixed time windows (pandas GroupBy with freq='3min')
- Less aggressive - events in different bins aren't merged

## Files Generated By

- `create_merge_comparison_spreadsheet.py` - Main comparison file
- `analyze_merged_events.py` - Alternative detail file

Generated: July 5, 2026
