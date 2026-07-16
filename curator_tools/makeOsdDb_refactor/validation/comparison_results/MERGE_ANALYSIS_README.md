# Merge Analysis Spreadsheet

**Generated:** 2026-07-05 20:31:05

## Summary

- **Total events**: 76
- **Merged**: 76
- **Removed**: 0
- **Within 3-min threshold**: 67
- **Beyond 3-min threshold**: 9

## File: merge_analysis.csv

This spreadsheet shows all events from the original database that are no longer
primary events in the refactored version.

### Columns

**Removed Event (Original):**
- `REMOVED_ID`: Event ID that's no longer primary
- `REMOVED_USER`: User ID
- `REMOVED_TIME`: Event timestamp
- `REMOVED_TYPE`: Event type/subtype
- `REMOVED_DATASOURCE`: Data source name
- `REMOVED_DATAPOINTS`: Number of datapoints

**Merge Information:**
- `TIME_DIFF_minutes`: Time difference between events (minutes)
- `WITHIN_3MIN`: YES/NO - is time difference within 3-minute threshold?

**Merge Target (Refactored):**
- `MERGED_INTO_ID`: ID of the primary event this was merged into
- `MERGED_INTO_*`: Details about the merge target event
- `MERGE_COUNT`: How many events were merged into the target
- `TOTAL_DATAPOINTS_AFTER`: Total datapoints after merge
- `TARGET_WAS_ORIGINAL`: Was the merge target also from original database?

**Status:**
- `STATUS`: MERGED (merged into another event) or REMOVED (truly removed)

### Notes

- **All merged events are preserved** in the `_merged_from_event_ids` field
- Events merged beyond 3 minutes are due to sliding window "chaining"
- When `TARGET_WAS_ORIGINAL=YES`, both events were from the published database
- Sort by `TIME_DIFF_minutes` to see events merged at different distances
- Filter by `WITHIN_3MIN=NO` to review merges beyond threshold
