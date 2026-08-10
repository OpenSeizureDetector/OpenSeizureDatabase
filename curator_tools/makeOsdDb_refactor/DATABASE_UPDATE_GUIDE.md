# Database Update & Maintenance Guide

## Overview

This guide covers the database update process and tools for managing duplicate merge notes that can accumulate when events are merged during repeated database updates.

## Issue: Duplicate Merge Notes

### Problem
When the database update process merges events from multiple sources, it adds a note to the event's `desc` field:
```
Includes data from merged event(s): 123, 456, 789
```

On repeated runs, if the same events are re-processed, the **same note was being appended again**, causing the description field to grow with duplicates:
```
Original: "Sample event description"
After 1st update: "Sample event description. Includes data from merged event(s): 456, 789"
After 2nd update: "Sample event description. Includes data from merged event(s): 456, 789. Includes data from merged event(s): 456, 789"
After 3rd update: "Sample event description. Includes data from merged event(s): 456, 789. Includes data from merged event(s): 456, 789. Includes data from merged event(s): 456, 789"
```

### Root Cause
In `src/event_grouping.py`, the `merge_grouped_events()` function was not checking if the merge note already existed before appending it.

### Solution Applied

#### Fix 1: Prevent Future Duplicates
**File: `src/event_grouping.py` (lines ~100-125)**

The `merge_grouped_events()` function now checks if the merge note already exists before appending:

```python
# Check if this exact merge note already exists to avoid duplicates
if merge_note not in current_desc:
    if current_desc and not current_desc.endswith('.'):
        current_desc += '.'
    if current_desc:
        current_desc += ' '
    
    merged['desc'] = current_desc + merge_note
else:
    # Note already exists, keep desc as is
    merged['desc'] = current_desc
```

This ensures that on repeated runs, the same merge note won't be added multiple times.

## Tools for Managing Existing Duplicates

### Option 1: GUI Cleanup Tool (Recommended)

#### Steps:
1. Open event_editor.py
2. Open your OSDB database file (File → Open Database)
3. Go to **Tools → Clean Duplicate Merge Notes...**
4. Confirm that you want to proceed
5. Wait for the cleanup to complete (progress dialog shows status)

#### What It Does:
- Scans all events in the database
- Identifies events with duplicate merge notes
- Removes all but the first occurrence of each merge note
- Updates the database automatically
- Shows a summary of how many events were cleaned

#### Example Output:
```
Cleanup Complete
Events cleaned: 47
Total events scanned: 1523
```

### Option 2: Command-Line Cleanup Script

You can also create a standalone cleanup script for batch processing:

```python
#!/usr/bin/env python3
import sys
import re
sys.path.insert(0, 'src')
from osdb_sqlite import OsdWorkingDb

def clean_duplicate_notes(db_path):
    db = OsdWorkingDb(db_path)
    all_events = db.get_events(include_datapoints=False)
    
    cleaned_count = 0
    merge_note_pattern = r"Includes data from merged event\(s\): [\d, ]+"
    
    for i, event in enumerate(all_events):
        if i % 100 == 0:
            print(f"Processing event {i}/{len(all_events)}...")
        
        desc = event.get('desc', '')
        if not desc:
            continue
        
        # Find all occurrences
        matches = list(re.finditer(merge_note_pattern, desc))
        if len(matches) > 1:
            # Keep only first occurrence
            for match in reversed(matches[1:]):
                start, end = match.span()
                if start > 0 and desc[start-1] in ' .':
                    start -= 1
                desc = desc[:start] + desc[end:]
            
            event['desc'] = desc
            db.update_event(event)
            cleaned_count += 1
    
    db.close()
    print(f"\nCleaned {cleaned_count} events out of {len(all_events)}")

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'osdb_working.db'
    clean_duplicate_notes(db_path)
```

## Database Update Process

### GUI Method (New in event_editor.py)

#### Steps:
1. Open event_editor.py
2. Open your OSDB database file (File → Open Database)
3. Go to **Tools → Update Database...**
4. Confirm that you want to proceed (will show a warning about download time)
5. Wait for the update to complete (may take several minutes)

#### What It Does:
- Closes the current database safely
- Runs `makeOsdDb_refactored_wrapper.py` with the correct configuration
- Downloads new events from configured data sources
- Merges events using the sliding-window grouping algorithm
- Reopens the database and reloads events
- Shows completion status

#### Requirements:
- Configuration file must be present at: `../osdb.cfg` (relative to database directory)
- Internet connection to download from data sources
- Sufficient disk space for temporary files

### Command-Line Method

You can also run the update directly from the terminal:

```bash
# Navigate to the refactored folder
cd /path/to/makeOsdDb_refactor

# Activate virtual environment
source ../venv/bin/activate

# Run the update wrapper
python3 makeOsdDb_refactored_wrapper.py \
    --config ../osdb.cfg \
    --osdb-dir /path/to/osdb/data
```

## Recommended Workflow

### First Time Setup
1. Create or open your OSDB database
2. Run **Tools → Update Database** to download initial data
3. If duplicates are created, run **Tools → Clean Duplicate Merge Notes** immediately

### Regular Maintenance
1. Periodically run **Tools → Update Database** to fetch new events (recommended: weekly or monthly)
2. After each update, run **Tools → Clean Duplicate Merge Notes** as preventive maintenance
3. Monitor database size in File properties

### Batch Automation
For automated updates via cron job, use the command-line method with a shell script:

```bash
#!/bin/bash
cd /path/to/makeOsdDb_refactor
source ../venv/bin/activate
python3 makeOsdDb_refactored_wrapper.py --config ../osdb.cfg --osdb-dir /path/to/data
python3 -c "
import sys
sys.path.insert(0, 'src')
from osdb_sqlite import OsdWorkingDb
db = OsdWorkingDb('/path/to/data/osdb_working.db')
# Run cleanup...
"
```

## Implementation Details

### Files Modified

#### 1. `src/event_grouping.py`
- **Function:** `merge_grouped_events()` (lines ~100-125)
- **Change:** Added check for existing merge notes before appending
- **Impact:** Prevents duplicate notes on repeated runs

#### 2. `event_editor.py`
- **Menu Addition:** Tools → Update Database...
- **Menu Addition:** Tools → Clean Duplicate Merge Notes...
- **Methods Added:**
  - `run_database_update()` - Runs database update via subprocess
  - `clean_duplicate_notes()` - Cleans duplicate merge notes
- **Actions Enabled/Disabled:** When database is opened/closed

### Database Schema Requirements

The cleanup and update tools require these fields in the events table:
- `id` - Event identifier
- `desc` - Event description (text field)
- All fields that `OsdWorkingDb.update_event()` requires

### Performance Considerations

- **Cleanup Speed:** ~50-100 events per second (depends on desc field length)
- **Update Speed:** Highly dependent on network speed and number of new events
- **Memory Usage:** ~1 KB per event for the cleanup process

## Troubleshooting

### Issue: "Config file not found" during update
- **Solution:** Ensure `osdb.cfg` exists in the parent directory
- **Example Path:** `/path/to/osdb.cfg`

### Issue: Update fails with "Permission denied"
- **Solution:** Ensure you have write permissions to the database directory
- **Command:** `chmod 755 /path/to/osdb/directory`

### Issue: Cleanup shows 0 events cleaned but desc field is long
- **Possible Cause:** Merge notes use different formatting
- **Solution:** Check actual note format with: `sqlite3 osdb_working.db "SELECT DISTINCT substr(desc, instr(desc, 'Includes')) FROM events WHERE desc LIKE '%Includes%' LIMIT 5;"`

### Issue: Database locked error during update/cleanup
- **Cause:** Another process has the database open
- **Solution:** Close event_editor and all other tools using the database

## Future Enhancements

Possible improvements for future versions:
1. Batch cleanup of all databases in a directory
2. Automatic cleanup after each update
3. Detailed log of which events were merged/cleaned
4. Backup creation before major operations
5. Dry-run mode to preview changes without modifying database
6. Integration with scheduled tasks/cron

## References

- **merge_grouped_events():** Handles event grouping and merging logic
- **OsdWorkingDb:** Database interface for reading/writing events
- **makeOsdDb_refactored_wrapper.py:** Main update script that downloads and processes events
