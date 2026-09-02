# Detecting Existing Local Changes

## Overview

If you've already made edits to events in your database **before implementing the local change preservation feature**, you need to run the change detection script to mark those existing edits. This ensures they won't be overwritten when you run future updates.

**Why?** The schema migration adds new tracking columns, but can't know about changes you made in the past. The detection script compares your database with the source JSON files to find and mark those existing edits.

---

## When to Run Detection

Run the detection script if:

✅ You have an existing database with events  
✅ You've already made edits to event types, descriptions, or seizure times  
✅ You want to preserve those edits in future updates  

You DON'T need to run it if:

❌ Your database is fresh with no manual edits  
❌ You're starting from scratch  
❌ You only want to preserve changes made AFTER implementing this feature  

---

## Quick Start

### 1. Dry Run (No Changes)

First, always do a dry run to see what will be detected:

```bash
cd /home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor
python3 src/detect_local_changes.py --db /home/graham/osd/osdb/osdb_working.db \
    --json-dir /home/graham/osd/osdb --dry-run
```

**Example Output:**
```
Loading events from 6 JSON files...
  Reading osdb_3min_allSeizures.json... ✓ (1200 events)
  Reading osdb_3min_tcSeizures.json... ✓ (250 events)
  ...

Total unique events loaded from JSON: 1200

Comparing database with JSON...

======================================================================
CHANGE DETECTION SUMMARY
======================================================================
Total events in database: 1250
Events in JSON files: 1200
Events NOT in JSON (local only): 50

Events with local changes: 45

Changes by field:
  type: 12 events
  desc: 28 events
  subType: 5 events
  seizureTimes: 8 events

First 10 events with changes:
----------------------------------------------------------------------

Event ID 12345: type, desc
  type:
    JSON: Seizure
    DB:   False Alarm
  desc:
    JSON: 
    DB:   Corrected based on chart review

Event ID 12346: desc
  desc:
    JSON: 
    DB:   Patient report: ~3 min duration

... and 35 more events with changes
======================================================================

[DRY RUN] Would update 45 events
Run without --dry-run to apply changes
```

### 2. Apply Detection

Once you've reviewed the dry run and confirmed the changes look correct, run without `--dry-run`:

```bash
python3 src/detect_local_changes.py --db /home/graham/osd/osdb/osdb_working.db \
    --json-dir /home/graham/osd/osdb
```

**Output:**
```
Loading events from 6 JSON files...
Total unique events loaded from JSON: 1200

Comparing database with JSON...

======================================================================
CHANGE DETECTION SUMMARY
======================================================================
Total events in database: 1250
Events in JSON files: 1200
Events NOT in JSON (local only): 50

Events with local changes: 45

Changes by field:
  type: 12 events
  desc: 28 events
  subType: 5 events
  seizureTimes: 8 events

First 10 events with changes:
...

======================================================================
✓ Updated 45 events with local change tracking
```

✅ Done! Your existing edits are now marked and will be preserved.

---

## Usage Options

### Process All JSON Files in Directory

```bash
python3 src/detect_local_changes.py --db /path/to/osdb_working.db \
    --json-dir /home/graham/osd/osdb
```

Loads all `osdb_*.json` files from the directory and compares.

### Process Single JSON File

```bash
python3 src/detect_local_changes.py --db /path/to/osdb_working.db \
    --json-file /home/graham/osd/osdb/osdb_3min_tcSeizures.json
```

Useful if you only want to check against specific event types.

### Dry Run (Recommended First Step)

```bash
python3 src/detect_local_changes.py --db /path/to/osdb_working.db \
    --json-dir /home/graham/osd/osdb --dry-run
```

Shows what would be detected without making any database changes.

---

## Understanding the Output

### CHANGE DETECTION SUMMARY Section

**Total events in database**: Total events currently in your database

**Events in JSON files**: Events from JSON that exist in database (can be compared)

**Events NOT in JSON (local only)**: Events you've added that aren't in the JSON source
- These are events you created locally (not from the API)
- They're not compared (can't tell what changed)
- They remain in database unchanged

**Events with local changes**: Events that differ between JSON and database
- These are the edits you made to existing events
- Will be marked with local change tracking
- Won't be overwritten in future updates

### Changes by Field

Shows breakdown by field type:

| Field | Meaning |
|-------|---------|
| `type` | Event type changed (e.g., "Unknown" → "Seizure") |
| `subType` | Event subtype changed (e.g., null → "Tonic-Clonic") |
| `desc` | Description added or modified |
| `seizureTimes` | Seizure time markers adjusted |

### First N Events with Changes

Shows detailed before/after for each change:

```
Event ID 12345: type, desc
  type:
    JSON: Seizure          # What's in the source JSON file
    DB:   False Alarm      # What's in your database (your local edit)
```

This tells you:
- The event ID
- Which fields you edited
- What the original JSON value was
- What you changed it to

---

## Common Scenarios

### Scenario 1: Bulk Edit Many Events (Type Corrections)

You found that many events were misclassified and corrected them all manually.

**Dry run output:**
```
Events with local changes: 87
Changes by field:
  type: 87 events
```

**Action**: Run without `--dry-run` to mark all 87 events. Future updates won't overwrite your type corrections.

### Scenario 2: Added Descriptions to Events

You reviewed events and added detailed descriptions to some.

**Dry run output:**
```
Events with local changes: 156
Changes by field:
  desc: 156 events
```

**Action**: Run detection to preserve all 156 descriptions.

### Scenario 3: Mixed Edits

You made various edits to different fields:

**Dry run output:**
```
Events with local changes: 245
Changes by field:
  type: 45 events
  subType: 23 events
  desc: 142 events
  seizureTimes: 35 events
```

**Action**: Run detection to preserve all 245 events with their various edits.

### Scenario 4: Some Events Added Locally

You've added some events that don't exist in the source JSON:

**Dry run output:**
```
Total events in database: 1250
Events in JSON files: 1200
Events NOT in JSON (local only): 50

Events with local changes: 45
```

**What this means:**
- 50 events are unique to your database (you created them)
- 45 of the remaining 1200 events have been edited
- Run detection to mark the 45 edited ones
- The 50 local-only events remain unchanged

---

## Troubleshooting

### Q: No JSON files found

**Error:**
```
No JSON files found in /home/graham/osd/osdb
```

**Cause**: Directory doesn't contain `osdb_*.json` files

**Fix**: 
- Publish the database first: `makeOsdDb_refactored_wrapper.py --osdb-dir /path --publish`
- Or specify the correct directory where JSON files are located

### Q: Shows "0 events with local changes"

**Meaning**: Your database events exactly match the JSON files - no local edits detected.

**Possible causes:**
- You haven't made any edits yet (normal for fresh database)
- Changes were made but then undone
- Database was recently synced from fresh JSON

**Action**: No detection needed, just proceed with normal workflow.

### Q: Shows many local-only events but few with changes

**Example output:**
```
Events NOT in JSON (local only): 300
Events with local changes: 5
```

**Meaning**: You've added 300 local events, only 5 of the existing JSON events were edited.

**Action**: Detection marks the 5 edited ones. The 300 local events are preserved as-is (never overwritten by updates).

### Q: Detects changes I didn't make

**Cause**: 
- Datapoint counts may differ (not tracked, but shown in database)
- Field values present in DB but missing in JSON
- JSON files are older versions

**Action**: 
- Review detected changes with `--dry-run` first
- If changes are legitimate edits you made, run detection
- If not, check your JSON files are current

### Q: Database is locked error

**Error:**
```
sqlite3.OperationalError: database is locked
```

**Cause**: Another process has the database open

**Fix**:
- Close all other connections (event_editor, other scripts)
- Wait a few seconds
- Retry the command

---

## After Running Detection

Once detection completes successfully:

✅ Your existing edits are **marked** with local change tracking  
✅ The `local_edits` column shows which fields you edited  
✅ The `has_local_changes` flag is set for edited events  
✅ Future updates will **preserve** these edits  

You can verify:
```bash
python3 manage_events.py show --db /home/graham/osd/osdb/osdb_working.db --event-id 12345
```

Look for: `local_edits: ["type", "desc"]` indicating which fields were tracked.

---

## Next Steps

After running detection:

1. ✅ Schema migration complete
2. ✅ Existing changes detected and marked
3. Continue with normal workflow:
   - Edit events through `event_editor.py`
   - Update database with `makeOsdDb_refactored_wrapper.py`
   - Your edits (old and new) are preserved!

---

## Technical Details

### How Detection Works

1. **Load JSON**: Reads all source JSON files and builds event dictionary
2. **Load Database**: Reads current events from database
3. **Compare**: For each event in database that exists in JSON:
   - Compare `type` field
   - Compare `subType` field
   - Compare `desc` field
   - Compare `seizureTimes` field
4. **Identify Differences**: Records which fields differ
5. **Mark Events**: Updates `local_edits` column with list of edited fields

### What Gets Compared

Only these **locally-editable** fields:
- `type` - Event type
- `subType` - Event subtype
- `desc` - Description
- `seizureTimes` - Seizure time markers

These fields are **NOT compared** (remote-sourced):
- `datapoints` - Updated by future imports
- `userId`, `dataTime`, `dataTimeEnd` - Metadata
- `alarmPhrase`, `alarmRationale` - Device info
- All other remote-sourced fields

### Performance

- Depends on database size
- ~1000 events: < 1 second
- ~10000 events: < 5 seconds
- ~100000 events: < 30 seconds

Mainly limited by JSON file I/O.

---

## Rollback (If Needed)

If you need to undo the detection:

```bash
# Reset all local_edits columns to NULL
sqlite3 /home/graham/osd/osdb/osdb_working.db << EOF
UPDATE events SET local_edits = NULL, has_local_changes = 0;
EOF
```

Then re-run detection if needed.

---

## Questions?

See the main documentation:
- [LOCAL_CHANGES_PRESERVATION_README.md](LOCAL_CHANGES_PRESERVATION_README.md) - Quick reference
- [LOCAL_CHANGES_PRESERVATION_SUMMARY.md](LOCAL_CHANGES_PRESERVATION_SUMMARY.md) - Complete overview
- [docs/LOCAL_CHANGES_PRESERVATION.md](docs/LOCAL_CHANGES_PRESERVATION.md) - Detailed guide

