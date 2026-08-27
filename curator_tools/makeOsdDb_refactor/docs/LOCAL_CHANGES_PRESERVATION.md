# Local Change Preservation Guide

## Overview

This guide explains how the OSDB database now preserves your local changes made through `event_editor.py` when you run database updates from the remote server.

**Problem Solved**: Previously, running `makeOsdDb_refactored_wrapper.py --osdb-dir /path` would **completely overwrite** any changes you made through the GUI editor. Now, local changes are **automatically preserved**.

---

## How It Works

### What Can Be Edited Locally

Through the event editor GUI, you can edit:
- **Event Type** - Seizure, Fall, False Alarm, etc.
- **Event Subtype** - Tonic-Clonic, Aura, Stumble, etc.
- **Description** - Custom notes about the event
- **Seizure Times** - Start and end times of the seizure period

### How Updates Work Now

When you run an update:
```bash
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb
```

The system now:

1. **Downloads** new events from the remote server
2. **Checks** each event against the database
3. **For new events**: Adds them normally
4. **For existing events with local edits**:
   - ✅ **Preserves** your local edits (type, subType, desc, seizureTimes)
   - ✅ **Updates** remote-sourced fields (datapoints, alarm info, hardware metadata)
   - ✅ **Detects conflicts** if remote data changed significantly
   - ✅ **Notifies** you of any conflicts to review

5. **For existing events without local edits**:
   - Updates all fields normally (standard behavior)

---

## Getting Started

### Step 1: Run Schema Migration (First Time Only)

Before using the new feature, you must add the tracking columns to your database:

```bash
python3 src/schema_migration_v2.py --db /home/graham/osd/osdb/osdb_working.db
```

**Expected Output**:
```
Connecting to database: /home/graham/osd/osdb/osdb_working.db
Current schema version: 1
Applying schema migration v2...
  - Adding 'local_edits' column...
  - Adding 'remote_hash' column...
  - Adding 'last_remote_update' column...
  - Adding 'has_local_changes' column...
  - Creating index on 'has_local_changes'...

✓ Schema migration v2 completed successfully
```

**Note**: This is a one-time operation. The database will automatically track changes going forward.

### Step 2: Edit Events Through GUI

Use the event editor normally:

```bash
python3 event_editor.py --db /home/graham/osd/osdb/osdb_working.db
```

Make any changes you want:
- Change event type/subtype
- Update description with notes
- Adjust seizure times (±5s buttons)
- Save changes

✅ Your changes are now **tracked** in the database

### Step 3: Update Database From Remote

Run the normal update command:

```bash
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb
```

The system will:
- Download new events from remote server
- **Preserve** your local edits
- **Update** remote-sourced data
- **Report** any conflicts

---

## Understanding the Output

### Normal Update (No Conflicts)

```
Merging 150 events into database (preserving local changes)...

✓ Merge complete: +42 new events, ~23 updated (preserved 18 local edits)
  Event type: tcSeizures
  Database: /home/graham/osd/osdb/osdb_working.db
```

**Interpretation**:
- `+42 new events` - 42 events downloaded for the first time
- `~23 updated` - 23 existing events had new remote data
- `preserved 18 local edits` - Your changes to 18 events were kept
- No conflicts detected ✓

### Update With Conflicts Warning

```
Merging 150 events into database (preserving local changes)...

⚠ Remote Data Changes Detected:
======================================================================

Event ID 12345:
  Remote data changed, but local edits exist:
  Local edits: type, desc
  Status: Local changes PRESERVED, remote datapoints UPDATED
  Action: Review the event in editor if needed

Event ID 67890:
  Remote data changed, but local edits exist:
  Local edits: seizureTimes
  Status: Local changes PRESERVED, remote datapoints UPDATED
  Action: Review the event in editor if needed

======================================================================

✓ Merge complete: +42 new events, ~23 updated (preserved 18 local edits), ⚠ 2 events with remote changes (review recommended)
  Event type: tcSeizures
  Database: /home/graham/osd/osdb/osdb_working.db
```

**Interpretation**:
- Remote data changed for 2 events
- Your local edits were **preserved**
- New remote datapoints were **imported**
- ⚠ Recommendation: Review these events in the GUI to ensure your edits still make sense with the new remote data

---

## Common Scenarios

### Scenario 1: Correct an Event Type, Then Update

**Your workflow**:
1. Open event_editor.py
2. Find event with wrong type (e.g., marked as "False Alarm" but is actually "Seizure")
3. Change type to "Seizure"
4. Save changes
5. Run update from remote server

**Result**: ✅ Your type correction is preserved, datapoints are updated

### Scenario 2: Add Custom Notes, Then Update

**Your workflow**:
1. Open event_editor.py
2. Find event with no description
3. Type custom notes: "Confirmed tonic-clonic seizure, lasted ~2 min"
4. Save changes
5. Run update from remote server

**Result**: ✅ Your description is preserved, other fields updated from remote

### Scenario 3: Adjust Seizure Times After Manual Review

**Your workflow**:
1. Open event_editor.py
2. View acceleration graph and seizure time markers
3. Adjust seizure start/end times using ±5s buttons
4. Save changes
5. Run update from remote server
6. Remote server has new accelerometer datapoints for this event

**Result**: ✅ Your adjusted seizure times are preserved, new datapoints imported

### Scenario 4: Remote Changes Conflict With Local Edits

**Situation**: You changed the event type, remote server sends updated data
```
⚠ Remote Data Changes Detected:
  Event ID 12345:
    Remote data changed, but local edits exist:
    Local edits: type, desc
    Status: Local changes PRESERVED, remote datapoints UPDATED
    Action: Review the event in editor if needed
```

**What to do**:
1. Open event_editor.py
2. Navigate to event 12345
3. Review if your type/description edits are still valid given the new remote data
4. If needed, make further adjustments
5. Save again

---

## Technical Details

### Which Fields Are Tracked?

**Locally-editable fields** (preserved during updates):
- `type` - Event classification
- `subType` - Event subclassification
- `desc` - Description/notes
- `seizureTimes` - Seizure start/end times

**Remote-only fields** (always updated from server):
- `userId`, `dataTime`, `dataTimeEnd` - Timing metadata
- `osdAlarmState` - Alarm state from device
- `dataSourceName`, `phoneAppVersion`, `watchSdVersion` - Source metadata
- `alarmPhrase`, `alarmRationale`, `alarmThresh*` - Alarm settings
- `hrThreshMin/Max`, `o2SatThresh*` - Device thresholds
- `datapoints` - All accelerometer, HR, and O2 saturation data

### Database Columns Added

Schema migration v2 adds these columns:
- `local_edits` (TEXT) - JSON array of field names edited locally
- `remote_hash` (TEXT) - MD5 hash of remote-only fields (for change detection)
- `last_remote_update` (TEXT) - Timestamp of last remote fetch
- `has_local_changes` (INTEGER) - Boolean flag: 1 if locally edited, 0 if not

### Conflict Detection

A "conflict" is reported when:
1. Event exists in local database
2. Event has local edits (preserved)
3. Remote data for that event has changed (different hash)

This is informational only - conflicts do NOT override local changes. You're just notified that the remote data changed while you had local edits.

---

## Troubleshooting

### Q: My changes disappeared after an update!

**Possible causes**:
1. ❌ **Database not migrated yet** - You skipped Step 1
   
   **Fix**: Run `schema_migration_v2.py`
   
2. ❌ **Used old `add_events()` method** - Old code still in use
   
   **Fix**: Ensure you're using the updated wrapper with `add_events_preserve_local()`

3. ❌ **Changes made after migration started** - Timing issue
   
   **Fix**: Make changes through event_editor.py AFTER schema migration completes

### Q: I got a conflict warning - what should I do?

**Answer**: 
1. Don't panic - your local edits were preserved
2. Open event_editor.py
3. Navigate to the conflicting event ID
4. Review if your edits still make sense with the new remote data
5. Make adjustments if needed
6. Save again

Conflicts are informational - they tell you that both you AND the remote server changed the same event. Worth reviewing but not an error.

### Q: How do I revert a local edit?

**Answer**: 
1. Open event_editor.py
2. Navigate to the event
3. Click "Revert Changes" button (clears unsaved changes)
4. OR manually restore original values and save

Note: This only reverts unsaved changes. To undo saved changes, you'd need to manually edit the fields again.

### Q: Can I disable local change tracking?

**Answer**: Not recommended, but you can:
1. Go back to using the old `db.add_events()` method
2. Accept that all local changes will be overwritten
3. OR keep manual backups of the database before updates

**Better option**: Just use the new system - it's designed to preserve your work.

---

## Best Practices

1. **Always run schema migration first** - Do this once before any updates
   ```bash
   python3 src/schema_migration_v2.py --db /home/graham/osd/osdb/osdb_working.db
   ```

2. **Make edits BEFORE updating** - This ensures they're tracked
   - Edit events in event_editor.py
   - Save changes
   - THEN run `makeOsdDb_refactored_wrapper.py --osdb-dir /path`

3. **Review conflict warnings** - When you get ⚠ notifications, take a quick look to make sure your edits still make sense

4. **Back up before major updates** - Optional but recommended
   ```bash
   cp /home/graham/osd/osdb/osdb_working.db /home/graham/osd/osdb/osdb_working.db.backup.$(date +%s)
   ```

5. **Document why you edited each event** - Use the description field to explain local changes
   - "Fixed incorrect type: was 'False Alarm', should be 'Tonic-Clonic'"
   - "Adjusted seizure times based on manual graph review: ±2 seconds"

---

## Additional Resources

- [Event Editor GUI Guide](docs/event_editor/README.md) - Detailed GUI usage
- [Event Management CLI Guide](docs/QUICKSTART_EVENT_MANAGEMENT.md) - Command-line editing
- [Database Structure](README.md#key-features) - Technical database details

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the conflict report output carefully
3. Inspect the database using `manage_events.py` CLI:
   ```bash
   python3 manage_events.py show --db /home/graham/osd/osdb/osdb_working.db --event-id <ID>
   ```

