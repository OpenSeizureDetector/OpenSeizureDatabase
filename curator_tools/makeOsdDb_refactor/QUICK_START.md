# Local Changes Preservation - Quick Reference Card

## Problem
Your local edits made in `event_editor.py` would be **overwritten** when running `makeOsdDb_refactored_wrapper.py`.

## Solution
Intelligent merge system that **preserves** your edits while updating remote data.

---

## Setup (One-Time)

### 1. Schema Migration
```bash
python3 src/schema_migration_v2.py --db /home/graham/osd/osdb/osdb_working.db
```

### 2. Detect Existing Edits (If Applicable)
```bash
# Dry run first
python3 src/detect_local_changes.py --db /home/graham/osd/osdb/osdb_working.db \
    --json-dir /home/graham/osd/osdb --dry-run

# Then apply
python3 src/detect_local_changes.py --db /home/graham/osd/osdb/osdb_working.db \
    --json-dir /home/graham/osd/osdb
```

---

## Normal Workflow (After Setup)

### Edit Events
```bash
python3 event_editor.py --db /home/graham/osd/osdb/osdb_working.db
```

### Update Database
```bash
python3 makeOsdDb_refactored_wrapper.py --osdb-dir /home/graham/osd/osdb
```

✅ Your edits are **automatically preserved**!

---

## What Gets Preserved

| Field | Preserved? |
|-------|-----------|
| Event Type | ✅ Yes |
| Event Subtype | ✅ Yes |
| Description | ✅ Yes |
| Seizure Times | ✅ Yes |
| Datapoints | ✅ Always Updated |
| Device Metadata | ✅ Always Updated |

---

## Output to Expect

### Normal Update
```
Merging 150 events into database (preserving local changes)...
✓ Merge complete: +42 new events, ~23 updated (preserved 18 local edits)
```

### Update with Conflicts
```
⚠ Remote Data Changes Detected:
  Event ID 12345:
    Local edits: type, desc
    Status: Local changes PRESERVED, remote datapoints UPDATED
    Action: Review the event in editor if needed
```

✅ Your edits are **still preserved** (not overwritten)!

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Changes disappeared | Run detection script to mark existing edits |
| Conflicts reported | Just review events to ensure edits still make sense |
| Database locked | Close event_editor and other scripts |
| No events detected | Your database exactly matches JSON (normal) |

---

## Files & Documentation

**Quick References:**
- [LOCAL_CHANGES_PRESERVATION_README.md](LOCAL_CHANGES_PRESERVATION_README.md) - Overview
- [LOCAL_CHANGES_PRESERVATION_SUMMARY.md](LOCAL_CHANGES_PRESERVATION_SUMMARY.md) - Complete guide

**Detailed Guides:**
- [docs/LOCAL_CHANGES_PRESERVATION.md](docs/LOCAL_CHANGES_PRESERVATION.md) - Full user guide
- [docs/DETECT_LOCAL_CHANGES.md](docs/DETECT_LOCAL_CHANGES.md) - Detection script guide
- [IMPLEMENTATION_LOCAL_CHANGES.md](IMPLEMENTATION_LOCAL_CHANGES.md) - Technical details

**Scripts:**
- `src/schema_migration_v2.py` - Add tracking columns (one-time)
- `src/detect_local_changes.py` - Find & mark existing edits

---

## Key Points

✅ **Automatic** - No workflow changes  
✅ **Smart** - Only preserves locally-edited fields  
✅ **Safe** - One-time migration, fully reversible  
✅ **Transparent** - Reports what was preserved  
✅ **Backward Compatible** - Works with existing databases  

---

## Implementation Summary

| Component | Status |
|-----------|--------|
| Schema migration | ✅ Ready |
| Change detection | ✅ Ready |
| Merge logic | ✅ Ready |
| Auto-tracking | ✅ Ready |
| Conflict detection | ✅ Ready |
| Documentation | ✅ Complete |

**Status**: ✅ **Ready to Deploy**

Just run the schema migration and you're done!

