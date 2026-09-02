# Folder Organization Summary

The makeOsdDb_refactor folder has been reorganized for clarity and maintainability.

## New Structure

```
makeOsdDb_refactor/
├── README.md                        # Main tool documentation (entry point)
├── makeOsdDb_refactored_wrapper.py # Main tool script
├── manage_events.py                # Event management CLI
├── event_editor.py                 # Qt5 GUI event editor
├── launch_editor.sh                # Event editor launch script
├── clean_existing_files.py         # Utility to clean existing files
├── publish_osdb.py                 # Publishing utility
├── osdb.cfg                        # Configuration file
├── client.cfg                      # API credentials (symlink)
│
├── src/                            # Shared Python modules (used by all tools)
│   ├── event_validation.py         # Event validation
│   ├── event_grouping.py           # Sliding window grouping
│   ├── event_deduplication.py      # Datapoint deduplication
│   ├── datetime_normalization.py   # Datetime normalization
│   ├── osdb_sqlite.py              # SQLite database interface
│   ├── database_utils.py           # Database utilities
│   ├── init_database.py            # JSON to SQLite import
│   ├── event_downloader.py         # Event download from API
│   └── osdb_publication.py         # Database to JSON export
│
├── tests/                          # Unit and integration tests
│   ├── test_database_utils.py
│   ├── test_wrapper_integration.py
│   └── test_database.py
│
├── docs/                           # All documentation (consolidated)
│   ├── README.md                   # Documentation index
│   ├── README_SQLITE.md            # SQLite integration guide
│   ├── QUICKSTART_EVENT_MANAGEMENT.md  # Event management CLI guide
│   ├── SCHEMA_ANALYSIS.md          # JSON vs SQLite schema
│   ├── IMPLEMENTATION_STATUS.md    # Implementation progress
│   ├── SQLITE_INTEGRATION_COMPLETE.md  # Integration summary
│   ├── CLEAN_FILES_USAGE.md        # Clean utility guide
│   ├── FOLDER_ORGANIZATION.md      # This file
│   ├── event_editor/               # Event editor specific docs
│   │   ├── README.md               # GUI usage guide
│   │   └── INSTALL.md              # Installation instructions
│   ├── EVENT_ID_PRESERVATION_FIX.md
│   ├── DATAPOINT_MERGE_EXPLANATION.md
│   └── [other technical documentation]
│
├── validation/                     # Validation scripts & results
│   ├── README.md                   # Validation index
│   ├── test_event_preservation.py
│   └── comparison_results/
│
└── archives/                       # Historical archives
    ├── README.md                   # Archive index
    └── [archived test results]
```

## What Goes Where

### Top Level (Production Files)
**Purpose:** Executable scripts and main entry point documentation

- `README.md` - Main documentation (entry point for users)
- `makeOsdDb_refactored_wrapper.py` - Main database update and publish tool
- `manage_events.py` - Event management CLI tool
- `event_editor.py` - Qt5 GUI event editor (shares src/ with wrapper)
- `clean_existing_files.py` - Utility to clean existing files
- `publish_osdb.py` - Publishing utility
- `osdb.cfg` - Configuration file
- `client.cfg` - API credentials

### src/
**Purpose:** Shared Python modules used by all tools

Contains the modular implementation of all processing phases:
- Database interface (`osdb_sqlite.py`)
- Event processing (`event_validation.py`, `event_grouping.py`, `event_deduplication.py`)
- Datetime normalization (`datetime_normalization.py`)
- Database utilities (`database_utils.py`, `init_database.py`)
- Event download and publication (`event_downloader.py`, `osdb_publication.py`)

**Shared by:** Both `makeOsdDb_refactored_wrapper.py` and `event_editor.py` import from this directory.

### tests/
**Purpose:** Unit and integration tests

Contains test files for validating individual modules and their integration:
- Database utilities tests
- Wrapper integration tests
- Database operation tests

### docs/
**Purpose:** All documentation (user guides + technical documentation)

**User Guides:**
- `README_SQLITE.md` - SQLite integration guide
- `QUICKSTART_EVENT_MANAGEMENT.md` - Event management CLI guide
- `CLEAN_FILES_USAGE.md` - Clean utility guide
- `event_editor/README.md` - GUI usage guide
- `event_editor/INSTALL.md` - GUI installation instructions

**Technical Documentation:**
- `SCHEMA_ANALYSIS.md` - JSON vs SQLite schema comparison
- `IMPLEMENTATION_STATUS.md` - Implementation progress tracking
- `SQLITE_INTEGRATION_COMPLETE.md` - Integration summary
- `EVENT_ID_PRESERVATION_FIX.md` - Technical details on event ID preservation
- `DATAPOINT_MERGE_EXPLANATION.md` - Merge behavior explanation
- `FOLDER_ORGANIZATION.md` - This file
- Other development documentation

**Who needs this:** All users for guides, developers/maintainers for technical docs.

### validation/
**Purpose:** One-off validation scripts and analysis results

- Scripts used to validate the refactored implementation
- Comparison and analysis tools
- Test execution scripts
- Generated reports and spreadsheets
- Investigation logs

**Who needs this:** Anyone validating the implementation or investigating specific behaviors.

### archives/
**Purpose:** Historical data and test results

- Archived test runs
- Previous analysis results  
- Compressed archives
- Test data samples

**Who needs this:** For historical reference only; not needed for operation.

## Path Updates

All validation scripts have been updated to work from their new location:

1. **Relative imports:** `sys.path.insert(0, '../src')` (up one level to src/)
2. **Output paths:** `Path(__file__).parent / 'comparison_results'`
3. **Test data paths:** Still use absolute paths to `/home/graham/osd/osdb_test_*`

## Quick Start

### For Users
1. Read `README.md` in the top level
2. Use `makeOsdDb_refactored_wrapper.py` to update databases
3. Use `clean_existing_files.py` if needed (see `CLEAN_FILES_USAGE.md`)

### For Developers
1. Review `docs/PROJECT_SUMMARY.md` for overview
2. Check `docs/EVENT_ID_PRESERVATION_FIX.md` for implementation details
3. Look at `src/` for the actual code
4. Run tests from `tests/` directory

### For Validation
1. See `validation/README.md` for script descriptions
2. Run validation scripts from `validation/` directory
3. Check `validation/comparison_results/` for reports

## Benefits of This Organization

1. **Cleaner top level** - Only production files and essential documentation
2. **Clear separation** - Development docs separate from user docs
3. **Easy maintenance** - Related files grouped together
4. **Better navigation** - Each folder has a README explaining its contents
5. **Scalability** - Easy to add new validation scripts or documentation

## Migration Notes

- All files moved without modification (except path updates)
- Functionality preserved - scripts still work from new locations
- Original file relationships maintained
- Git history preserved (files moved, not deleted/recreated)

---

**Date Organized:** 2026-07-05
**Organized By:** File structure cleanup after validation completion
