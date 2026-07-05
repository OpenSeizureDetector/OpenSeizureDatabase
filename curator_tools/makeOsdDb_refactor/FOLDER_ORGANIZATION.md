# Folder Organization Summary

The makeOsdDb_refactor folder has been reorganized for clarity and maintainability.

## New Structure

```
makeOsdDb_refactor/
├── README.md                        # Main tool documentation
├── CLEAN_FILES_USAGE.md            # Clean utility documentation
├── makeOsdDb_refactored_wrapper.py # Main tool script
├── clean_existing_files.py         # Utility to clean existing files
├── publish_osdb.py                 # Publishing utility
├── client.cfg                      # Configuration (symlink)
│
├── src/                            # Core source code modules
│   ├── event_validation.py
│   ├── event_grouping.py
│   ├── event_deduplication.py
│   └── datetime_normalization.py
│
├── tests/                          # Unit tests
│   └── [test files]
│
├── docs/                           # Development documentation
│   ├── README.md                   # Documentation index
│   ├── PROJECT_SUMMARY.md
│   ├── EVENT_ID_PRESERVATION_FIX.md
│   ├── DATAPOINT_MERGE_EXPLANATION.md
│   └── [other docs]
│
├── validation/                     # Validation scripts & results
│   ├── README.md                   # Validation index
│   ├── generate_comparison_report.py
│   ├── test_event_preservation.py
│   ├── comparison_results/         # Generated reports
│   └── [other validation files]
│
└── archives/                       # Historical archives
    ├── README.md                   # Archive index
    ├── archived_tests/
    ├── previous_analyses/
    └── *.tgz files
```

## What Goes Where

### Top Level (Production Files)
**Purpose:** Files needed to run the tool in production

- `makeOsdDb_refactored_wrapper.py` - Main tool
- `clean_existing_files.py` - Utility tool
- `publish_osdb.py` - Publishing utility
- `README.md` - User documentation
- `CLEAN_FILES_USAGE.md` - Utility documentation

### src/
**Purpose:** Core implementation modules

Contains the modular implementation of all processing phases:
- Phase 1 & 2: Event validation and grouping
- Phase 3: Deduplication  
- Phase 4: Datetime normalization
- Supporting utilities

### tests/
**Purpose:** Unit and integration tests

Contains test files for validating individual modules and their integration.

### docs/
**Purpose:** Development and technical documentation

- Implementation details and rationale
- Bug fix explanations
- Test results and summaries
- Development progress logs
- Technical deep-dives

**Who needs this:** Developers, maintainers, or anyone wanting to understand implementation details.

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
