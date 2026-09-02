# Archives

This folder contains archived test results, analyses, and historical data from the development and validation process.

## Contents

### Archived Test Runs

- **archived_tests/** - Complete test runs with all outputs
  - Contains timestamped test directories with full results
  - Each test run includes logs, comparison files, and analysis reports

### Previous Analyses

- **previous_analyses/** - Historical analysis runs
  - Earlier versions of comparison reports
  - Intermediate validation results

### Test Data

- **test_data/** - Test datasets used during validation
  - Sample events
  - Test scenarios
  - Edge cases

### Test Results

- **test_results/** - Historical test outputs
  - Various test configurations
  - Performance measurements
  - Validation runs

### Compressed Archives

- **analysis_results_*.tgz** - Compressed analysis result archives
- **final_test_results_*.tgz** - Compressed test result archives
- **previous_analyses_archive_*.tgz** - Compressed previous analysis archives
- **makeOsdDb_refactor_history.bundle** - Git history from nested repository (before integration into main repo)

### Index

- **ARCHIVE_INDEX.md** - Detailed index of archived materials

## Purpose

These archives serve as:
1. **Historical record** of development and testing process
2. **Reference** for understanding implementation decisions
3. **Backup** of validation data
4. **Audit trail** showing evolution of the refactored implementation

## Usage

These files are for **reference only** and are not needed for normal operation of the tool.

- To review historical test results, see the compressed archives
- To understand the development process, see archived_tests/ directories
- To examine specific test cases, see test_data/

## Cleanup

If disk space is a concern, these archives can be safely removed after:
- Verifying production deployment is successful
- Documenting key findings in the main documentation
- Ensuring no outstanding questions about the implementation

The core functionality is fully contained in the parent directory's `src/` folder.
