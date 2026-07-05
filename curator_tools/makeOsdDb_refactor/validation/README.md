# Validation Scripts and Results

This folder contains one-off validation scripts and analysis results used to verify the correctness of the refactored makeOsdDb implementation.

## Contents

### Validation Scripts

These scripts were used to validate the implementation and investigate issues:

- **test_event_preservation.py** - Unit test verifying 100% event preservation
- **validate_refactored_merging.py** - Validates merge logic
- **run_final_test.py** - Comprehensive test runner
- **compare_osdb_versions.py** - Compares different versions of OSDB
- **check_remote_server_status.py** - Checks API server status

### Analysis Scripts

These scripts generate detailed analysis and comparison reports:

- **generate_comparison_report.py** - Generates comparison between baseline and updated versions
- **generate_merge_analysis.py** - Creates detailed merge analysis spreadsheet
- **generate_enhanced_merge_analysis.py** - Enhanced merge statistics with datapoint analysis
- **generate_desc_field_report.py** - Reports on desc field updates for merged events
- **analyze_datapoint_times.py** - Analyzes datapoint time ranges
- **debug_merge_events_115_119.py** - Debug script for specific merge case

### Processing Scripts

- **download_and_process.py** - Downloads and processes events
- **demonstrate_consistency.py** - Demonstrates consistency checks

### Results

- **comparison_results/** - Generated comparison reports and CSV files
- **test_*.log** - Test execution logs
- **investigation_results.log** - Investigation findings

## Usage

These scripts are intended for **validation and development purposes only**. They:
- Reference test databases (osdb_test_original, osdb_test_refactored)
- Generate reports in the `comparison_results/` subdirectory
- Use absolute paths to test data locations

### Running Validation Scripts

Example:
```bash
cd validation/
python generate_comparison_report.py
python generate_enhanced_merge_analysis.py
```

### Important Notes

1. **Test Data Locations**: Scripts reference test databases at:
   - `/home/graham/osd/osdb_test_original/`
   - `/home/graham/osd/osdb_test_refactored/`

2. **Output Location**: Reports are generated in:
   - `validation/comparison_results/`

3. **Path Updates**: Scripts have been updated to work from the `validation/` subdirectory

## For Production Use

For production database updates, use the main tool in the parent directory:
- `makeOsdDb_refactored_wrapper.py` - Production database update tool

See the parent directory README.md for production usage documentation.
