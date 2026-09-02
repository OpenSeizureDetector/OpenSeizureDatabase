#!/usr/bin/env python3
"""
Final comparison test: Run both original and refactored makeOsdDb
from the same baseline (/home/graham/osd/osdb) and compare results.
"""

import sys
import os
import json
import subprocess
import shutil
from datetime import datetime

# Add parent directory to path (validation/ is one level deep, so ../../.. for libosd)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a shell command and report results"""
    print(f"\n>>> {description}")
    print(f"Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Command failed with code {result.returncode}")
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def count_events(json_file):
    """Count events in a JSON file"""
    try:
        with open(json_file, 'r') as f:
            events = json.load(f)
        return len(events)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return 0

def main():
    print_section("FINAL TEST: Original vs Refactored makeOsdDb")
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTest Setup:")
    print("  - Both versions start from: /home/graham/osd/osdb (407 allSeizures)")
    print("  - Original will update: /home/graham/osd/osdb_test_original")
    print("  - Refactored will update: /home/graham/osd/osdb_test_refactored")
    print("  - Comparison baseline: /home/graham/osd/osdb/V1.10 (257 allSeizures)")
    print("  - Data source filtering: ENABLED (excluding Phone, AndroidWear)")
    
    # Create config file for original pointing to test directory
    print_section("Step 1: Preparing Configurations")
    
    original_config = {
        "osdbDir": "/home/graham/osd/osdb_test_original",
        "groupingPeriod": "3min",
        "includeWarnings": 1,
        "excludeDataSources": ["Phone", "AndroidWear"],
        "includeDataSources": [],
        "dataFiles": [
            "osdb_3min_allSeizures.json",
            "osdb_3min_tcSeizures.json",
            "osdb_3min_fallEvents.json",
            "osdb_3min_falseAlarms.json"
        ],
        "credentialsFname": "../client.cfg",
        "seizureTimesFname": "seizure_durations.csv",
        "skipElements": ["accMean", "accSd", "updated", "created", "dataTimeStr", 
                        "maxVal", "maxFreq", "statusStr", "categoryId"],
        "invalidEvents": []  # Will be loaded from main config
    }
    
    # Load invalid events from main config
    import libosd.configUtils
    main_config = libosd.configUtils.loadConfig('../osdb.cfg')
    original_config['invalidEvents'] = main_config.get('invalidEvents', [])
    
    # Save test config for original
    with open('osdb_test_original.cfg', 'w') as f:
        json.dump(original_config, f, indent=4)
    print("✓ Created osdb_test_original.cfg")
    
    # Save test config for refactored
    refactored_config = original_config.copy()
    refactored_config['osdbDir'] = "/home/graham/osd/osdb_test_refactored"
    with open('osdb_test_refactored.cfg', 'w') as f:
        json.dump(refactored_config, f, indent=4)
    print("✓ Created osdb_test_refactored.cfg")
    
    # Record baseline counts
    print_section("Step 2: Recording Baseline Counts")
    baseline_counts = {}
    for event_type in ['allSeizures', 'fallEvents']:
        v110_file = f"/home/graham/osd/osdb/V1.10/osdb_3min_{event_type}.json"
        test_orig_file = f"/home/graham/osd/osdb_test_original/osdb_3min_{event_type}.json"
        
        v110_count = count_events(v110_file)
        test_count = count_events(test_orig_file)
        
        baseline_counts[event_type] = {
            'v110': v110_count,
            'test_baseline': test_count
        }
        print(f"{event_type}:")
        print(f"  V1.10 baseline: {v110_count}")
        print(f"  Test baseline:  {test_count}")
    
    # Run original makeOsdDb
    print_section("Step 3: Running Original makeOsdDb")
    os.chdir('/home/graham/osd/OpenSeizureDatabase/curator_tools')
    
    original_cmd = (
        "/home/graham/osd/OpenSeizureDatabase/venv/bin/python "
        "makeOsdDb.py -c makeOsdDb_refactor/osdb_test_original.cfg 2>&1 | "
        "tee /tmp/original_final_test.log"
    )
    
    if not run_command(original_cmd, "Running original makeOsdDb.py"):
        print("ERROR: Original makeOsdDb failed!")
        return 1
    
    # Run refactored makeOsdDb
    print_section("Step 4: Running Refactored makeOsdDb")
    os.chdir('/home/graham/osd/OpenSeizureDatabase/curator_tools')
    
    refactored_cmd = (
        "/home/graham/osd/OpenSeizureDatabase/venv/bin/python "
        "makeOsdDb_refactor/makeOsdDb_refactored_wrapper.py "
        "-c makeOsdDb_refactor/osdb_test_refactored.cfg 2>&1 | "
        "tee /tmp/refactored_final_test.log"
    )
    
    if not run_command(refactored_cmd, "Running refactored wrapper"):
        print("ERROR: Refactored makeOsdDb failed!")
        return 1
    
    # Compare results
    print_section("Step 5: Comparing Results")
    
    results = {}
    for event_type in ['allSeizures', 'fallEvents']:
        v110_file = f"/home/graham/osd/osdb/V1.10/osdb_3min_{event_type}.json"
        orig_file = f"/home/graham/osd/osdb_test_original/osdb_3min_{event_type}.json"
        ref_file = f"/home/graham/osd/osdb_test_refactored/osdb_3min_{event_type}.json"
        
        v110_count = count_events(v110_file)
        orig_count = count_events(orig_file)
        ref_count = count_events(ref_file)
        
        results[event_type] = {
            'v110': v110_count,
            'original': orig_count,
            'refactored': ref_count,
            'original_vs_v110': orig_count - v110_count,
            'refactored_vs_v110': ref_count - v110_count,
            'original_vs_refactored': orig_count - ref_count
        }
        
        print(f"\n{event_type}:")
        print(f"  V1.10:       {v110_count:4d} events")
        print(f"  Original:    {orig_count:4d} events ({orig_count - v110_count:+4d} vs V1.10)")
        print(f"  Refactored:  {ref_count:4d} events ({ref_count - v110_count:+4d} vs V1.10)")
        print(f"  Difference:  {orig_count - ref_count:4d} events (Original - Refactored)")
    
    # Run comparison analysis
    print_section("Step 6: Running Detailed Comparison Analysis")
    os.chdir('/home/graham/osd/OpenSeizureDatabase/curator_tools/makeOsdDb_refactor')
    
    comparison_cmd = (
        "/home/graham/osd/OpenSeizureDatabase/venv/bin/python "
        "compare_osdb_versions.py 2>&1 | tee /tmp/comparison_final.log"
    )
    
    # Update compare script to use test directories
    print("Updating comparison script to use test directories...")
    
    run_command(comparison_cmd, "Running comparison analysis")
    
    print_section("Test Complete!")
    print(f"Test Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults Summary:")
    print("  Original makeOsdDb:")
    for event_type, counts in results.items():
        print(f"    {event_type}: {counts['original']} events")
    print("\n  Refactored makeOsdDb:")
    for event_type, counts in results.items():
        print(f"    {event_type}: {counts['refactored']} events")
    
    print("\nLogs saved to:")
    print("  - /tmp/original_final_test.log")
    print("  - /tmp/refactored_final_test.log")
    print("  - /tmp/comparison_final.log")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
