#!/usr/bin/env python3
"""
Generate comprehensive comparison report between original and refactored versions.
Compares against V1.10 baseline to show what each version added/removed/modified.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
BASELINE_DIR = Path("/home/graham/osd/osdb/V1.10")
ORIGINAL_DIR = Path("/home/graham/osd/osdb_test_original")
REFACTORED_DIR = Path("/home/graham/osd/osdb_test_refactored")
# Output to validation/comparison_results/ subdirectory
OUTPUT_DIR = Path(__file__).parent / "comparison_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_event_fingerprint(event):
    """Compute MD5 hash of event for comparison (excluding internal fields)."""
    exclude_fields = ['_merged_from_event_ids', '_merged_event_count', '_merged_datapoint_count', '_is_existing_event']
    
    # Create copy without excluded fields
    event_copy = {k: v for k, v in event.items() if k not in exclude_fields}
    
    # Convert to canonical JSON string
    json_str = json.dumps(event_copy, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def load_events(filepath):
    """Load events from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_event_sets(baseline_events, version1_events, version2_events, version1_name, version2_name):
    """Compare three sets of events and return statistics."""
    
    baseline_ids = {int(e['id']): e for e in baseline_events}
    v1_ids = {int(e['id']): e for e in version1_events}
    v2_ids = {int(e['id']): e for e in version2_events}
    
    # Track merged_from IDs in v2 (refactored)
    v2_merged_ids = set()
    for e in version2_events:
        merged_from = e.get('_merged_from_event_ids', [])
        v2_merged_ids.update(int(mid) for mid in merged_from)
    
    # All preserved IDs in v2 (primary + merged)
    v2_all_ids = set(v2_ids.keys()) | v2_merged_ids
    
    # Calculate differences from baseline
    v1_added = set(v1_ids.keys()) - set(baseline_ids.keys())
    v1_removed = set(baseline_ids.keys()) - set(v1_ids.keys())
    
    v2_added = set(v2_ids.keys()) - set(baseline_ids.keys())
    v2_removed = set(baseline_ids.keys()) - v2_all_ids  # Only truly removed if not in merged_from either
    
    # Events that were in baseline, are primary in v1, but only in merged_from in v2
    v2_merged_only = (set(baseline_ids.keys()) & set(v1_ids.keys())) - set(v2_ids.keys())
    v2_merged_only = v2_merged_only & v2_merged_ids  # Confirm they're actually merged
    
    # Modified events (same ID but different content)
    common_ids = set(baseline_ids.keys()) & set(v1_ids.keys()) & set(v2_ids.keys())
    v1_modified = set()
    v2_modified = set()
    
    for eid in common_ids:
        baseline_fp = compute_event_fingerprint(baseline_ids[eid])
        v1_fp = compute_event_fingerprint(v1_ids[eid])
        v2_fp = compute_event_fingerprint(v2_ids[eid])
        
        if v1_fp != baseline_fp:
            v1_modified.add(eid)
        if v2_fp != baseline_fp:
            v2_modified.add(eid)
    
    results = {
        'baseline_count': len(baseline_events),
        'v1_count': len(version1_events),
        'v2_count': len(version2_events),
        'v1_added': sorted(list(v1_added)),
        'v1_removed': sorted(list(v1_removed)),
        'v1_modified': sorted(list(v1_modified)),
        'v2_added': sorted(list(v2_added)),
        'v2_removed': sorted(list(v2_removed)),
        'v2_merged_only': sorted(list(v2_merged_only)),
        'v2_modified': sorted(list(v2_modified)),
    }
    
    return results


def generate_report():
    """Generate comprehensive comparison report."""
    
    print("=" * 80)
    print("makeOsdDb Comparison Report - Fixed Refactored Version")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("makeOsdDb Comparison Report - Fixed Refactored Version")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Compare each file type
    file_types = [
        ('osdb_3min_allSeizures.json', 'All Seizures'),
        ('osdb_3min_tcSeizures.json', 'Tonic-Clonic Seizures'),
        ('osdb_3min_fallEvents.json', 'Fall Events')
    ]
    
    for filename, file_desc in file_types:
        print(f"\n{'=' * 80}")
        print(f"{file_desc}: {filename}")
        print(f"{'=' * 80}\n")
        report_lines.append(f"\n{'=' * 80}")
        report_lines.append(f"{file_desc}: {filename}")
        report_lines.append(f"{'=' * 80}\n")
        
        # Load events
        baseline_file = BASELINE_DIR / filename
        original_file = ORIGINAL_DIR / filename
        refactored_file = REFACTORED_DIR / filename
        
        if not baseline_file.exists():
            msg = f"⚠️  Baseline file not found: {baseline_file}"
            print(msg)
            report_lines.append(msg)
            continue
        
        baseline_events = load_events(baseline_file)
        original_events = load_events(original_file)
        refactored_events = load_events(refactored_file)
        
        # Compare
        results = compare_event_sets(
            baseline_events, original_events, refactored_events,
            "Original", "Refactored (Fixed)"
        )
        
        # Print summary
        print("Event Counts:")
        print(f"  Baseline (V1.10):        {results['baseline_count']}")
        print(f"  Original (after update): {results['v1_count']}")
        print(f"  Refactored (after update): {results['v2_count']}")
        
        report_lines.append("Event Counts:")
        report_lines.append(f"  Baseline (V1.10):        {results['baseline_count']}")
        report_lines.append(f"  Original (after update): {results['v1_count']}")
        report_lines.append(f"  Refactored (after update): {results['v2_count']}")
        
        print("\nOriginal Version vs Baseline:")
        print(f"  Added: {len(results['v1_added'])} events")
        print(f"  Removed: {len(results['v1_removed'])} events")
        print(f"  Modified: {len(results['v1_modified'])} events")
        
        report_lines.append("\nOriginal Version vs Baseline:")
        report_lines.append(f"  Added: {len(results['v1_added'])} events")
        report_lines.append(f"  Removed: {len(results['v1_removed'])} events")
        report_lines.append(f"  Modified: {len(results['v1_modified'])} events")
        
        print("\nRefactored Version vs Baseline:")
        print(f"  Added: {len(results['v2_added'])} events")
        print(f"  Removed: {len(results['v2_removed'])} events")
        print(f"  Merged (not primary): {len(results['v2_merged_only'])} events")
        print(f"  Modified: {len(results['v2_modified'])} events")
        
        report_lines.append("\nRefactored Version vs Baseline:")
        report_lines.append(f"  Added: {len(results['v2_added'])} events")
        report_lines.append(f"  Removed: {len(results['v2_removed'])} events")
        report_lines.append(f"  Merged (not primary): {len(results['v2_merged_only'])} events")
        report_lines.append(f"  Modified: {len(results['v2_modified'])} events")
        
        # Check preservation
        if results['v2_removed']:
            print(f"\n⚠️  WARNING: {len(results['v2_removed'])} baseline events LOST in refactored version!")
            print(f"  Lost IDs: {results['v2_removed'][:20]}")
            report_lines.append(f"\n⚠️  WARNING: {len(results['v2_removed'])} baseline events LOST in refactored version!")
            report_lines.append(f"  Lost IDs: {results['v2_removed'][:20]}")
        else:
            print("\n✓ All baseline events preserved in refactored version")
            report_lines.append("\n✓ All baseline events preserved in refactored version")
        
        if results['v2_merged_only']:
            print(f"\nℹ️  {len(results['v2_merged_only'])} baseline events merged into other events:")
            print(f"  IDs: {results['v2_merged_only'][:20]}")
            report_lines.append(f"\nℹ️  {len(results['v2_merged_only'])} baseline events merged into other events:")
            report_lines.append(f"  IDs: {results['v2_merged_only'][:20]}")
        
        # Save detailed comparison for this file
        detail_file = OUTPUT_DIR / f"detail_{filename.replace('.json', '.txt')}"
        with open(detail_file, 'w') as f:
            f.write(f"Detailed Comparison: {file_desc}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Original - Added IDs ({len(results['v1_added'])}):\n")
            f.write(f"{results['v1_added']}\n\n")
            f.write(f"Original - Removed IDs ({len(results['v1_removed'])}):\n")
            f.write(f"{results['v1_removed']}\n\n")
            f.write(f"Refactored - Added IDs ({len(results['v2_added'])}):\n")
            f.write(f"{results['v2_added']}\n\n")
            f.write(f"Refactored - Removed IDs ({len(results['v2_removed'])}):\n")
            f.write(f"{results['v2_removed']}\n\n")
            f.write(f"Refactored - Merged Only IDs ({len(results['v2_merged_only'])}):\n")
            f.write(f"{results['v2_merged_only']}\n\n")
        
        print(f"\n  → Detailed comparison saved to: {detail_file}")
        report_lines.append(f"\n  → Detailed comparison saved to: {detail_file}")
    
    # Save summary report
    summary_file = OUTPUT_DIR / "COMPARISON_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n{'=' * 80}")
    print(f"✓ Reports generated in: {OUTPUT_DIR}")
    print(f"  - {summary_file.name}")
    print(f"  - detail_osdb_3min_*.txt files")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    generate_report()
