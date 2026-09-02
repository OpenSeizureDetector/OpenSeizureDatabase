#!/usr/bin/env python3
"""
Create detailed merge analysis spreadsheet for manual review.
Shows all events that were merged, with details about the merge targets.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from dateutil import parser as dateutil_parser

# Paths
ORIGINAL_FILE = Path("/home/graham/osd/osdb_test_original/osdb_3min_allSeizures.json")
REFACTORED_FILE = Path("/home/graham/osd/osdb_test_refactored/osdb_3min_allSeizures.json")
# Output to validation/comparison_results/ subdirectory
OUTPUT_DIR = Path(__file__).parent / "comparison_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_datatime(dt_str):
    """Parse datetime string, handling multiple formats. Returns timezone-naive datetime."""
    try:
        dt = dateutil_parser.parse(dt_str, dayfirst=True)
        # Convert to timezone-naive if it has timezone info
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        return None


def load_events(filepath):
    """Load events from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def generate_merge_spreadsheet():
    """Generate CSV spreadsheet with merge analysis."""
    
    print("=" * 80)
    print("Generating Merge Analysis Spreadsheet")
    print("=" * 80)
    
    # Load events
    print(f"\nLoading events...")
    original_events = load_events(ORIGINAL_FILE)
    refactored_events = load_events(REFACTORED_FILE)
    
    # Create maps
    original_map = {int(e['id']): e for e in original_events}
    refactored_map = {int(e['id']): e for e in refactored_events}
    
    # Build reverse map: merged_id -> primary_id
    merged_into_map = {}
    for event in refactored_events:
        primary_id = int(event['id'])
        merged_from = event.get('_merged_from_event_ids', [])
        # Normalize to list (handle legacy formats)
        if merged_from is None:
            merged_from = []
        elif not isinstance(merged_from, list):
            merged_from = [merged_from]
        for merged_id in merged_from:
            merged_into_map[int(merged_id)] = primary_id
    
    # Find all events that exist in original but not as primary in refactored
    original_ids = set(original_map.keys())
    refactored_primary_ids = set(refactored_map.keys())
    
    merged_or_removed_ids = original_ids - refactored_primary_ids
    
    print(f"  Original events: {len(original_events)}")
    print(f"  Refactored events: {len(refactored_events)}")
    print(f"  Events merged or removed: {len(merged_or_removed_ids)}")
    
    # Create CSV rows
    rows = []
    
    for removed_id in sorted(merged_or_removed_ids):
        removed_event = original_map[removed_id]
        
        # Get basic info about removed event
        removed_user = removed_event.get('userId', 'N/A')
        removed_time = removed_event.get('dataTime', 'N/A')
        removed_type = f"{removed_event.get('type', 'N/A')}/{removed_event.get('subType', 'N/A')}"
        removed_datasource = removed_event.get('dataSourceName', '')
        removed_datapoints = len(removed_event.get('datapoints', []))
        
        # Check if it was merged
        if removed_id in merged_into_map:
            merged_into_id = merged_into_map[removed_id]
            merged_into_event = refactored_map[merged_into_id]
            
            # Get info about merge target
            merged_into_user = merged_into_event.get('userId', 'N/A')
            merged_into_time = merged_into_event.get('dataTime', 'N/A')
            merged_into_type = f"{merged_into_event.get('type', 'N/A')}/{merged_into_event.get('subType', 'N/A')}"
            merged_into_datasource = merged_into_event.get('dataSourceName', '')
            merged_into_datapoints = len(merged_into_event.get('datapoints', []))
            
            # Count how many events were merged into this one
            merge_count = merged_into_event.get('_merged_event_count', 1)
            total_datapoints = merged_into_event.get('_merged_datapoint_count', merged_into_datapoints)
            
            # Calculate time difference
            removed_dt = parse_datatime(removed_time)
            merged_dt = parse_datatime(merged_into_time)
            
            if removed_dt and merged_dt:
                time_diff = abs((removed_dt - merged_dt).total_seconds() / 60.0)
                time_diff_str = f"{time_diff:.2f}"
                within_threshold = "YES" if time_diff <= 3.0 else "NO"
            else:
                time_diff_str = "N/A"
                within_threshold = "N/A"
            
            # Was the merge target also from original?
            merged_target_was_original = "YES" if merged_into_id in original_ids else "NO"
            
            row = {
                'REMOVED_ID': removed_id,
                'REMOVED_USER': removed_user,
                'REMOVED_TIME': removed_time,
                'REMOVED_TYPE': removed_type,
                'REMOVED_DATASOURCE': removed_datasource,
                'REMOVED_DATAPOINTS': removed_datapoints,
                'TIME_DIFF_minutes': time_diff_str,
                'WITHIN_3MIN': within_threshold,
                'MERGED_INTO_ID': merged_into_id,
                'MERGED_INTO_USER': merged_into_user,
                'MERGED_INTO_TIME': merged_into_time,
                'MERGED_INTO_TYPE': merged_into_type,
                'MERGED_INTO_DATASOURCE': merged_into_datasource,
                'MERGED_INTO_DATAPOINTS': merged_into_datapoints,
                'MERGE_COUNT': merge_count,
                'TOTAL_DATAPOINTS_AFTER': total_datapoints,
                'TARGET_WAS_ORIGINAL': merged_target_was_original,
                'STATUS': 'MERGED'
            }
        else:
            # Not merged - truly removed
            row = {
                'REMOVED_ID': removed_id,
                'REMOVED_USER': removed_user,
                'REMOVED_TIME': removed_time,
                'REMOVED_TYPE': removed_type,
                'REMOVED_DATASOURCE': removed_datasource,
                'REMOVED_DATAPOINTS': removed_datapoints,
                'TIME_DIFF_minutes': 'N/A',
                'WITHIN_3MIN': 'N/A',
                'MERGED_INTO_ID': 'REMOVED',
                'MERGED_INTO_USER': '',
                'MERGED_INTO_TIME': '',
                'MERGED_INTO_TYPE': '',
                'MERGED_INTO_DATASOURCE': '',
                'MERGED_INTO_DATAPOINTS': '',
                'MERGE_COUNT': '',
                'TOTAL_DATAPOINTS_AFTER': '',
                'TARGET_WAS_ORIGINAL': '',
                'STATUS': 'REMOVED'
            }
        
        rows.append(row)
    
    # Write CSV
    output_file = OUTPUT_DIR / "merge_analysis.csv"
    
    fieldnames = [
        'REMOVED_ID', 'REMOVED_USER', 'REMOVED_TIME', 'REMOVED_TYPE', 'REMOVED_DATASOURCE', 'REMOVED_DATAPOINTS',
        'TIME_DIFF_minutes', 'WITHIN_3MIN',
        'MERGED_INTO_ID', 'MERGED_INTO_USER', 'MERGED_INTO_TIME', 'MERGED_INTO_TYPE', 'MERGED_INTO_DATASOURCE',
        'MERGED_INTO_DATAPOINTS', 'MERGE_COUNT', 'TOTAL_DATAPOINTS_AFTER', 'TARGET_WAS_ORIGINAL',
        'STATUS'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    # Print summary
    merged_count = sum(1 for r in rows if r['STATUS'] == 'MERGED')
    removed_count = sum(1 for r in rows if r['STATUS'] == 'REMOVED')
    within_threshold = sum(1 for r in rows if r['WITHIN_3MIN'] == 'YES')
    beyond_threshold = sum(1 for r in rows if r['WITHIN_3MIN'] == 'NO')
    
    print(f"\n{'=' * 80}")
    print("Merge Analysis Summary")
    print(f"{'=' * 80}")
    print(f"Total events analyzed: {len(rows)}")
    print(f"  Merged into other events: {merged_count}")
    print(f"  Truly removed: {removed_count}")
    print(f"\nMerge Time Analysis:")
    print(f"  Within 3-minute threshold: {within_threshold}")
    print(f"  Beyond 3-minute threshold: {beyond_threshold}")
    
    print(f"\n✓ Merge analysis saved to: {output_file}")
    print(f"  {len(rows)} rows")
    print(f"  Open in spreadsheet software for manual review")
    print(f"{'=' * 80}\n")
    
    # Create README
    readme_file = OUTPUT_DIR / "MERGE_ANALYSIS_README.md"
    with open(readme_file, 'w') as f:
        f.write("# Merge Analysis Spreadsheet\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total events**: {len(rows)}\n")
        f.write(f"- **Merged**: {merged_count}\n")
        f.write(f"- **Removed**: {removed_count}\n")
        f.write(f"- **Within 3-min threshold**: {within_threshold}\n")
        f.write(f"- **Beyond 3-min threshold**: {beyond_threshold}\n\n")
        f.write("## File: merge_analysis.csv\n\n")
        f.write("This spreadsheet shows all events from the original database that are no longer\n")
        f.write("primary events in the refactored version.\n\n")
        f.write("### Columns\n\n")
        f.write("**Removed Event (Original):**\n")
        f.write("- `REMOVED_ID`: Event ID that's no longer primary\n")
        f.write("- `REMOVED_USER`: User ID\n")
        f.write("- `REMOVED_TIME`: Event timestamp\n")
        f.write("- `REMOVED_TYPE`: Event type/subtype\n")
        f.write("- `REMOVED_DATASOURCE`: Data source name\n")
        f.write("- `REMOVED_DATAPOINTS`: Number of datapoints\n\n")
        f.write("**Merge Information:**\n")
        f.write("- `TIME_DIFF_minutes`: Time difference between events (minutes)\n")
        f.write("- `WITHIN_3MIN`: YES/NO - is time difference within 3-minute threshold?\n\n")
        f.write("**Merge Target (Refactored):**\n")
        f.write("- `MERGED_INTO_ID`: ID of the primary event this was merged into\n")
        f.write("- `MERGED_INTO_*`: Details about the merge target event\n")
        f.write("- `MERGE_COUNT`: How many events were merged into the target\n")
        f.write("- `TOTAL_DATAPOINTS_AFTER`: Total datapoints after merge\n")
        f.write("- `TARGET_WAS_ORIGINAL`: Was the merge target also from original database?\n\n")
        f.write("**Status:**\n")
        f.write("- `STATUS`: MERGED (merged into another event) or REMOVED (truly removed)\n\n")
        f.write("### Notes\n\n")
        f.write("- **All merged events are preserved** in the `_merged_from_event_ids` field\n")
        f.write("- Events merged beyond 3 minutes are due to sliding window \"chaining\"\n")
        f.write("- When `TARGET_WAS_ORIGINAL=YES`, both events were from the published database\n")
        f.write("- Sort by `TIME_DIFF_minutes` to see events merged at different distances\n")
        f.write("- Filter by `WITHIN_3MIN=NO` to review merges beyond threshold\n")
    
    print(f"✓ README saved to: {readme_file}\n")


if __name__ == '__main__':
    generate_merge_spreadsheet()
