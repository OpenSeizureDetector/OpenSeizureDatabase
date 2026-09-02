#!/usr/bin/env python3
"""
clean_existing_files.py

Utility to process existing OSDB JSON files to:
- Remove duplicates (same event downloaded multiple times)
- Merge events within time threshold (e.g., 3min) with datapoint concatenation
- Generate cleaned output files

This applies Phase 2 processing to existing database files.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from event_grouping import apply_sliding_window_grouping
from event_deduplication import remove_duplicate_events
from datetime_normalization import normalize_events_batch, detect_datetime_formats


def load_json_file(filepath: str) -> list:
    """Load events from a JSON file.
    
    Handles two formats:
    - Plain JSON array: [event1, event2, ...]
    - Wrapped format: {"events": [event1, event2, ...], "description": ...}
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle wrapped format (test data files)
    if isinstance(data, dict) and 'events' in data:
        return data['events']
    
    # Handle plain array format (standard OSDB files)
    if isinstance(data, list):
        return data
    
    # Unknown format
    raise ValueError(f"Unknown JSON format in {filepath}. Expected list or dict with 'events' key.")


def save_json_file(events: list, filepath: str, indent: int = 2):
    """Save events to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(events, f, indent=indent)


def clean_events(events: list, 
                 time_threshold: str = '3min',
                 concatenate_datapoints: bool = True,
                 remove_duplicates: bool = True,
                 duplicate_method: str = 'hash',
                 keep_strategy: str = 'most_datapoints',
                 normalize_datetimes: bool = True,
                 show_progress: bool = True) -> tuple:
    """
    Clean a list of events by removing duplicates and merging close events.
    
    Args:
        events: List of event dictionaries
        time_threshold: Time threshold for grouping (e.g., '3min', '180s')
        concatenate_datapoints: Whether to concatenate datapoints from grouped events
        remove_duplicates: Whether to remove duplicate events first
        duplicate_method: Deduplication method ('hash' or 'id')
        keep_strategy: Which duplicate to keep ('first', 'last', 'most_datapoints')
        normalize_datetimes: Whether to normalize datetime formats to ISO 8601
        show_progress: Show progress bars
        
    Returns:
        Tuple of (cleaned_events, statistics_dict)
    """
    stats = {
        'input_events': len(events),
        'after_dedup': 0,
        'after_normalization': 0,
        'after_grouping': 0,
        'duplicates_removed': 0,
        'groups_merged': 0,
        'datapoints_before': 0,
        'datapoints_after': 0,
        'datetime_formats_before': {},
        'datetime_formats_after': {}
    }
    
    # Step 0: Detect datetime formats (before normalization)
    if normalize_datetimes and events:
        stats['datetime_formats_before'] = detect_datetime_formats(events)
    
    # Step 1: Remove duplicates if requested
    if remove_duplicates and events:
        print(f"\n=== Step 1: Deduplication (method: {duplicate_method}) ===")
        deduplicated, dedup_info = remove_duplicate_events(
            events,
            method=duplicate_method,
            keep=keep_strategy
        )
        stats['after_dedup'] = len(deduplicated)
        stats['duplicates_removed'] = dedup_info['duplicates_removed']
        print(f"  Input: {stats['input_events']} events")
        print(f"  Found: {dedup_info['duplicates_found']} duplicates")
        print(f"  Removed: {stats['duplicates_removed']} events")
        print(f"  Output: {stats['after_dedup']} unique events")
        events = deduplicated
    else:
        stats['after_dedup'] = len(events)
    
    # Step 2: Normalize datetime formats
    if normalize_datetimes and events:
        print(f"\n=== Step 2: Datetime Normalization (ISO 8601) ===")
        normalized, norm_stats = normalize_events_batch(
            events,
            normalize_datapoints=True,
            show_progress=show_progress
        )
        stats['after_normalization'] = len(normalized)
        stats['datetime_formats_after'] = detect_datetime_formats(normalized)
        
        print(f"  Events normalized: {norm_stats['events_normalized']}")
        print(f"  Datapoints normalized: {norm_stats['datapoints_normalized']}")
        if norm_stats['errors'] > 0:
            print(f"  Errors: {norm_stats['errors']}")
        
        # Show format changes
        before = stats['datetime_formats_before']
        after = stats['datetime_formats_after']
        if before.get('old_format', 0) > 0:
            print(f"  Format conversion: {before['old_format']} old format → ISO 8601")
        
        events = normalized
    else:
        stats['after_normalization'] = len(events)
    
    # Step 3: Group and merge close events
    if events:
        print(f"\n=== Step 3: Grouping & Merging (threshold: {time_threshold}) ===")
        grouped, group_info = apply_sliding_window_grouping(
            events,
            time_threshold=time_threshold,
            selection_strategy='alarm_first',
            concatenate_datapoints_flag=concatenate_datapoints,
            show_progress=show_progress
        )
        
        stats['after_grouping'] = len(grouped)
        stats['groups_merged'] = group_info.get('total_groups', 0)
        stats['datapoints_before'] = group_info.get('total_datapoints_before', 0)
        stats['datapoints_after'] = group_info.get('total_datapoints_after', 0)
        
        print(f"  Input: {group_info['total_input_events']} events")
        print(f"  Groups: {stats['groups_merged']} groups identified")
        print(f"  Output: {stats['after_grouping']} merged events")
        if concatenate_datapoints:
            print(f"  Datapoints: {stats['datapoints_before']} → {stats['datapoints_after']}")
        
        events = grouped
    else:
        stats['after_grouping'] = 0
    
    return events, stats


def process_file(input_path: str,
                 output_path: str = None,
                 time_threshold: str = '3min',
                 concatenate_datapoints: bool = True,
                 remove_duplicates: bool = True,
                 duplicate_method: str = 'hash',
                 keep_strategy: str = 'most_datapoints',
                 normalize_datetimes: bool = True,
                 dry_run: bool = False,
                 show_progress: bool = True):
    """
    Process a single OSDB JSON file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (default: input_cleaned.json)
        time_threshold: Time threshold for grouping
        concatenate_datapoints: Whether to concatenate datapoints
        remove_duplicates: Whether to remove duplicates
        duplicate_method: Deduplication method
        keep_strategy: Which duplicate to keep
        dry_run: If True, don't write output file
        show_progress: Show progress bars
    """
    # Generate output path if not provided
    if output_path is None:
        input_pathobj = Path(input_path)
        output_path = str(input_pathobj.parent / f"{input_pathobj.stem}_cleaned{input_pathobj.suffix}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    
    # Load events
    print("\nLoading events...")
    events = load_json_file(input_path)
    print(f"Loaded {len(events)} events")
    
    # Clean events
    cleaned_events, stats = clean_events(
        events,
        time_threshold=time_threshold,
        concatenate_datapoints=concatenate_datapoints,
        remove_duplicates=remove_duplicates,
        duplicate_method=duplicate_method,
        keep_strategy=keep_strategy,
        normalize_datetimes=normalize_datetimes,
        show_progress=show_progress
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Input events:        {stats['input_events']}")
    if remove_duplicates:
        print(f"  After deduplication: {stats['after_dedup']} (-{stats['duplicates_removed']})")
    print(f"  After grouping:      {stats['after_grouping']} (-{stats['input_events'] - stats['after_grouping']} total)")
    if concatenate_datapoints:
        print(f"  Datapoints:          {stats['datapoints_before']} → {stats['datapoints_after']}")
    reduction_pct = (1 - stats['after_grouping'] / stats['input_events']) * 100 if stats['input_events'] > 0 else 0
    print(f"  Reduction:           {reduction_pct:.1f}%")
    print(f"{'='*60}")
    
    # Save output
    if not dry_run:
        print(f"\nSaving to {output_path}...")
        save_json_file(cleaned_events, output_path)
        print("Done!")
    else:
        print("\n[DRY RUN] Output file not written")
    
    return cleaned_events, stats


def main():
    parser = argparse.ArgumentParser(
        description='Clean existing OSDB JSON files by removing duplicates and merging close events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a file with default settings (3min threshold, concatenate datapoints)
  python clean_existing_files.py input.json
  
  # Dry run to see statistics without writing output
  python clean_existing_files.py input.json --dry-run
  
  # Custom output path
  python clean_existing_files.py input.json -o cleaned_output.json
  
  # Disable datapoint concatenation (just group, don't merge datapoints)
  python clean_existing_files.py input.json --no-concatenate
  
  # Skip deduplication step
  python clean_existing_files.py input.json --no-dedup
  
  # Use different time threshold
  python clean_existing_files.py input.json --time-threshold 5min
        """
    )
    
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output JSON file path (default: input_cleaned.json)')
    parser.add_argument('--time-threshold', default='3min',
                       help='Time threshold for grouping (e.g., "3min", "180s") (default: 3min)')
    parser.add_argument('--no-concatenate', dest='concatenate', action='store_false',
                       help='Disable datapoint concatenation')
    parser.add_argument('--no-dedup', dest='remove_duplicates', action='store_false',
                       help='Skip deduplication step')
    parser.add_argument('--no-normalize-datetimes', dest='normalize_datetimes', action='store_false',
                       help='Skip datetime normalization (keep original formats)')
    parser.add_argument('--dedup-method', choices=['hash', 'id'], default='hash',
                       help='Deduplication method (default: hash)')
    parser.add_argument('--keep', choices=['first', 'last', 'most_datapoints'], 
                       default='most_datapoints',
                       help='Which duplicate to keep (default: most_datapoints)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show statistics without writing output file')
    parser.add_argument('--no-progress', dest='show_progress', action='store_false',
                       help='Hide progress bars')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Process file
    process_file(
        input_path=args.input,
        output_path=args.output,
        time_threshold=args.time_threshold,
        concatenate_datapoints=args.concatenate,
        remove_duplicates=args.remove_duplicates,
        duplicate_method=args.dedup_method,
        keep_strategy=args.keep,
        normalize_datetimes=args.normalize_datetimes,
        dry_run=args.dry_run,
        show_progress=args.show_progress
    )


if __name__ == '__main__':
    main()
