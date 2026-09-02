#!/usr/bin/env python3
"""
download_and_process.py - Phase 3: Complete Download and Processing Pipeline

This script integrates:
- Phase 1: Event validation and sliding window grouping
- Phase 2: Datapoint concatenation and deduplication  
- Phase 3: Robust downloads with retry, checkpointing, and parallel processing

Usage:
    python3 download_and_process.py --event-ids 123,456,789
    python3 download_and_process.py --event-list events.txt
    python3 download_and_process.py --event-ids 123-456  # Range
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

# Add paths (validation/ is one level deep, so ../src and ../../.. for libosd)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Import Phase 1 & 2 modules
from event_validation import validate_events_batch, print_validation_summary
from event_grouping import apply_sliding_window_grouping
from event_deduplication import remove_duplicate_events
from datetime_normalization import normalize_events_batch, detect_datetime_formats

# Import Phase 3 modules
from event_downloader import (
    DownloadCheckpoint,
    download_events_batch,
    download_events_parallel
)

# Import libosd
import libosd.webApiConnection
import libosd.configUtils


def parse_event_ids(event_ids_str: str) -> List[int]:
    """
    Parse event IDs from string.
    
    Supports:
    - Comma-separated: "123,456,789"
    - Range: "100-200"
    - Mixed: "100,105-110,200"
    """
    ids = set()
    
    parts = event_ids_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Range
            start, end = part.split('-', 1)
            ids.update(range(int(start), int(end) + 1))
        else:
            # Single ID
            ids.add(int(part))
    
    return sorted(list(ids))


def load_event_ids_from_file(filepath: str) -> List[int]:
    """Load event IDs from a text file (one per line or comma-separated)."""
    ids = set()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try parsing as comma-separated
            if ',' in line:
                ids.update(parse_event_ids(line))
            else:
                # Single ID per line
                try:
                    ids.add(int(line))
                except ValueError:
                    print(f"Warning: Skipping invalid line: {line}")
    
    return sorted(list(ids))


def download_and_process_pipeline(
    event_ids: List[int],
    config_file: str = '../../client.cfg',
    output_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    parallel: bool = False,
    num_workers: int = 4,
    max_retries: int = 3,
    time_threshold: str = '3min',
    concatenate_datapoints: bool = True,
    normalize_datetimes: bool = True,
    show_progress: bool = True
):
    """
    Complete pipeline: download → validate → normalize → dedupe → group → save.
    
    Args:
        event_ids: List of event IDs to download
        config_file: Path to client.cfg
        output_file: Output JSON file path
        checkpoint_file: Checkpoint file for resumable downloads
        parallel: Use parallel downloads
        num_workers: Number of parallel workers
        max_retries: Maximum retry attempts per event
        time_threshold: Time threshold for grouping
        concatenate_datapoints: Concatenate datapoints from grouped events
        normalize_datetimes: Normalize datetime formats
        show_progress: Show progress bars
    """
    print("=" * 70)
    print("OSDB Event Download and Processing Pipeline - Phase 3")
    print("=" * 70)
    print(f"\nTotal events to process: {len(event_ids)}")
    print(f"Configuration: {config_file}")
    print(f"Output file: {output_file or 'None (dry run)'}")
    print(f"Parallel mode: {parallel} (workers: {num_workers if parallel else 1})")
    print(f"Checkpoint: {checkpoint_file or 'Disabled'}")
    
    # Step 1: Setup connections
    print(f"\n{'='*70}")
    print("Step 1: Initializing Connections")
    print("=" * 70)
    
    cfg = libosd.configUtils.loadConfig(config_file)
    
    if parallel:
        # Create connection pool
        print(f"Creating connection pool with {num_workers} workers...")
        connections = [
            libosd.webApiConnection.WebApiConnection(
                cfg=cfg['credentialsFname'],
                download=True,
                debug=False
            )
            for _ in range(num_workers)
        ]
    else:
        # Single connection
        connection = libosd.webApiConnection.WebApiConnection(
            cfg=cfg['credentialsFname'],
            download=True,
            debug=False
        )
        connections = [connection]
    
    print(f"✓ Connected with {len(connections)} connection(s)")
    
    # Step 2: Setup checkpoint
    checkpoint = None
    if checkpoint_file:
        print(f"\n{'='*70}")
        print("Step 2: Loading Checkpoint")
        print("=" * 70)
        checkpoint = DownloadCheckpoint(checkpoint_file)
        pending = checkpoint.get_pending(event_ids)
        print(f"  Total events: {len(event_ids)}")
        print(f"  Already completed: {len(event_ids) - len(pending)}")
        print(f"  Pending: {len(pending)}")
    
    # Step 3: Download events
    print(f"\n{'='*70}")
    print("Step 3: Downloading Events")
    print("=" * 70)
    
    invalid_ids = cfg.get('invalidEvents', [])
    
    if parallel and len(connections) > 1:
        from event_downloader import download_events_parallel
        events, download_stats = download_events_parallel(
            connections,
            event_ids,
            checkpoint=checkpoint,
            max_workers=num_workers,
            max_retries=max_retries,
            show_progress=show_progress,
            invalid_ids=invalid_ids
        )
    else:
        events, download_stats = download_events_batch(
            connections[0],
            event_ids,
            checkpoint=checkpoint,
            max_retries=max_retries,
            show_progress=show_progress,
            invalid_ids=invalid_ids
        )
    
    print(f"\nDownload Statistics:")
    print(f"  Total requested: {download_stats.total_requested}")
    print(f"  Successfully downloaded: {download_stats.successful}")
    print(f"  Failed: {download_stats.failed}")
    print(f"  Skipped: {download_stats.skipped}")
    print(f"  Retried: {download_stats.retried}")
    print(f"  Rate: {download_stats.rate():.1f} events/second")
    print(f"  Elapsed: {download_stats.elapsed_time():.1f} seconds")
    
    if not events:
        print("\n⚠ No events downloaded. Exiting.")
        return
    
    # Step 4: Validate events
    print(f"\n{'='*70}")
    print("Step 4: Validating Events")
    print("=" * 70)
    
    valid_events, validation_report = validate_events_batch(
        events,
        show_progress=show_progress
    )
    
    print_validation_summary(validation_report)
    
    if not valid_events:
        print("\n⚠ No valid events after validation. Exiting.")
        return
    
    # Step 5: Normalize datetimes
    if normalize_datetimes:
        print(f"\n{'='*70}")
        print("Step 5: Normalizing Datetime Formats")
        print("=" * 70)
        
        formats_before = detect_datetime_formats(valid_events)
        normalized_events, norm_stats = normalize_events_batch(
            valid_events,
            normalize_datapoints=True,
            show_progress=show_progress
        )
        formats_after = detect_datetime_formats(normalized_events)
        
        print(f"  Events normalized: {norm_stats['events_normalized']}")
        print(f"  Datapoints normalized: {norm_stats['datapoints_normalized']}")
        if formats_before.get('old_format', 0) > 0:
            print(f"  Format conversion: {formats_before['old_format']} old → ISO 8601")
        
        valid_events = normalized_events
    
    # Step 6: Deduplicate
    print(f"\n{'='*70}")
    print("Step 6: Deduplicating Events")
    print("=" * 70)
    
    deduplicated, dedup_info = remove_duplicate_events(
        valid_events,
        method='hash',
        keep='most_datapoints'
    )
    
    print(f"  Input events: {len(valid_events)}")
    print(f"  Duplicates found: {dedup_info['duplicates_found']}")
    print(f"  Duplicates removed: {dedup_info['duplicates_removed']}")
    print(f"  Output events: {dedup_info['total_output']}")
    
    # Step 7: Group and merge
    print(f"\n{'='*70}")
    print("Step 7: Grouping and Merging Events")
    print("=" * 70)
    
    grouped_events, grouping_info = apply_sliding_window_grouping(
        deduplicated,
        time_threshold=time_threshold,
        selection_strategy='alarm_first',
        concatenate_datapoints_flag=concatenate_datapoints,
        show_progress=show_progress
    )
    
    print(f"  Input events: {grouping_info['total_input_events']}")
    print(f"  Groups identified: {grouping_info['total_groups']}")
    print(f"  Output events: {grouping_info['total_output_events']}")
    if concatenate_datapoints:
        print(f"  Datapoints: {grouping_info['total_datapoints_before']} → {grouping_info['total_datapoints_after']}")
    
    # Step 8: Save output
    if output_file:
        print(f"\n{'='*70}")
        print("Step 8: Saving Output")
        print("=" * 70)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(grouped_events, f, indent=2)
        
        print(f"  ✓ Saved {len(grouped_events)} events to {output_file}")
        
        # Save statistics
        stats_file = output_path.with_suffix('.stats.json')
        stats = {
            'download': download_stats.to_dict(),
            'validation': validation_report,
            'deduplication': dedup_info,
            'grouping': grouping_info
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✓ Saved statistics to {stats_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Downloaded: {download_stats.successful} events")
    print(f"  Valid: {len(valid_events)} events")
    print(f"  After deduplication: {len(deduplicated)} events")
    print(f"  After grouping: {len(grouped_events)} events")
    print(f"  Final output: {len(grouped_events)} unique events")
    print("=" * 70)
    print("✓ Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Download and process OSDB events with full Phase 1-3 pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific events
  python3 download_and_process.py --event-ids 12345,12346,12347 -o output.json
  
  # Download range of events
  python3 download_and_process.py --event-ids 10000-10100 -o output.json
  
  # Download from file
  python3 download_and_process.py --event-list events.txt -o output.json
  
  # Parallel download with 8 workers
  python3 download_and_process.py --event-ids 10000-11000 -o output.json --parallel --workers 8
  
  # Resumable download with checkpoint
  python3 download_and_process.py --event-ids 10000-20000 -o output.json --checkpoint download.ckpt
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--event-ids', help='Event IDs (comma-separated or range: 100-200)')
    input_group.add_argument('--event-list', help='File containing event IDs')
    
    # Output
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('--checkpoint', help='Checkpoint file for resumable downloads')
    
    # Connection
    parser.add_argument('--config', default='../../client.cfg', help='Config file (default: ../../client.cfg)')
    parser.add_argument('--parallel', action='store_true', help='Use parallel downloads')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retry attempts (default: 3)')
    
    # Processing
    parser.add_argument('--time-threshold', default='3min', help='Time threshold for grouping (default: 3min)')
    parser.add_argument('--no-concatenate', dest='concatenate', action='store_false',
                       help='Disable datapoint concatenation')
    parser.add_argument('--no-normalize-datetimes', dest='normalize', action='store_false',
                       help='Skip datetime normalization')
    parser.add_argument('--no-progress', dest='show_progress', action='store_false',
                       help='Hide progress bars')
    
    args = parser.parse_args()
    
    # Parse event IDs
    if args.event_ids:
        event_ids = parse_event_ids(args.event_ids)
    else:
        event_ids = load_event_ids_from_file(args.event_list)
    
    if not event_ids:
        print("Error: No event IDs to process")
        sys.exit(1)
    
    # Run pipeline
    download_and_process_pipeline(
        event_ids=event_ids,
        config_file=args.config,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
        parallel=args.parallel,
        num_workers=args.workers,
        max_retries=args.max_retries,
        time_threshold=args.time_threshold,
        concatenate_datapoints=args.concatenate,
        normalize_datetimes=args.normalize,
        show_progress=args.show_progress
    )


if __name__ == '__main__':
    main()
