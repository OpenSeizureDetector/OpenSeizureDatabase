#!/usr/bin/env python3
"""
event_downloader.py - Phase 3: Robust Event Download System

Features:
- Retry logic with exponential backoff
- Progress tracking and checkpointing
- Parallel downloads with connection pooling
- Batch download management
- Error handling and recovery
"""

import sys
import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path for libosd imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
        def update(self, n=1):
            self.n += n
        def close(self):
            pass


class DownloadStats:
    """Track download statistics."""
    
    def __init__(self):
        self.total_requested = 0
        self.successful = 0
        self.failed = 0
        self.retried = 0
        self.skipped = 0
        self.start_time = None
        self.end_time = None
        self.failed_event_ids = []
        self._lock = threading.Lock()
    
    def start(self):
        self.start_time = datetime.now()
    
    def finish(self):
        self.end_time = datetime.now()
    
    def record_success(self):
        with self._lock:
            self.successful += 1
    
    def record_failure(self, event_id: int):
        with self._lock:
            self.failed += 1
            self.failed_event_ids.append(event_id)
    
    def record_retry(self):
        with self._lock:
            self.retried += 1
    
    def record_skip(self):
        with self._lock:
            self.skipped += 1
    
    def elapsed_time(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def rate(self) -> float:
        """Return download rate in events/second."""
        elapsed = self.elapsed_time()
        if elapsed > 0:
            return self.successful / elapsed
        return 0.0
    
    def to_dict(self) -> dict:
        """Export stats as dictionary."""
        return {
            'total_requested': self.total_requested,
            'successful': self.successful,
            'failed': self.failed,
            'retried': self.retried,
            'skipped': self.skipped,
            'failed_event_ids': self.failed_event_ids,
            'elapsed_seconds': self.elapsed_time(),
            'rate_per_second': self.rate(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class DownloadCheckpoint:
    """Manage download checkpoints for resumable downloads."""
    
    def __init__(self, checkpoint_file: str):
        self.checkpoint_file = checkpoint_file
        self.completed_ids = set()
        self.failed_ids = set()
        self.metadata = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load checkpoint from file if it exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    self.completed_ids = set(data.get('completed', []))
                    self.failed_ids = set(data.get('failed', []))
                    self.metadata = data.get('metadata', {})
            except Exception as e:
                print(f"Warning: Could not load checkpoint file: {e}")
    
    def _save(self):
        """Save checkpoint to file."""
        try:
            data = {
                'completed': list(self.completed_ids),
                'failed': list(self.failed_ids),
                'metadata': self.metadata,
                'last_updated': datetime.now().isoformat()
            }
            # Write atomically
            tmp_file = self.checkpoint_file + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_file, self.checkpoint_file)
        except Exception as e:
            print(f"Warning: Could not save checkpoint file: {e}")
    
    def mark_completed(self, event_id: int):
        """Mark an event as successfully downloaded."""
        with self._lock:
            self.completed_ids.add(event_id)
            self.failed_ids.discard(event_id)
            self._save()
    
    def mark_failed(self, event_id: int):
        """Mark an event as failed."""
        with self._lock:
            self.failed_ids.add(event_id)
            self._save()
    
    def is_completed(self, event_id: int) -> bool:
        """Check if event was already downloaded."""
        with self._lock:
            return event_id in self.completed_ids
    
    def is_failed(self, event_id: int) -> bool:
        """Check if event previously failed."""
        with self._lock:
            return event_id in self.failed_ids
    
    def get_pending(self, event_ids: List[int]) -> List[int]:
        """Return list of event IDs that need to be downloaded."""
        with self._lock:
            return [eid for eid in event_ids if eid not in self.completed_ids]
    
    def clear(self):
        """Clear checkpoint data."""
        with self._lock:
            self.completed_ids.clear()
            self.failed_ids.clear()
            self.metadata.clear()
            self._save()


def download_event_with_retry(osd_connection,
                               event_id: int,
                               max_retries: int = 3,
                               initial_delay: float = 1.0,
                               backoff_factor: float = 2.0,
                               stats: Optional[DownloadStats] = None) -> Optional[Dict[str, Any]]:
    """
    Download a single event with exponential backoff retry logic.
    
    Args:
        osd_connection: WebApiConnection object
        event_id: Event ID to download
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        stats: DownloadStats object to track statistics
        
    Returns:
        Event dictionary or None if download failed after all retries
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            event = osd_connection.getEvent(event_id, includeDatapoints=True)
            if stats:
                stats.record_success()
            return event
        
        except Exception as e:
            last_error = e
            
            if attempt < max_retries:
                if stats:
                    stats.record_retry()
                
                # Exponential backoff
                time.sleep(delay)
                delay *= backoff_factor
            else:
                # Final attempt failed
                if stats:
                    stats.record_failure(event_id)
                print(f"Error: Failed to download event {event_id} after {max_retries + 1} attempts: {e}")
                return None
    
    return None


def download_events_batch(osd_connection,
                          event_ids: List[int],
                          checkpoint: Optional[DownloadCheckpoint] = None,
                          max_retries: int = 3,
                          show_progress: bool = True,
                          skip_invalid: bool = True,
                          invalid_ids: Optional[List[int]] = None) -> Tuple[List[Dict], DownloadStats]:
    """
    Download a batch of events sequentially with retry logic and checkpointing.
    
    Args:
        osd_connection: WebApiConnection object
        event_ids: List of event IDs to download
        checkpoint: DownloadCheckpoint object for resumable downloads
        max_retries: Maximum retry attempts per event
        show_progress: Show progress bar
        skip_invalid: Skip events marked as invalid
        invalid_ids: List of invalid event IDs to skip
        
    Returns:
        Tuple of (downloaded_events, download_stats)
    """
    stats = DownloadStats()
    stats.total_requested = len(event_ids)
    stats.start()
    
    invalid_ids = set(invalid_ids or [])
    downloaded_events = []
    
    # Filter out invalid and already completed events
    pending_ids = []
    for eid in event_ids:
        if skip_invalid and eid in invalid_ids:
            stats.record_skip()
            continue
        if checkpoint and checkpoint.is_completed(eid):
            stats.record_skip()
            continue
        pending_ids.append(eid)
    
    # Download events
    progress = tqdm(pending_ids, desc="Downloading events", disable=not show_progress)
    
    for event_id in progress:
        event = download_event_with_retry(
            osd_connection,
            event_id,
            max_retries=max_retries,
            stats=stats
        )
        
        if event is not None:
            downloaded_events.append(event)
            if checkpoint:
                checkpoint.mark_completed(event_id)
        else:
            if checkpoint:
                checkpoint.mark_failed(event_id)
        
        # Update progress bar with stats
        if show_progress:
            progress.set_postfix({
                'success': stats.successful,
                'failed': stats.failed,
                'rate': f'{stats.rate():.1f}/s'
            })
    
    progress.close()
    stats.finish()
    
    return downloaded_events, stats


def download_events_parallel(osd_connections: List,
                             event_ids: List[int],
                             checkpoint: Optional[DownloadCheckpoint] = None,
                             max_workers: int = 4,
                             max_retries: int = 3,
                             show_progress: bool = True,
                             skip_invalid: bool = True,
                             invalid_ids: Optional[List[int]] = None) -> Tuple[List[Dict], DownloadStats]:
    """
    Download events in parallel using multiple connections.
    
    Args:
        osd_connections: List of WebApiConnection objects (connection pool)
        event_ids: List of event IDs to download
        checkpoint: DownloadCheckpoint object
        max_workers: Maximum number of parallel workers
        max_retries: Maximum retry attempts per event
        show_progress: Show progress bar
        skip_invalid: Skip invalid events
        invalid_ids: List of invalid event IDs
        
    Returns:
        Tuple of (downloaded_events, download_stats)
    """
    stats = DownloadStats()
    stats.total_requested = len(event_ids)
    stats.start()
    
    invalid_ids = set(invalid_ids or [])
    downloaded_events = []
    
    # Filter pending events
    pending_ids = []
    for eid in event_ids:
        if skip_invalid and eid in invalid_ids:
            stats.record_skip()
            continue
        if checkpoint and checkpoint.is_completed(eid):
            stats.record_skip()
            continue
        pending_ids.append(eid)
    
    if not pending_ids:
        stats.finish()
        return downloaded_events, stats
    
    # Create connection pool round-robin
    connection_pool = osd_connections * ((len(pending_ids) // len(osd_connections)) + 1)
    
    # Download in parallel
    with ThreadPoolExecutor(max_workers=min(max_workers, len(osd_connections))) as executor:
        # Submit download tasks
        future_to_id = {}
        for event_id, connection in zip(pending_ids, connection_pool[:len(pending_ids)]):
            future = executor.submit(
                download_event_with_retry,
                connection,
                event_id,
                max_retries,
                1.0,  # initial_delay
                2.0,  # backoff_factor
                stats
            )
            future_to_id[future] = event_id
        
        # Collect results with progress bar
        progress = tqdm(total=len(pending_ids), desc="Downloading events", disable=not show_progress)
        
        for future in as_completed(future_to_id):
            event_id = future_to_id[future]
            
            try:
                event = future.result()
                if event is not None:
                    downloaded_events.append(event)
                    if checkpoint:
                        checkpoint.mark_completed(event_id)
                else:
                    if checkpoint:
                        checkpoint.mark_failed(event_id)
            
            except Exception as e:
                print(f"Unexpected error downloading event {event_id}: {e}")
                stats.record_failure(event_id)
                if checkpoint:
                    checkpoint.mark_failed(event_id)
            
            progress.update(1)
            if show_progress:
                progress.set_postfix({
                    'success': stats.successful,
                    'failed': stats.failed,
                    'rate': f'{stats.rate():.1f}/s'
                })
        
        progress.close()
    
    stats.finish()
    return downloaded_events, stats


# Example usage and testing
if __name__ == '__main__':
    print("Event Downloader Module - Phase 3")
    print("=" * 60)
    print("\nThis module provides:")
    print("  - Retry logic with exponential backoff")
    print("  - Checkpoint system for resumable downloads")
    print("  - Parallel downloads with connection pooling")
    print("  - Progress tracking and statistics")
    print("\nImport this module to use in your download scripts.")
