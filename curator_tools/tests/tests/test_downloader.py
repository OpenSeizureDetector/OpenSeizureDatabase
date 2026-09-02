#!/usr/bin/env python3
"""
test_downloader.py - Unit tests for event download system
"""

import sys
import os
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from event_downloader import (
    DownloadStats,
    DownloadCheckpoint,
    download_event_with_retry,
    download_events_batch
)


class TestDownloadStats(unittest.TestCase):
    """Test download statistics tracking."""
    
    def test_init(self):
        """Test stats initialization."""
        stats = DownloadStats()
        assert stats.successful == 0
        assert stats.failed == 0
        assert stats.retried == 0
        assert stats.skipped == 0
    
    def test_record_operations(self):
        """Test recording various operations."""
        stats = DownloadStats()
        
        stats.record_success()
        stats.record_success()
        assert stats.successful == 2
        
        stats.record_failure(123)
        assert stats.failed == 1
        assert 123 in stats.failed_event_ids
        
        stats.record_retry()
        assert stats.retried == 1
        
        stats.record_skip()
        assert stats.skipped == 1
    
    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        import time
        stats = DownloadStats()
        
        stats.start()
        time.sleep(0.1)
        stats.finish()
        
        elapsed = stats.elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be quick
    
    def test_rate_calculation(self):
        """Test download rate calculation."""
        import time
        stats = DownloadStats()
        
        stats.start()
        stats.record_success()
        stats.record_success()
        time.sleep(0.1)
        
        rate = stats.rate()
        assert rate > 0
        assert rate < 100  # Reasonable rate
    
    def test_to_dict(self):
        """Test stats export to dictionary."""
        stats = DownloadStats()
        stats.total_requested = 10
        stats.record_success()
        stats.record_failure(123)
        
        data = stats.to_dict()
        assert data['total_requested'] == 10
        assert data['successful'] == 1
        assert data['failed'] == 1
        assert 123 in data['failed_event_ids']


class TestDownloadCheckpoint(unittest.TestCase):
    """Test checkpoint system."""
    
    def setUp(self):
        """Create temporary checkpoint file."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_file = os.path.join(self.temp_dir, 'checkpoint.json')
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_new_checkpoint(self):
        """Test creating new checkpoint."""
        checkpoint = DownloadCheckpoint(self.checkpoint_file)
        assert len(checkpoint.completed_ids) == 0
        assert len(checkpoint.failed_ids) == 0
    
    def test_mark_completed(self):
        """Test marking events as completed."""
        checkpoint = DownloadCheckpoint(self.checkpoint_file)
        
        checkpoint.mark_completed(123)
        assert checkpoint.is_completed(123)
        assert not checkpoint.is_failed(123)
    
    def test_mark_failed(self):
        """Test marking events as failed."""
        checkpoint = DownloadCheckpoint(self.checkpoint_file)
        
        checkpoint.mark_failed(456)
        assert checkpoint.is_failed(456)
        assert not checkpoint.is_completed(456)
    
    def test_persistence(self):
        """Test checkpoint persistence across instances."""
        # Create checkpoint and mark some events
        checkpoint1 = DownloadCheckpoint(self.checkpoint_file)
        checkpoint1.mark_completed(123)
        checkpoint1.mark_completed(456)
        checkpoint1.mark_failed(789)
        
        # Load checkpoint in new instance
        checkpoint2 = DownloadCheckpoint(self.checkpoint_file)
        assert checkpoint2.is_completed(123)
        assert checkpoint2.is_completed(456)
        assert checkpoint2.is_failed(789)
    
    def test_get_pending(self):
        """Test getting pending event IDs."""
        checkpoint = DownloadCheckpoint(self.checkpoint_file)
        checkpoint.mark_completed(1)
        checkpoint.mark_completed(2)
        
        all_ids = [1, 2, 3, 4, 5]
        pending = checkpoint.get_pending(all_ids)
        
        assert 1 not in pending
        assert 2 not in pending
        assert 3 in pending
        assert 4 in pending
        assert 5 in pending
    
    def test_clear(self):
        """Test clearing checkpoint."""
        checkpoint = DownloadCheckpoint(self.checkpoint_file)
        checkpoint.mark_completed(123)
        checkpoint.mark_failed(456)
        
        checkpoint.clear()
        
        assert len(checkpoint.completed_ids) == 0
        assert len(checkpoint.failed_ids) == 0


class TestDownloadWithRetry(unittest.TestCase):
    """Test download with retry logic."""
    
    def test_successful_download(self):
        """Test successful download on first try."""
        mock_connection = Mock()
        mock_connection.getEvent.return_value = {'id': 123, 'data': 'test'}
        
        result = download_event_with_retry(mock_connection, 123, max_retries=3)
        
        assert result is not None
        assert result['id'] == 123
        assert mock_connection.getEvent.call_count == 1
    
    def test_retry_on_failure(self):
        """Test retry logic when download fails."""
        mock_connection = Mock()
        mock_connection.getEvent.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            {'id': 123, 'data': 'test'}  # Success on 3rd try
        ]
        
        stats = DownloadStats()
        result = download_event_with_retry(
            mock_connection,
            123,
            max_retries=3,
            initial_delay=0.01,  # Fast for testing
            stats=stats
        )
        
        assert result is not None
        assert result['id'] == 123
        assert mock_connection.getEvent.call_count == 3
        assert stats.retried == 2  # 2 retries before success
    
    def test_exhausted_retries(self):
        """Test behavior when all retries are exhausted."""
        mock_connection = Mock()
        mock_connection.getEvent.side_effect = Exception("Persistent error")
        
        stats = DownloadStats()
        result = download_event_with_retry(
            mock_connection,
            123,
            max_retries=2,
            initial_delay=0.01,
            stats=stats
        )
        
        assert result is None
        assert stats.failed == 1
        assert 123 in stats.failed_event_ids
        assert mock_connection.getEvent.call_count == 3  # Initial + 2 retries


class TestDownloadBatch(unittest.TestCase):
    """Test batch download functionality."""
    
    def test_batch_download_success(self):
        """Test successful batch download."""
        mock_connection = Mock()
        mock_connection.getEvent.side_effect = [
            {'id': 1, 'data': 'event1'},
            {'id': 2, 'data': 'event2'},
            {'id': 3, 'data': 'event3'}
        ]
        
        events, stats = download_events_batch(
            mock_connection,
            [1, 2, 3],
            max_retries=1,
            show_progress=False
        )
        
        assert len(events) == 3
        assert stats.successful == 3
        assert stats.failed == 0
    
    def test_skip_invalid_events(self):
        """Test skipping invalid events."""
        mock_connection = Mock()
        mock_connection.getEvent.return_value = {'id': 2, 'data': 'event2'}
        
        events, stats = download_events_batch(
            mock_connection,
            [1, 2, 3],
            skip_invalid=True,
            invalid_ids=[1, 3],
            max_retries=1,
            show_progress=False
        )
        
        assert len(events) == 1
        assert events[0]['id'] == 2
        assert stats.skipped == 2
        assert mock_connection.getEvent.call_count == 1
    
    def test_checkpoint_integration(self):
        """Test checkpoint integration with batch download."""
        temp_dir = tempfile.mkdtemp()
        checkpoint_file = os.path.join(temp_dir, 'test_checkpoint.json')
        
        try:
            checkpoint = DownloadCheckpoint(checkpoint_file)
            checkpoint.mark_completed(1)  # Already downloaded
            
            mock_connection = Mock()
            mock_connection.getEvent.side_effect = [
                {'id': 2, 'data': 'event2'},
                {'id': 3, 'data': 'event3'}
            ]
            
            events, stats = download_events_batch(
                mock_connection,
                [1, 2, 3],
                checkpoint=checkpoint,
                max_retries=1,
                show_progress=False
            )
            
            assert len(events) == 2  # Only 2 and 3
            assert stats.skipped == 1  # Event 1 was skipped
            assert checkpoint.is_completed(2)
            assert checkpoint.is_completed(3)
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
