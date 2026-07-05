#!/usr/bin/env python3
"""
test_publication.py - Tests for Multi-Format Publication (Phase 5)

Tests verify:
- Format conversions preserve data
- Compression works correctly
- Parquet format is queryable
- File sizes are as expected
"""

import sys
import os
import json
import gzip
import tempfile
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from osdb_publication import OsdbPublisher


class TestJSONPublication(unittest.TestCase):
    """Test JSON publication."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.publisher = OsdbPublisher(debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_publish_json(self):
        """Test basic JSON publication."""
        events = [
            {'id': 1, 'userId': 42, 'type': 'Seizure', 'datapoints': []},
            {'id': 2, 'userId': 42, 'type': 'FalseAlarm', 'datapoints': []}
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json')
        stats = self.publisher.publish_json(events, output_path)
        
        assert stats['event_count'] == 2
        assert stats['format'] == 'json'
        assert os.path.exists(output_path)
        
        # Verify content
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]['id'] == 1
    
    def test_json_preserves_datapoints(self):
        """Test that JSON preserves datapoints."""
        events = [
            {
                'id': 1,
                'userId': 42,
                'type': 'Seizure',
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 80},
                    {'dataTime': '2024-01-01T10:00:05Z', 'hr': 85}
                ]
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json')
        self.publisher.publish_json(events, output_path)
        
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded[0]['datapoints']) == 2
        assert loaded[0]['datapoints'][0]['hr'] == 80


class TestCompressedJSONPublication(unittest.TestCase):
    """Test compressed JSON publication."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.publisher = OsdbPublisher(debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_publish_json_gz(self):
        """Test compressed JSON publication."""
        events = [
            {'id': i, 'userId': 42, 'type': 'Seizure', 'datapoints': []}
            for i in range(100)
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json.gz')
        stats = self.publisher.publish_json_gz(events, output_path)
        
        assert stats['event_count'] == 100
        assert stats['format'] == 'json.gz'
        assert os.path.exists(output_path)
        assert stats['compression_ratio_percent'] > 0
    
    def test_json_gz_can_be_decompressed(self):
        """Test that compressed JSON can be read."""
        events = [
            {'id': 1, 'userId': 42, 'type': 'Seizure', 'datapoints': []},
            {'id': 2, 'userId': 43, 'type': 'FalseAlarm', 'datapoints': []}
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.json.gz')
        self.publisher.publish_json_gz(events, output_path)
        
        # Read back
        with gzip.open(output_path, 'rt') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 2
        assert loaded[0]['id'] == 1
        assert loaded[1]['id'] == 2
    
    def test_compression_saves_space(self):
        """Test that compression actually reduces file size."""
        # Create events with repetitive data (compresses well)
        events = [
            {
                'id': i,
                'userId': 42,
                'type': 'Seizure',
                'desc': 'Repetitive description' * 10,
                'datapoints': [
                    {'dataTime': f'2024-01-01T10:{j:02d}:00Z', 'hr': 80}
                    for j in range(20)
                ]
            }
            for i in range(50)
        ]
        
        json_path = os.path.join(self.temp_dir, 'test.json')
        gz_path = os.path.join(self.temp_dir, 'test.json.gz')
        
        self.publisher.publish_json(events, json_path, pretty=False)
        stats_gz = self.publisher.publish_json_gz(events, gz_path)
        
        json_size = os.path.getsize(json_path)
        gz_size = os.path.getsize(gz_path)
        
        assert gz_size < json_size
        assert stats_gz['compression_ratio_percent'] > 30  # Should compress well


class TestParquetPublication(unittest.TestCase):
    """Test Parquet publication."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.publisher = OsdbPublisher(debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_publish_parquet_flattened(self):
        """Test flattened Parquet publication."""
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pandas and pyarrow required for Parquet tests")
        
        events = [
            {
                'id': 1,
                'userId': 42,
                'type': 'Seizure',
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 80, 'o2Sat': 95},
                    {'dataTime': '2024-01-01T10:00:05Z', 'hr': 85, 'o2Sat': 94}
                ]
            },
            {
                'id': 2,
                'userId': 42,
                'type': 'FalseAlarm',
                'datapoints': [
                    {'dataTime': '2024-01-01T11:00:00Z', 'hr': 70, 'o2Sat': 98}
                ]
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.parquet')
        stats = self.publisher.publish_parquet(events, output_path, flatten_datapoints=True)
        
        assert stats['event_count'] == 2
        assert stats['row_count'] == 3  # 2 + 1 datapoints
        assert stats['format'] == 'parquet'
        assert os.path.exists(output_path)
        
        # Verify can be read
        df = pd.read_parquet(output_path)
        assert len(df) == 3
        assert 'event_id' in df.columns
        assert 'datapoint_hr' in df.columns
    
    def test_parquet_preserves_data(self):
        """Test that Parquet preserves data correctly."""
        try:
            import pandas as pd
            import pyarrow
        except ImportError:
            self.skipTest("pandas and pyarrow required for Parquet tests")
        
        events = [
            {
                'id': 100,
                'userId': 42,
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'datapoints': [
                    {'dataTime': '2024-01-01T10:00:00Z', 'hr': 80}
                ]
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.parquet')
        self.publisher.publish_parquet(events, output_path)
        
        df = pd.read_parquet(output_path)
        
        assert df.iloc[0]['event_id'] == 100
        assert df.iloc[0]['userId'] == 42
        assert df.iloc[0]['event_type'] == 'Seizure'
        assert df.iloc[0]['event_subtype'] == 'Tonic-Clonic'
        assert df.iloc[0]['datapoint_hr'] == 80


class TestCSVPublication(unittest.TestCase):
    """Test CSV publication."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.publisher = OsdbPublisher(debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_publish_csv(self):
        """Test CSV publication (metadata only)."""
        events = [
            {
                'id': 1,
                'userId': 42,
                'dataTime': '2024-01-01T10:00:00Z',
                'type': 'Seizure',
                'subType': 'Tonic-Clonic',
                'datapoints': [{'hr': 80}]  # Should be excluded from CSV
            },
            {
                'id': 2,
                'userId': 43,
                'dataTime': '2024-01-01T11:00:00Z',
                'type': 'FalseAlarm',
                'datapoints': []
            }
        ]
        
        output_path = os.path.join(self.temp_dir, 'test.csv')
        stats = self.publisher.publish_csv(events, output_path)
        
        assert stats['event_count'] == 2
        assert stats['format'] == 'csv'
        assert os.path.exists(output_path)
        
        # Verify content
        import csv
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]['id'] == '1'
        assert rows[0]['userId'] == '42'
        assert rows[0]['type'] == 'Seizure'


class TestMultiFormatPublication(unittest.TestCase):
    """Test publishing in multiple formats simultaneously."""
    
    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.publisher = OsdbPublisher(debug=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_publish_all_formats(self):
        """Test publishing in all formats."""
        events = [
            {
                'id': i,
                'userId': 42,
                'type': 'Seizure',
                'dataTime': f'2024-01-01T{i:02d}:00:00Z',
                'datapoints': [
                    {'dataTime': f'2024-01-01T{i:02d}:00:00Z', 'hr': 80 + i}
                ]
            }
            for i in range(10)
        ]
        
        results = self.publisher.publish_all_formats(
            events,
            'test_osdb',
            formats=['json', 'json.gz', 'csv'],
            output_dir=self.temp_dir
        )
        
        assert 'json' in results
        assert 'json.gz' in results
        assert 'csv' in results
        
        # Verify all files exist
        assert os.path.exists(os.path.join(self.temp_dir, 'test_osdb.json'))
        assert os.path.exists(os.path.join(self.temp_dir, 'test_osdb.json.gz'))
        assert os.path.exists(os.path.join(self.temp_dir, 'test_osdb.csv'))


class TestFormatConsistency(unittest.TestCase):
    """Test consistency across different formats."""
    
    def test_json_formats_are_equivalent(self):
        """Test that JSON and JSON.GZ contain same data."""
        temp_dir = tempfile.mkdtemp()
        publisher = OsdbPublisher(debug=False)
        
        try:
            events = [
                {
                    'id': i,
                    'userId': 42,
                    'type': 'Seizure',
                    'datapoints': [{'hr': 80}]
                }
                for i in range(5)
            ]
            
            json_path = os.path.join(temp_dir, 'test.json')
            gz_path = os.path.join(temp_dir, 'test.json.gz')
            
            publisher.publish_json(events, json_path)
            publisher.publish_json_gz(events, gz_path)
            
            # Load both
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            with gzip.open(gz_path, 'rt') as f:
                gz_data = json.load(f)
            
            # Compare
            assert json_data == gz_data
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
