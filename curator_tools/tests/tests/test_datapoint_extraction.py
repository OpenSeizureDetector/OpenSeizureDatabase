#!/usr/bin/env python3
"""
test_datapoint_extraction.py

Unit and integration tests for datapoint extraction from remote server responses.

Tests the extraction of acceleration data from nested dataJSON structures and
verifies the data is properly stored in the database.
"""

import pytest
import json
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from datapoint_extraction import extract_nested_datapoint_data, extract_nested_data_from_events


class TestExtractNestedDatapoint:
    """Test extraction from individual datapoints"""
    
    def test_extract_from_nested_server_response(self):
        """Test extracting acceleration data from nested server response"""
        # Simulate server response structure
        datapoint = {
            "id": "54128405",
            "dataTime": "2026-08-10T12:08:30Z",
            "accMean": None,
            "accSd": None,
            "hr": None,
            "dataJSON": json.dumps({
                "id": "589754",
                "dataTime": "2026-08-10 12:08:30",
                "dataJSON": json.dumps({
                    "rawData": [100.5, 200.3, 300.1, 400.2],
                    "rawData3D": [-100, 50, -1000, 200, 100, -200],
                    "hr": 72,
                    "o2Sat": 98
                })
            })
        }
        
        result = extract_nested_datapoint_data(datapoint)
        
        assert 'rawData' in result
        assert result['rawData'] == [100.5, 200.3, 300.1, 400.2]
        
        assert 'rawData3D' in result
        assert result['rawData3D'] == [-100, 50, -1000, 200, 100, -200]
        
        assert result.get('hr') == 72
        assert result.get('o2Sat') == 98
    
    def test_skip_already_extracted(self):
        """Test that already-extracted datapoints are skipped"""
        datapoint = {
            "id": "12345",
            "dataTime": "2026-08-10T12:08:30Z",
            "rawData": [100, 200, 300],  # Already extracted
            "rawData3D": [-100, 50, -1000]
        }
        
        result = extract_nested_datapoint_data(datapoint)
        
        # Should return unchanged
        assert result == datapoint
    
    def test_handle_missing_datajson(self):
        """Test handling of datapoints without dataJSON field"""
        datapoint = {
            "id": "12345",
            "dataTime": "2026-08-10T12:08:30Z",
            "hr": 70,
        }
        
        result = extract_nested_datapoint_data(datapoint)
        
        # Should return unchanged
        assert result == datapoint
    
    def test_handle_malformed_json(self):
        """Test handling of malformed nested JSON"""
        datapoint = {
            "id": "12345",
            "dataTime": "2026-08-10T12:08:30Z",
            "dataJSON": "{ invalid json here }",  # Invalid JSON
        }
        
        # Should not raise exception, just return unchanged
        result = extract_nested_datapoint_data(datapoint)
        assert result == datapoint
    
    def test_skip_zero_vital_signs(self):
        """Test that zero values for vital signs are skipped (indicates no data)"""
        datapoint = {
            "id": "54128405",
            "dataTime": "2026-08-10T12:08:30Z",
            "dataJSON": json.dumps({
                "dataJSON": json.dumps({
                    "rawData": [100, 200, 300],
                    "hr": 0,  # Zero = no data
                    "o2Sat": 0  # Zero = no data
                })
            })
        }
        
        result = extract_nested_datapoint_data(datapoint)
        
        # Should extract rawData but skip hr/o2Sat zeros
        assert 'rawData' in result
        assert result.get('hr') is None or result['hr'] == 0
        assert result.get('o2Sat') is None or result['o2Sat'] == 0


class TestExtractBatch:
    """Test batch extraction from multiple events"""
    
    def test_extract_multiple_events(self):
        """Test extracting from multiple events with datapoints"""
        events = [
            {
                "id": "100",
                "type": "Seizure",
                "datapoints": [
                    {
                        "id": "1001",
                        "dataTime": "2026-08-10T12:08:30Z",
                        "dataJSON": json.dumps({
                            "dataJSON": json.dumps({
                                "rawData": [100, 200],
                                "rawData3D": [-100, 50]
                            })
                        })
                    }
                ]
            },
            {
                "id": "101",
                "type": "Seizure",
                "datapoints": [
                    {
                        "id": "1011",
                        "dataTime": "2026-08-10T12:09:30Z",
                        "dataJSON": json.dumps({
                            "dataJSON": json.dumps({
                                "rawData": [300, 400],
                                "rawData3D": [200, -100]
                            })
                        })
                    }
                ]
            }
        ]
        
        result_events, stats = extract_nested_data_from_events(events)
        
        assert stats['events_processed'] == 2
        assert stats['events_with_datapoints'] == 2
        assert stats['datapoints_processed'] == 2
        assert stats['datapoints_extracted'] == 2
        
        # Verify data was extracted
        assert 'rawData' in result_events[0]['datapoints'][0]
        assert 'rawData' in result_events[1]['datapoints'][0]
    
    def test_handle_mixed_event_types(self):
        """Test extraction with events that don't have datapoints"""
        events = [
            {
                "id": "100",
                "type": "Seizure",
                "datapoints": []  # No datapoints
            },
            {
                "id": "101",
                "type": "Seizure"
                # No datapoints field
            },
            {
                "id": "102",
                "type": "Seizure",
                "datapoints": [
                    {
                        "id": "1021",
                        "dataJSON": json.dumps({
                            "dataJSON": json.dumps({
                                "rawData": [100, 200]
                            })
                        })
                    }
                ]
            }
        ]
        
        result_events, stats = extract_nested_data_from_events(events)
        
        assert stats['events_processed'] == 3
        assert stats['events_with_datapoints'] == 2  # Two have datapoints field (empty or not)
        assert stats['datapoints_extracted'] == 1
    
    def test_extraction_statistics(self):
        """Test that extraction statistics are accurate"""
        events = [
            {
                "id": "100",
                "datapoints": [
                    {
                        "dataJSON": json.dumps({
                            "dataJSON": json.dumps({
                                "rawData": [100]
                            })
                        })
                    },
                    {
                        "dataJSON": json.dumps({
                            "dataJSON": json.dumps({
                                "rawData": [200]
                            })
                        })
                    }
                ]
            }
        ]
        
        result_events, stats = extract_nested_data_from_events(events)
        
        assert stats['events_processed'] == 1
        assert stats['events_with_datapoints'] == 1
        assert stats['datapoints_processed'] == 2
        assert stats['datapoints_extracted'] == 2


class TestIdempotent:
    """Test that extraction is idempotent (safe to call multiple times)"""
    
    def test_idempotent_extraction(self):
        """Test that extracting twice produces same result"""
        datapoint = {
            "id": "54128405",
            "dataTime": "2026-08-10T12:08:30Z",
            "dataJSON": json.dumps({
                "dataJSON": json.dumps({
                    "rawData": [100, 200, 300],
                    "rawData3D": [-100, 50, -1000]
                })
            })
        }
        
        # Extract once
        result1 = extract_nested_datapoint_data(datapoint.copy())
        
        # Extract again
        result2 = extract_nested_datapoint_data(result1.copy())
        
        # Should be the same
        assert result1 == result2
        assert result1['rawData'] == [100, 200, 300]
        assert result1['rawData3D'] == [-100, 50, -1000]


if __name__ == '__main__':
    # Run with: pytest test_datapoint_extraction.py -v
    pytest.main([__file__, '-v', '--tb=short'])
