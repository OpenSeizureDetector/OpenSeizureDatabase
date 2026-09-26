#!/usr/bin/env python3
"""
Test flattenData.py datapoint validation feature.

This test verifies that the --validate-datapoints option correctly:
1. Detects time gaps in datapoint sequences and leaves them as discontinuities
   (no synthetic filler rows - downstream rolling buffers restart at the gap)
2. Detects and skips overlapping datapoints
3. Handles multiple date/time formats robustly
4. Reports issues to the user when debug mode is enabled
"""

import os
import sys
import json
import tempfile
import subprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def create_test_data_with_gap():
    """Create test data with a 15-second gap between datapoints."""
    return [{
        "id": 100,
        "userId": 1,
        "dataTime": "09-05-2022 02:37:25",
        "type": "seizure",
        "subType": "test",
        "desc": "test event with gap",
        "datapoints": [{
            "id": -1,
            "dataTime": "09-05-2022 02:37:25",
            "hr": 75,
            "rawData": list(range(125))
        }, {
            "id": -1,
            "dataTime": "09-05-2022 02:37:45",  # 20 second gap
            "hr": 75,
            "rawData": list(range(125))
        }]
    }]


def create_test_data_with_overlap():
    """Create test data with overlapping datapoints."""
    return [{
        "id": 101,
        "userId": 2,
        "dataTime": "09-05-2022 03:00:00",
        "type": "seizure",
        "subType": "test",
        "desc": "test event with overlap",
        "datapoints": [{
            "id": -1,
            "dataTime": "09-05-2022 03:00:10",
            "hr": 75,
            "rawData": list(range(125))
        }, {
            "id": -1,
            "dataTime": "09-05-2022 03:00:12",  # Overlaps previous (only 2s apart, needs 5s)
            "hr": 75,
            "rawData": list(range(125))
        }, {
            "id": -1,
            "dataTime": "09-05-2022 03:00:17",  # Valid continuation
            "hr": 75,
            "rawData": list(range(125))
        }]
    }]


def create_test_data_different_formats():
    """Create test data with different date/time formats."""
    return [{
        "id": 102,
        "userId": 3,
        "dataTime": "2022-05-09 04:00:00",
        "type": "seizure",
        "subType": "test",
        "desc": "YYYY-MM-DD format",
        "datapoints": [{
            "id": -1,
            "dataTime": "2022-05-09 04:00:05",
            "hr": 75,
            "rawData": list(range(125))
        }]
    }, {
        "id": 103,
        "userId": 3,
        "dataTime": "09/05/2022 05:00:00",
        "type": "seizure",
        "subType": "test",
        "desc": "DD/MM/YYYY format",
        "datapoints": [{
            "id": -1,
            "dataTime": "09/05/2022 05:00:05",
            "hr": 75,
            "rawData": list(range(125))
        }]
    }, {
        "id": 104,
        "userId": 4,
        "dataTime": "2022-05-09T06:00:00Z",
        "type": "seizure",
        "subType": "test",
        "desc": "ISO 8601 format with Z",
        "datapoints": [{
            "id": -1,
            "dataTime": "2022-05-09T06:00:05Z",
            "hr": 75,
            "rawData": list(range(125))
        }]
    }, {
        "id": 105,
        "userId": 5,
        "dataTime": "2022-05-09T07:00:00.123456Z",
        "type": "seizure",
        "subType": "test",
        "desc": "ISO 8601 with microseconds",
        "datapoints": [{
            "id": -1,
            "dataTime": "2022-05-09T07:00:05.123456Z",
            "hr": 75,
            "rawData": list(range(125))
        }]
    }]


def run_flatten_test(test_data, validate=False):
    """Run flattenData.py on test data and return output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        input_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='r', suffix='.csv', delete=False) as f:
        output_file = f.name
    
    try:
        script_path = os.path.join(os.path.dirname(__file__), '..', 'user_tools', 'nnTraining2', 'flattenData.py')
        # Use the active interpreter so subprocess has the same dependencies as the test runner.
        cmd = [sys.executable, script_path, '-i', input_file, '-o', output_file]
        if validate:
            cmd.append('--validate-datapoints')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'output_lines': lines
        }
    finally:
        os.unlink(input_file)
        os.unlink(output_file)


def test_gap_detection():
    """Test that gaps are detected and left as discontinuities (no filler rows)."""
    print("Testing gap detection (no synthetic rows)...")
    test_data = create_test_data_with_gap()
    result = run_flatten_test(test_data, validate=True)

    # Header + 2 original datapoints only: the 15 s gap is reported but NOT
    # filled - downstream buffers restart at the time discontinuity.
    assert len(result['output_lines']) == 3, f"Expected 3 lines, got {len(result['output_lines'])}"

    # Check that gap was reported (20s end-time delta minus 5s datapoint duration = 15s gap)
    assert "Gap #1: 15000ms" in result['stdout'], "Gap should be reported in stdout"

    # Verify neither data row is synthetic: M000/M001 hold the real samples 0, 1
    for gap_row in result['output_lines'][1:]:
        fields = gap_row.split(',')
        assert fields[10] == '0', "Real row should have M000=0"
        assert fields[11] == '1', "Real row should have M001=1"

    print("  ✓ Gap detection leaves a discontinuity (no synthetic rows)")


def test_overlap_detection():
    """Test that overlapping datapoints are detected and skipped."""
    print("Testing overlap detection...")
    test_data = create_test_data_with_overlap()
    result = run_flatten_test(test_data, validate=True)
    
    # Should have header + 1 original + 0 overlapped (skipped) + 1 original = 3 lines
    assert len(result['output_lines']) == 3, f"Expected 3 lines, got {len(result['output_lines'])}"
    
    # Check that overlap was reported
    assert "Overlap #1" in result['stdout'], "Overlap should be reported in stdout"
    
    print("  ✓ Overlap detection works correctly")


def test_date_format_parsing():
    """Test that different date/time formats are parsed correctly."""
    print("Testing multiple date/time formats...")
    test_data = create_test_data_different_formats()
    result = run_flatten_test(test_data, validate=True)
    
    # Should process all events successfully (4 formats)
    assert result['returncode'] == 0, "Should process successfully"
    assert len(result['output_lines']) == 5, f"Expected 5 lines (header + 4 data), got {len(result['output_lines'])}"
    
    print("  ✓ Date format parsing works correctly")


def test_backward_compatibility():
    """Test that validation is optional and doesn't break normal operation."""
    print("Testing backward compatibility (no validation)...")
    test_data = create_test_data_with_gap()
    result = run_flatten_test(test_data, validate=False)
    
    # Without validation, should just include the 2 original datapoints
    assert len(result['output_lines']) == 3, f"Expected 3 lines (header + 2 data), got {len(result['output_lines'])}"
    
    # No gap should be reported
    assert "Gap" not in result['stdout'], "Gap should not be reported without --validate-datapoints"
    
    print("  ✓ Backward compatibility maintained")


if __name__ == "__main__":
    print("\nRunning flattenData validation tests...\n")
    
    try:
        test_gap_detection()
        test_overlap_detection()
        test_date_format_parsing()
        test_backward_compatibility()
        
        print("\n✓ All tests passed!\n")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
