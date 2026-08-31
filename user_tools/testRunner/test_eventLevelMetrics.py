#!/usr/bin/env python3
"""Unit tests for eventLevelMetrics.py

Tests event-level metrics calculation with synthetic data.
"""
import sys
import os
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import eventLevelMetrics


class MockOsdConnection:
    """Mock OsdDbConnection for testing."""
    
    def __init__(self, events):
        self.events = events
    
    def getEvent(self, eventId, includeDatapoints=False):
        for event in self.events:
            if event['id'] == eventId:
                return event
        return None


def test_calculate_event_level_metrics():
    """Test basic event-level metrics calculation."""
    print("Testing calculate_event_level_metrics...")
    
    # Create synthetic test data
    # 3 events: 2 seizures, 1 non-seizure
    eventIdsLst = ['ev1', 'ev2', 'ev3']
    
    mock_events = [
        {
            'id': 'ev1',
            'type': 'seizure',
            'subType': 'tonic-clonic',
            'userId': 'user1',
            'dataTime': '2024-01-01T10:00:00',
            'desc': 'Test seizure 1'
        },
        {
            'id': 'ev2',
            'type': 'seizure',
            'subType': 'focal',
            'userId': 'user2',
            'dataTime': '2024-01-02T10:00:00',
            'desc': 'Test seizure 2'
        },
        {
            'id': 'ev3',
            'type': 'false alarm',
            'subType': '',
            'userId': 'user3',
            'dataTime': '2024-01-03T10:00:00',
            'desc': 'Test false alarm'
        }
    ]
    
    osd = MockOsdConnection(mock_events)
    
    # Results array: [nEvents, nAlgs, nStatus]
    # nStatus: [0=OK, 1=WARNING, 2=ALARM, 3+...]
    # Algorithm detects:
    # - ev1 (seizure): 10 ALARM states -> TP
    # - ev2 (seizure): 5 WARNING states -> FN in standard mode, TP in sensitive mode
    # - ev3 (false alarm): 0 alarms -> TN
    results = np.array([
        [[0, 0, 10, 0, 0]],  # ev1: 10 alarms
        [[0, 5, 0, 0, 0]],   # ev2: 5 warnings
        [[20, 0, 0, 0, 0]]   # ev3: 20 ok
    ])
    
    algNames = ['TestAlg']
    
    # Test standard mode (alarm=2)
    print("  Testing standard mode (threshold=2)...")
    standard_results = eventLevelMetrics.calculate_event_level_metrics(
        results, eventIdsLst, osd, algNames, alarm_threshold=2, debug=False
    )
    
    std_metrics = standard_results['metrics']['TestAlg']
    assert std_metrics['TP'] == 1, f"Expected TP=1, got {std_metrics['TP']}"
    assert std_metrics['FN'] == 1, f"Expected FN=1, got {std_metrics['FN']}"
    assert std_metrics['TN'] == 1, f"Expected TN=1, got {std_metrics['TN']}"
    assert std_metrics['FP'] == 0, f"Expected FP=0, got {std_metrics['FP']}"
    print(f"    ✓ Standard mode: TP={std_metrics['TP']}, FN={std_metrics['FN']}, "
          f"TN={std_metrics['TN']}, FP={std_metrics['FP']}")
    print(f"    ✓ TPR={std_metrics['TPR']:.3f}, TNR={std_metrics['TNR']:.3f}")
    
    # Test sensitive mode (alarm>=1)
    print("  Testing sensitive mode (threshold=1)...")
    sensitive_results = eventLevelMetrics.calculate_event_level_metrics(
        results, eventIdsLst, osd, algNames, alarm_threshold=1, debug=False
    )
    
    sen_metrics = sensitive_results['metrics']['TestAlg']
    assert sen_metrics['TP'] == 2, f"Expected TP=2, got {sen_metrics['TP']}"
    assert sen_metrics['FN'] == 0, f"Expected FN=0, got {sen_metrics['FN']}"
    assert sen_metrics['TN'] == 1, f"Expected TN=1, got {sen_metrics['TN']}"
    assert sen_metrics['FP'] == 0, f"Expected FP=0, got {sen_metrics['FP']}"
    print(f"    ✓ Sensitive mode: TP={sen_metrics['TP']}, FN={sen_metrics['FN']}, "
          f"TN={sen_metrics['TN']}, FP={sen_metrics['FP']}")
    print(f"    ✓ TPR={sen_metrics['TPR']:.3f}, TNR={sen_metrics['TNR']:.3f}")
    
    # Verify sensitivity increase
    assert sen_metrics['TPR'] > std_metrics['TPR'], "Sensitive mode should have higher TPR"
    print(f"    ✓ Sensitivity increase: {sen_metrics['TPR'] - std_metrics['TPR']:.3f}")
    
    print("  ✓ calculate_event_level_metrics passed!\n")


def test_compare_sensitivity_modes():
    """Test sensitivity mode comparison."""
    print("Testing compare_sensitivity_modes...")
    
    eventIdsLst = ['ev1', 'ev2']
    
    mock_events = [
        {
            'id': 'ev1',
            'type': 'seizure',
            'subType': '',
            'userId': 'user1',
            'dataTime': '2024-01-01T10:00:00',
            'desc': 'Seizure'
        },
        {
            'id': 'ev2',
            'type': 'false alarm',
            'subType': '',
            'userId': 'user2',
            'dataTime': '2024-01-02T10:00:00',
            'desc': 'False alarm'
        }
    ]
    
    osd = MockOsdConnection(mock_events)
    
    # ev1 (seizure): 1 WARNING, 5 ALARM -> detected in both modes
    # ev2 (false alarm): 2 WARNING -> FP in sensitive mode only
    results = np.array([
        [[0, 1, 5, 0, 0]],
        [[10, 2, 0, 0, 0]]
    ])
    
    algNames = ['TestAlg']
    
    comparison = eventLevelMetrics.compare_sensitivity_modes(
        results, eventIdsLst, osd, algNames, debug=False
    )
    
    assert 'standard' in comparison, "Missing 'standard' results"
    assert 'sensitive' in comparison, "Missing 'sensitive' results"
    
    std_metrics = comparison['standard']['metrics']['TestAlg']
    sen_metrics = comparison['sensitive']['metrics']['TestAlg']
    
    print(f"  Standard: TP={std_metrics['TP']}, FP={std_metrics['FP']}")
    print(f"  Sensitive: TP={sen_metrics['TP']}, FP={sen_metrics['FP']}")
    
    assert std_metrics['TP'] == 1, "Standard mode should detect seizure"
    assert std_metrics['FP'] == 0, "Standard mode should not false alarm"
    assert sen_metrics['TP'] == 1, "Sensitive mode should detect seizure"
    assert sen_metrics['FP'] == 1, "Sensitive mode should have 1 false alarm"
    
    print("  ✓ compare_sensitivity_modes passed!\n")


def test_save_event_results_csv():
    """Test CSV file generation."""
    print("Testing save_event_results_csv...")
    
    import tempfile
    import csv
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        event_predictions = [
            {
                'eventId': 'ev1',
                'userId': 'user1',
                'type': 'seizure',
                'subType': 'tonic-clonic',
                'dataTime': '2024-01-01T10:00:00',
                'desc': 'Test event',
                'true_label': 1,
                'predictions': {'TestAlg': 1},
                'alarm_counts': {
                    'TestAlg': {'alarm': 5, 'warning': 2, 'ok': 3}
                }
            }
        ]
        
        algNames = ['TestAlg']
        
        csv_path = eventLevelMetrics.save_event_results_csv(
            tmpdir, event_predictions, algNames, mode_name='test'
        )
        
        # Verify file was created
        assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"
        
        # Verify CSV contents
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
            row = rows[0]
            
            assert row['EventID'] == 'ev1', "EventID mismatch"
            assert row['TrueLabel'] == '1', "TrueLabel mismatch"
            assert row['TestAlg_Pred'] == '1', "Prediction mismatch"
            assert row['TestAlg_AlarmCount'] == '5', "AlarmCount mismatch"
            assert row['TestAlg_WarnCount'] == '2', "WarnCount mismatch"
            
        print(f"  ✓ CSV file created and validated: {csv_path}")
        print("  ✓ save_event_results_csv passed!\n")


def run_all_tests():
    """Run all unit tests."""
    print("="*80)
    print("RUNNING EVENT-LEVEL METRICS UNIT TESTS")
    print("="*80 + "\n")
    
    try:
        test_calculate_event_level_metrics()
        test_compare_sensitivity_modes()
        test_save_event_results_csv()
        
        print("="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
