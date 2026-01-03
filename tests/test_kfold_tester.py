#!/usr/bin/env python3
"""Test script to verify k-fold testing functionality in nnTester"""

import sys
import os
import json
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_kfold_validation():
    """Test that k-fold validation properly validates fold count"""
    print("Testing k-fold validation...")
    
    # Create temporary directory structure
    tmpdir = tempfile.mkdtemp()
    try:
        # Create only 3 fold directories
        for i in range(3):
            fold_dir = os.path.join(tmpdir, f"fold{i}")
            os.makedirs(fold_dir)
            
            # Create minimal test results
            results = {
                'accuracy': 0.85 + i * 0.01,
                'tpr': 0.80 + i * 0.01,
                'fpr': 0.10 - i * 0.01,
                'accuracyOsd': 0.75,
                'tprOsd': 0.70,
                'fprOsd': 0.15,
                'event_tpr': 0.82 + i * 0.01,
                'event_fpr': 0.12 - i * 0.01,
                'osd_event_tpr': 0.72,
                'osd_event_fpr': 0.18,
                'tp': 80, 'fp': 10, 'tn': 90, 'fn': 20,
                'tpOsd': 70, 'fpOsd': 15, 'tnOsd': 85, 'fnOsd': 30,
                'event_tp': 8, 'event_fp': 1, 'event_tn': 9, 'event_fn': 2,
                'osd_event_tp': 7, 'osd_event_fp': 2, 'osd_event_tn': 8, 'osd_event_fn': 3,
                'num_positive_epoch': 100,
                'num_positive_event': 10
            }
            
            with open(os.path.join(fold_dir, 'testResults.json'), 'w') as f:
                json.dump(results, f)
        
        # Test 1: Correct number of folds (should succeed)
        print("  Test 1: Correct fold count (3 folds)...")
        from user_tools.nnTraining2 import nnTester
        
        mock_config = {
            'debug': False,
            'modelConfig': {'modelFname': 'test_model', 'framework': 'tensorflow'},
            'dataFileNames': {}
        }
        
        try:
            # This should succeed
            nnTester.testKFold(mock_config, kfold=3, dataDir=tmpdir, rerun=False, debug=False)
            print("    ✓ Correctly handled 3 folds")
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            return False
        
        # Verify summary files were created
        if not os.path.exists(os.path.join(tmpdir, 'kfold_summary.txt')):
            print("    ✗ kfold_summary.txt not created")
            return False
        if not os.path.exists(os.path.join(tmpdir, 'kfold_summary.json')):
            print("    ✗ kfold_summary.json not created")
            return False
        
        # Test 2: Wrong number of folds (should fail)
        print("  Test 2: Wrong fold count (expecting 5, have 3)...")
        try:
            nnTester.testKFold(mock_config, kfold=5, dataDir=tmpdir, rerun=False, debug=False)
            print("    ✗ Should have raised ValueError for mismatched fold count")
            return False
        except ValueError as e:
            if "fold3" in str(e) and "does not exist" in str(e):
                print(f"    ✓ Correctly raised error: {e}")
            else:
                print(f"    ✗ Wrong error message: {e}")
                return False
        except Exception as e:
            print(f"    ✗ Wrong exception type: {type(e).__name__}: {e}")
            return False
        
        # Test 3: Verify summary statistics
        print("  Test 3: Verify summary statistics...")
        with open(os.path.join(tmpdir, 'kfold_summary.json'), 'r') as f:
            summary = json.load(f)
        
        if summary['kfold'] != 3:
            print(f"    ✗ Wrong kfold value: {summary['kfold']}")
            return False
        
        # Check that averages were computed
        avg_accuracy = sum(0.85 + i * 0.01 for i in range(3)) / 3
        if abs(summary['averages']['accuracy'] - avg_accuracy) > 0.001:
            print(f"    ✗ Wrong average accuracy: {summary['averages']['accuracy']} vs {avg_accuracy}")
            return False
        
        # Check that std dev exists
        if 'accuracy_std' not in summary['averages']:
            print("    ✗ Standard deviation not calculated")
            return False
        
        print("    ✓ Summary statistics correct")
        
        print("\n✓ All k-fold validation tests passed!")
        return True
        
    finally:
        # Cleanup
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    success = test_kfold_validation()
    sys.exit(0 if success else 1)
