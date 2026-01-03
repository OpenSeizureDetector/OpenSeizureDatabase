#!/usr/bin/env python3
"""
Demonstration of k-fold testing validation in nnTester.py

This script creates a mock directory structure to demonstrate:
1. Successful k-fold testing when fold count matches
2. Error when fold count doesn't match
"""

import os
import json
import tempfile
import shutil

def create_mock_fold_structure(base_dir, num_folds):
    """Create mock fold directories with test results"""
    for i in range(num_folds):
        fold_dir = os.path.join(base_dir, f"fold{i}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Create mock test results
        results = {
            'num_positive_epoch': 100,
            'num_positive_event': 10,
            'accuracy': 0.85 + i * 0.01,
            'accuracyOsd': 0.75,
            'tpr': 0.80 + i * 0.02,
            'fpr': 0.10 - i * 0.01,
            'tprOsd': 0.70,
            'fprOsd': 0.15,
            'tn': 90, 'fp': 10, 'fn': 20, 'tp': 80,
            'tnOsd': 85, 'fpOsd': 15, 'fnOsd': 30, 'tpOsd': 70,
            'event_tpr': 0.82 + i * 0.01,
            'event_fpr': 0.12 - i * 0.01,
            'event_tp': 8, 'event_fp': 1, 'event_fn': 2, 'event_tn': 9,
            'osd_event_tpr': 0.72,
            'osd_event_fpr': 0.18,
            'osd_event_tp': 7, 'osd_event_fp': 2, 'osd_event_fn': 3, 'osd_event_tn': 8
        }
        
        results_path = os.path.join(fold_dir, 'testResults.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Created {results_path}")
    
    return base_dir

def main():
    print("=== K-Fold Testing Validation Demo ===\n")
    
    # Create temporary directory
    tmpdir = tempfile.mkdtemp(prefix='kfold_test_')
    print(f"Created temporary directory: {tmpdir}\n")
    
    try:
        # Create 3 folds
        num_folds = 3
        print(f"Creating {num_folds} mock fold directories...")
        create_mock_fold_structure(tmpdir, num_folds)
        
        print(f"\nDirectory structure:")
        for i in range(num_folds):
            fold_dir = os.path.join(tmpdir, f"fold{i}")
            print(f"  {fold_dir}")
            print(f"    testResults.json")
        
        print("\n" + "="*70)
        print("VALIDATION BEHAVIOR:")
        print("="*70)
        
        print(f"\n✓ If you run with --kfold={num_folds}:")
        print(f"  python nnTester.py --kfold={num_folds}")
        print("  → Will succeed and aggregate results from all 3 folds")
        print("  → Creates kfold_summary.txt and kfold_summary.json")
        
        print(f"\n✗ If you run with --kfold={num_folds + 2} (wrong count):")
        print(f"  python nnTester.py --kfold={num_folds + 2}")
        print(f"  → Will raise ValueError: 'Fold directory not found: ...fold{num_folds}'")
        print(f"  → Expected {num_folds + 2} folds but fold{num_folds} does not exist")
        
        print(f"\n✓ With --rerun flag:")
        print(f"  python nnTester.py --kfold={num_folds} --rerun")
        print("  → Re-runs testModel() for each fold even if results exist")
        print("  → Useful when you've modified test data or want fresh results")
        
        print("\n" + "="*70)
        print("EXPECTED OUTPUT FILES:")
        print("="*70)
        print("  kfold_summary.txt  - Human-readable with means ± std dev")
        print("  kfold_summary.json - Complete data including individual fold results")
        
        print("\n" + "="*70)
        print("SUMMARY INCLUDES:")
        print("="*70)
        print("  • Epoch-based metrics: accuracy, TPR, FPR (model & OSD)")
        print("  • Event-based metrics: TPR, FPR (model & OSD)")
        print("  • Mean and standard deviation for all metrics")
        print("  • Individual fold results for detailed analysis")
        
        print(f"\n(Mock data created in: {tmpdir})")
        print("(Directory will be cleaned up on exit)\n")
        
    finally:
        # Cleanup
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temporary directory")

if __name__ == '__main__':
    main()
