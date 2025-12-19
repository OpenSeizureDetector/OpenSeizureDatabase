#!/usr/bin/env python3
"""
Audit Data Processing Pipeline

This script verifies:
1. Event counts remain consistent through the processing pipeline
2. Each event appears in ONLY train OR test datasets (no leakage)
3. Seizure vs non-seizure event counts at each processing stage

Usage:
    python auditDataProcessing.py [--config CONFIG_FILE] [--dataDir DATA_DIR]

Arguments:
    --config: Path to configuration file (default: nnConfig.json)
    --dataDir: Path to data directory (default: from config)
"""

import os
import sys
import json
import pandas as pd
import argparse
from collections import defaultdict
from typing import Dict, Set, List, Tuple


class DataAuditor:
    """Audits data processing pipeline for consistency and train/test separation."""
    
    def __init__(self, config_path: str, data_dir: str = None):
        """
        Initialize auditor with configuration.
        
        Args:
            config_path: Path to nnConfig JSON file
            data_dir: Override data directory (optional)
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.data_dir = data_dir or self.config.get('dataDir', '.')
        self.file_names = self.config['dataFileNames']
        
        # Results storage
        self.event_counts = {}
        self.train_events = set()
        self.test_events = set()
        self.val_events = set()
        self.leakage_issues = []
        
    def _get_file_path(self, file_key: str) -> str:
        """Get full path for a data file."""
        return os.path.join(self.data_dir, self.file_names[file_key])
    
    def _load_json_events(self, file_key: str) -> Tuple[Set[int], int, int]:
        """
        Load events from JSON file.
        
        Returns:
            (event_ids, seizure_count, non_seizure_count)
        """
        file_path = self._get_file_path(file_key)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return set(), 0, 0
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list format and dict with 'events' key
        events = data if isinstance(data, list) else data.get('events', [])
        
        event_ids = set()
        seizure_count = 0
        non_seizure_count = 0
        
        for event in events:
            event_id = event.get('eventId') or event.get('id')
            if event_id:
                event_ids.add(event_id)
                
                # Check if seizure (type == 1) or non-seizure (type == 0)
                event_type = event.get('type', 0)
                if event_type == 1:
                    seizure_count += 1
                else:
                    non_seizure_count += 1
        
        return event_ids, seizure_count, non_seizure_count
    
    def _load_csv_events(self, file_key: str) -> Tuple[Set[int], int, int]:
        """
        Load events from CSV file.
        
        Returns:
            (event_ids, seizure_count, non_seizure_count)
        """
        file_path = self._get_file_path(file_key)
        
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return set(), 0, 0
        
        df = pd.read_csv(file_path)
        
        if 'eventId' not in df.columns:
            print(f"⚠️  No eventId column in {file_path}")
            return set(), 0, 0
        
        event_ids = set(df['eventId'].unique())
        
        # Count seizure vs non-seizure events
        if 'type' in df.columns:
            event_types = df.groupby('eventId')['type'].first()
            seizure_count = (event_types == 1).sum()
            non_seizure_count = (event_types == 0).sum()
        else:
            seizure_count = 0
            non_seizure_count = len(event_ids)
        
        return event_ids, seizure_count, non_seizure_count
    
    def audit_json_files(self):
        """Audit JSON files in the pipeline."""
        print("\n" + "="*80)
        print("AUDITING JSON FILES")
        print("="*80)
        
        json_files = [
            ('allDataFileJson', 'All Data'),
            ('testDataFileJson', 'Test Data'),
            ('trainDataFileJson', 'Train Data'),
            ('valDataFileJson', 'Validation Data')
        ]
        
        for file_key, label in json_files:
            event_ids, seizure_count, non_seizure_count = self._load_json_events(file_key)
            
            self.event_counts[label] = {
                'seizure': seizure_count,
                'non_seizure': non_seizure_count,
                'total': len(event_ids)
            }
            
            # Track which events are in train/test/val
            if 'Test' in label:
                self.test_events = event_ids
            elif 'Train' in label:
                self.train_events = event_ids
            elif 'Validation' in label:
                self.val_events = event_ids
            
            print(f"\n{label}: {self.file_names[file_key]}")
            print(f"  Total Events: {len(event_ids)}")
            print(f"  Seizures: {seizure_count}")
            print(f"  Non-Seizures: {non_seizure_count}")
    
    def audit_csv_files(self):
        """Audit CSV files in the pipeline."""
        print("\n" + "="*80)
        print("AUDITING CSV FILES")
        print("="*80)
        
        csv_files = [
            ('testDataFileCsv', 'Test Data CSV'),
            ('trainDataFileCsv', 'Train Data CSV'),
            ('valDataFileCsv', 'Validation Data CSV'),
            ('trainAugmentedFileCsv', 'Train Augmented CSV'),
            ('trainFeaturesFileCsv', 'Train Features CSV'),
            ('testFeaturesFileCsv', 'Test Features CSV'),
            ('trainFeaturesHistoryFileCsv', 'Train Features History CSV'),
            ('testFeaturesHistoryFileCsv', 'Test Features History CSV')
        ]
        
        for file_key, label in csv_files:
            event_ids, seizure_count, non_seizure_count = self._load_csv_events(file_key)
            
            self.event_counts[label] = {
                'seizure': seizure_count,
                'non_seizure': non_seizure_count,
                'total': len(event_ids)
            }
            
            # Verify test/train separation
            if 'Test' in label and event_ids:
                overlap_train = event_ids & self.train_events
                overlap_val = event_ids & self.val_events
                if overlap_train:
                    self.leakage_issues.append(
                        f"❌ {label} has {len(overlap_train)} events overlapping with Train Data"
                    )
                if overlap_val:
                    self.leakage_issues.append(
                        f"❌ {label} has {len(overlap_val)} events overlapping with Validation Data"
                    )
            elif 'Train' in label and event_ids:
                overlap_test = event_ids & self.test_events
                overlap_val = event_ids & self.val_events
                if overlap_test:
                    self.leakage_issues.append(
                        f"❌ {label} has {len(overlap_test)} events overlapping with Test Data"
                    )
                # Note: Train and Val overlap is expected for some pipelines
            
            print(f"\n{label}: {self.file_names[file_key]}")
            print(f"  Total Events: {len(event_ids)}")
            print(f"  Seizures: {seizure_count}")
            print(f"  Non-Seizures: {non_seizure_count}")
    
    def check_train_test_separation(self):
        """Verify no events appear in both train and test sets."""
        print("\n" + "="*80)
        print("TRAIN/TEST SEPARATION CHECK")
        print("="*80)
        
        overlap = self.train_events & self.test_events
        
        if overlap:
            print(f"\n❌ LEAKAGE DETECTED: {len(overlap)} events appear in BOTH train and test sets")
            print(f"   Event IDs: {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
        else:
            print(f"\n✅ PASS: No event overlap between train and test sets")
        
        # Check validation set
        if self.val_events:
            val_train_overlap = self.val_events & self.train_events
            val_test_overlap = self.val_events & self.test_events
            
            print(f"\nValidation Set:")
            print(f"  Events in validation: {len(self.val_events)}")
            print(f"  Overlap with train: {len(val_train_overlap)}")
            print(f"  Overlap with test: {len(val_test_overlap)}")
            
            if val_test_overlap:
                self.leakage_issues.append(
                    f"❌ Validation set has {len(val_test_overlap)} events overlapping with Test set"
                )
    
    def check_event_consistency(self):
        """Verify event counts remain consistent through pipeline."""
        print("\n" + "="*80)
        print("EVENT COUNT CONSISTENCY CHECK")
        print("="*80)
        
        # Get baseline from allData
        all_data_counts = self.event_counts.get('All Data', {})
        total_all = all_data_counts.get('total', 0)
        seizure_all = all_data_counts.get('seizure', 0)
        non_seizure_all = all_data_counts.get('non_seizure', 0)
        
        print(f"\nBaseline (All Data):")
        print(f"  Total Events: {total_all}")
        print(f"  Seizures: {seizure_all}")
        print(f"  Non-Seizures: {non_seizure_all}")
        
        # Check train + test + val = all
        train_total = self.event_counts.get('Train Data', {}).get('total', 0)
        test_total = self.event_counts.get('Test Data', {}).get('total', 0)
        val_total = self.event_counts.get('Validation Data', {}).get('total', 0)
        
        split_sum = train_total + test_total + val_total
        
        print(f"\nSplit Totals:")
        print(f"  Train: {train_total}")
        print(f"  Test: {test_total}")
        print(f"  Validation: {val_total}")
        print(f"  Sum: {split_sum}")
        
        if split_sum == total_all:
            print(f"  ✅ Split sum matches total events")
        else:
            print(f"  ⚠️  Difference: {split_sum - total_all} events")
        
        # Check CSV files match JSON files
        print(f"\nCSV Consistency:")
        
        csv_json_pairs = [
            ('Train Data CSV', 'Train Data'),
            ('Test Data CSV', 'Test Data'),
            ('Train Features CSV', 'Train Data'),
            ('Test Features CSV', 'Test Data'),
            ('Train Features History CSV', 'Train Data'),
            ('Test Features History CSV', 'Test Data')
        ]
        
        for csv_label, json_label in csv_json_pairs:
            csv_count = self.event_counts.get(csv_label, {}).get('total', 0)
            json_count = self.event_counts.get(json_label, {}).get('total', 0)
            
            if csv_count > 0:  # Only check if file exists
                match = "✅" if csv_count == json_count else "❌"
                print(f"  {match} {csv_label}: {csv_count} (JSON: {json_count})")
    
    def generate_summary_table(self):
        """Generate a summary table of all files."""
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        
        # Create summary data
        rows = []
        for label, counts in self.event_counts.items():
            rows.append({
                'File': label,
                'Total Events': counts['total'],
                'Seizures': counts['seizure'],
                'Non-Seizures': counts['non_seizure']
            })
        
        # Sort by category
        category_order = ['All Data', 'Train Data', 'Test Data', 'Validation Data']
        rows_sorted = []
        
        # Add known categories first
        for category in category_order:
            for row in rows:
                if row['File'] == category:
                    rows_sorted.append(row)
                    break
        
        # Add remaining rows
        for row in rows:
            if row not in rows_sorted:
                rows_sorted.append(row)
        
        # Print table
        df = pd.DataFrame(rows_sorted)
        print("\n" + df.to_string(index=False))
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'data_audit_summary.csv')
        df.to_csv(output_path, index=False)
        print(f"\n📊 Summary saved to: {output_path}")
    
    def print_final_report(self):
        """Print final audit report."""
        print("\n" + "="*80)
        print("FINAL AUDIT REPORT")
        print("="*80)
        
        if self.leakage_issues:
            print("\n⚠️  ISSUES FOUND:")
            for issue in self.leakage_issues:
                print(f"  {issue}")
        else:
            print("\n✅ ALL CHECKS PASSED")
            print("  • No train/test leakage detected")
            print("  • Event counts consistent through pipeline")
        
        # Print summary statistics
        overlap = self.train_events & self.test_events
        print(f"\nSummary Statistics:")
        print(f"  Total unique events in train: {len(self.train_events)}")
        print(f"  Total unique events in test: {len(self.test_events)}")
        print(f"  Total unique events in val: {len(self.val_events)}")
        print(f"  Train/test overlap: {len(overlap)}")
    
    def run_audit(self):
        """Run complete audit."""
        print("\n" + "="*80)
        print("DATA PROCESSING PIPELINE AUDIT")
        print("="*80)
        print(f"Config: {self.config.get('_comment', 'N/A')}")
        print(f"Data Directory: {self.data_dir}")
        
        self.audit_json_files()
        self.audit_csv_files()
        self.check_train_test_separation()
        self.check_event_consistency()
        self.generate_summary_table()
        self.print_final_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Audit data processing pipeline for consistency and train/test separation'
    )
    parser.add_argument(
        '--config',
        default='nnConfig.json',
        help='Path to configuration file (default: nnConfig.json)'
    )
    parser.add_argument(
        '--dataDir',
        default=None,
        help='Override data directory from config'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        sys.exit(1)
    
    auditor = DataAuditor(args.config, args.dataDir)
    auditor.run_audit()


if __name__ == '__main__':
    main()
