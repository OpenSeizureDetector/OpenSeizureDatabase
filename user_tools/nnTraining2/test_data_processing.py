#!/usr/bin/env python3
"""
Test script to validate data processing pipeline integrity.

Tests:
1. Event integrity: No event is split between train and test sets
2. Window/stride calculation: Windows are created correctly
3. Data integrity: All source data is accounted for in processed windows
4. Acceleration magnitude: Calculations are correct

Usage:
    python test_data_processing.py --config nnConfig_test.json
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from libosd.configUtils import loadConfig


class DataProcessingValidator:
    """Validates data processing pipeline integrity."""
    
    def __init__(self, config_path):
        """Initialize with config file."""
        self.config = loadConfig(config_path)
        self.config_path = config_path
        self.errors = []
        self.warnings = []
        
    def log_error(self, msg):
        """Log an error."""
        self.errors.append(msg)
        print(f"❌ ERROR: {msg}")
        
    def log_warning(self, msg):
        """Log a warning."""
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")
        
    def log_success(self, msg):
        """Log a success."""
        print(f"✅ {msg}")
        
    def load_json_events(self):
        """Load original event data from JSON files."""
        print("\n" + "="*80)
        print("LOADING ORIGINAL EVENT DATA")
        print("="*80)
        
        osdb_files = self.config['osdbConfig']['osdbFiles']
        all_events = []
        
        for osdb_file in osdb_files:
            file_path = Path(self.config['osdbConfig'].get('_cacheDir', '.')) / osdb_file
            
            if not file_path.exists():
                self.log_error(f"OSDB file not found: {file_path}")
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Data can be a list directly or wrapped in {'events': [...]}
                events = data if isinstance(data, list) else data.get('events', [])
                all_events.extend(events)
                print(f"Loaded {len(events)} events from {osdb_file}")
        
        self.original_events = {event['id']: event for event in all_events}
        self.log_success(f"Total events loaded: {len(self.original_events)}")
        
        return self.original_events
    
    def load_processed_data(self):
        """Load processed train and test CSV files."""
        print("\n" + "="*80)
        print("LOADING PROCESSED DATA")
        print("="*80)
        
        # Get file names from config
        train_file = self.config['dataFileNames']['trainDataFileCsv']
        test_file = self.config['dataFileNames']['testDataFileCsv']
        model_fname = self.config['modelConfig']['modelFname']
        
        # Look for files in the output folder (most recent run)
        output_base = Path('./output') / model_fname
        if output_base.exists():
            # Find the most recent output folder
            folders = sorted([f for f in output_base.iterdir() if f.is_dir()], 
                           key=lambda x: int(x.name) if x.name.isdigit() else 0)
            if folders:
                latest_folder = folders[-1]
                print(f"Using latest output folder: {latest_folder}")
                train_file = latest_folder / train_file
                test_file = latest_folder / test_file
        
        files_to_load = {
            'train': train_file,
            'test': test_file
        }
        
        loaded_data = {}
        
        for data_type, filename in files_to_load.items():
            file_path = Path(filename)
            
            if not file_path.exists():
                self.log_warning(f"{data_type.upper()} file not found: {file_path}")
                loaded_data[data_type] = pd.DataFrame()
                continue
            
            df = pd.read_csv(file_path)
            loaded_data[data_type] = df
            print(f"Loaded {data_type}: {len(df)} rows, columns: {list(df.columns)}")
        
        self.train_df = loaded_data['train']
        self.test_df = loaded_data['test']
        
        return loaded_data
    
    def test_event_integrity(self):
        """Test 1: Verify no event is split between train and test sets."""
        print("\n" + "="*80)
        print("TEST 1: EVENT INTEGRITY (No event split between train/test)")
        print("="*80)
        
        if self.train_df.empty or self.test_df.empty:
            self.log_warning("Train or test data is empty, skipping event integrity test")
            return False
        
        # Get event IDs from each dataset
        train_events = set(self.train_df['eventId'].unique()) if 'eventId' in self.train_df.columns else set()
        test_events = set(self.test_df['eventId'].unique()) if 'eventId' in self.test_df.columns else set()
        
        print(f"\nTrain set contains {len(train_events)} unique events")
        print(f"Test set contains {len(test_events)} unique events")
        
        # Check for overlap
        overlap = train_events.intersection(test_events)
        
        if overlap:
            self.log_error(f"Found {len(overlap)} events in BOTH train and test sets!")
            print(f"Overlapping event IDs: {sorted(overlap)}")
            return False
        else:
            self.log_success(f"No event overlap - all {len(train_events) + len(test_events)} events are properly separated")
            
        # Verify all original events are accounted for
        all_processed_events = train_events.union(test_events)
        original_event_ids = set(self.original_events.keys())
        
        missing = original_event_ids - all_processed_events
        extra = all_processed_events - original_event_ids
        
        if missing:
            self.log_warning(f"{len(missing)} events from original data not found in processed data: {sorted(missing)}")
        
        if extra:
            self.log_warning(f"{len(extra)} events in processed data not found in original: {sorted(extra)}")
        
        if not missing and not extra:
            self.log_success("All original events accounted for in processed data")
        
        return len(overlap) == 0
    
    def test_window_stride_calculation(self):
        """Test 2: Verify window and stride calculations are correct."""
        print("\n" + "="*80)
        print("TEST 2: WINDOW/STRIDE CALCULATION")
        print("="*80)
        
        window_size = self.config['dataProcessing']['window']
        step_size = self.config['dataProcessing']['step']
        
        print(f"\nConfiguration:")
        print(f"  Window size: {window_size} samples")
        print(f"  Step size: {step_size} samples")
        
        if self.train_df.empty:
            self.log_warning("Train data is empty, skipping window/stride test")
            return False
        
        # Get all acceleration columns (should contain window_size values each)
        acc_cols = [col for col in self.train_df.columns if col.startswith('acc_')]
        
        if not acc_cols:
            self.log_error("No acceleration columns found in processed data")
            return False
        
        # Check that we have the expected number of acceleration columns
        expected_cols = window_size
        actual_cols = len(acc_cols)
        
        print(f"\nAcceleration columns:")
        print(f"  Expected: {expected_cols} columns (window size)")
        print(f"  Found: {actual_cols} columns")
        
        if actual_cols != expected_cols:
            self.log_error(f"Column count mismatch! Expected {expected_cols}, found {actual_cols}")
            return False
        
        self.log_success(f"Window size correct: {actual_cols} columns match configuration")
        
        # For each event in train data, verify stride is applied correctly
        test_passed = True
        for event_id in self.train_df['eventId'].unique()[:3]:  # Test first 3 events
            event_windows = self.train_df[self.train_df['eventId'] == event_id]
            num_windows = len(event_windows)
            
            if event_id in self.original_events:
                original_event = self.original_events[event_id]
                num_datapoints = len(original_event['datapoints'])
                
                # Calculate expected number of windows
                # Formula: (total_points - window_size) / step_size + 1
                expected_windows = max(0, (num_datapoints - window_size) // step_size + 1)
                
                print(f"\nEvent {event_id}:")
                print(f"  Original datapoints: {num_datapoints}")
                print(f"  Expected windows: {expected_windows}")
                print(f"  Actual windows: {num_windows}")
                
                if num_windows != expected_windows:
                    self.log_error(f"Window count mismatch for event {event_id}")
                    test_passed = False
                else:
                    self.log_success(f"Window count correct for event {event_id}")
        
        return test_passed
    
    def test_acceleration_magnitude(self):
        """Test 3: Verify acceleration magnitude calculations are correct."""
        print("\n" + "="*80)
        print("TEST 3: ACCELERATION MAGNITUDE CALCULATION")
        print("="*80)
        
        if self.train_df.empty:
            self.log_warning("Train data is empty, skipping magnitude test")
            return False
        
        # Pick first event and first window to verify
        first_event_id = self.train_df['eventId'].iloc[0]
        first_window = self.train_df[self.train_df['eventId'] == first_event_id].iloc[0]
        
        print(f"\nTesting event: {first_event_id}")
        
        if first_event_id not in self.original_events:
            self.log_error(f"Event {first_event_id} not found in original data")
            return False
        
        original_event = self.original_events[first_event_id]
        window_size = self.config['dataProcessing']['window']
        
        # Get acceleration columns from processed data
        acc_cols = [col for col in self.train_df.columns if col.startswith('acc_')]
        processed_magnitudes = first_window[acc_cols].values
        
        # Calculate expected magnitudes from original data
        expected_magnitudes = []
        for i in range(window_size):
            if i < len(original_event['datapoints']):
                dp = original_event['datapoints'][i]
                accX = dp.get('accX', 0)
                accY = dp.get('accY', 0)
                accZ = dp.get('accZ', 0)
                magnitude = np.sqrt(accX**2 + accY**2 + accZ**2)
                expected_magnitudes.append(magnitude)
        
        expected_magnitudes = np.array(expected_magnitudes)
        
        print(f"\nFirst window comparison:")
        print(f"  Expected magnitudes: {expected_magnitudes[:5]}... (showing first 5)")
        print(f"  Processed magnitudes: {processed_magnitudes[:5]}... (showing first 5)")
        
        # Check if they match (within floating point tolerance)
        if len(processed_magnitudes) >= len(expected_magnitudes):
            if np.allclose(processed_magnitudes[:len(expected_magnitudes)], expected_magnitudes, rtol=1e-5):
                self.log_success("Acceleration magnitudes match original data")
                return True
            else:
                self.log_error("Acceleration magnitude mismatch!")
                max_diff = np.max(np.abs(processed_magnitudes[:len(expected_magnitudes)] - expected_magnitudes))
                print(f"  Maximum difference: {max_diff}")
                return False
        else:
            self.log_error(f"Processed data has fewer values than expected")
            return False
    
    def test_data_completeness(self):
        """Test 4: Verify all data from original events appears in processed windows."""
        print("\n" + "="*80)
        print("TEST 4: DATA COMPLETENESS")
        print("="*80)
        
        if self.train_df.empty and self.test_df.empty:
            self.log_error("Both train and test data are empty!")
            return False
        
        combined_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
        
        total_windows = len(combined_df)
        unique_events = combined_df['eventId'].nunique()
        
        print(f"\nProcessed data summary:")
        print(f"  Total windows: {total_windows}")
        print(f"  Unique events: {unique_events}")
        print(f"  Original events: {len(self.original_events)}")
        
        # Check label distribution
        if 'label' in combined_df.columns:
            label_counts = combined_df['label'].value_counts()
            print(f"\nLabel distribution:")
            for label, count in label_counts.items():
                print(f"  {label}: {count} windows ({100*count/total_windows:.1f}%)")
        
        self.log_success(f"Data completeness check passed: {total_windows} total windows from {unique_events} events")
        return True
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("\n" + "="*80)
        print("OPENSEIZUREDATABASE DATA PROCESSING VALIDATION")
        print("="*80)
        print(f"Config file: {self.config_path}")
        
        # Load data
        self.load_json_events()
        self.load_processed_data()
        
        # Run tests
        test_results = {
            'Event Integrity': self.test_event_integrity(),
            'Window/Stride Calculation': self.test_window_stride_calculation(),
            'Acceleration Magnitude': self.test_acceleration_magnitude(),
            'Data Completeness': self.test_data_completeness()
        }
        
        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.errors:
            print(f"\n❌ {len(self.errors)} errors:")
            for error in self.errors:
                print(f"   - {error}")
        
        all_passed = all(test_results.values()) and len(self.errors) == 0
        
        print("\n" + "="*80)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED - Review errors above")
        print("="*80 + "\n")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate data processing pipeline integrity'
    )
    parser.add_argument(
        '--config',
        default='nnConfig_test.json',
        help='Configuration file path (default: nnConfig_test.json)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    validator = DataProcessingValidator(args.config)
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
