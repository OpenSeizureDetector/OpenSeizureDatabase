#!/usr/bin/env python3
"""
Update test data with new events not used in original training.

This module creates a new test dataset (testDataNew.json) containing all events
from the database that:
  1. Meet the filtering criteria from the configuration file
  2. Were NOT used in the original training dataset (neither train nor test splits)

This allows testing a trained model on new data while ensuring no data leakage
from the original training set.

Usage:
    updateTestData(configObj, rerunNum, outDir, dbDirOverride=None, debug=False)

Arguments:
    configObj: Configuration object containing filter criteria
    rerunNum: Original training run number (e.g., 24) to identify training data
    outDir: Output directory where testDataNew.json will be saved
    dbDirOverride: Optional override for the database directory (dbDir parameter)
    debug: Enable debug output
"""

import argparse
import sys
import os
import json
import csv
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libosd.osdDbConnection
import libosd.configUtils


def load_event_ids_from_json(json_file_path):
    """
    Load event IDs from a JSON file.
    
    Args:
        json_file_path: Path to JSON file (either testData.json or trainData.json format)
    
    Returns:
        Set of event IDs that were in the file
    """
    event_ids = set()
    
    if not os.path.exists(json_file_path):
        return event_ids
    
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Handle both direct list and dict with 'events' key
        events_list = data if isinstance(data, list) else data.get('events', [])
        
        for event in events_list:
            # Events should have an 'id' field
            event_id = event.get('id')
            if event_id is not None:
                event_ids.add(event_id)
    
    except Exception as e:
        print(f"WARNING: Could not load events from {json_file_path}: {e}")
    
    return event_ids


def load_event_ids_from_csv(csv_file_path):
    """
    Load event IDs from a CSV file (flattened data format).
    
    Args:
        csv_file_path: Path to CSV file (either testData.csv or trainData.csv)
    
    Returns:
        Set of event IDs that were in the file
    """
    event_ids = set()
    
    if not os.path.exists(csv_file_path):
        return event_ids
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Try multiple possible column names for event ID
        id_column = None
        for col_name in ['eventId', 'event_id', 'id', 'ID', 'EventId']:
            if col_name in df.columns:
                id_column = col_name
                break
        
        if id_column is None:
            print(f"WARNING: Could not find event ID column in {csv_file_path}")
            print(f"         Available columns: {list(df.columns[:10])}")
            return event_ids
        
        # Extract event IDs from the dataframe
        for event_id in df[id_column].unique():
            if pd.notna(event_id):  # Skip NaN values
                event_ids.add(int(event_id) if isinstance(event_id, (int, float)) else event_id)
    
    except Exception as e:
        print(f"WARNING: Could not load events from {csv_file_path}: {e}")
    
    return event_ids


def load_event_ids_from_file(file_path):
    """
    Load event IDs from either JSON or CSV file format.
    
    Tries JSON first, then falls back to CSV if JSON doesn't exist or fails.
    
    Args:
        file_path: Base path to file without extension (will try .json then .csv)
    
    Returns:
        Set of event IDs that were in the file
    """
    event_ids = set()
    
    # Try JSON first
    json_path = file_path if file_path.endswith('.json') else file_path + '.json'
    if os.path.exists(json_path):
        event_ids = load_event_ids_from_json(json_path)
        if event_ids:
            return event_ids
    
    # Fall back to CSV
    csv_path = file_path if file_path.endswith('.csv') else file_path + '.csv'
    if os.path.exists(csv_path):
        event_ids = load_event_ids_from_csv(csv_path)
        if event_ids:
            return event_ids
    
    return event_ids


def get_training_event_ids(config_obj, rerun_num, out_base_dir, debug=False):
    """
    Get IDs of all events used in the original training run.
    
    This includes only trainData from the original training run.
    testData from the original run should remain in testDataNew and is used as
    the baseline dataset to which new events are added.
    
    Supports both JSON and CSV file formats (new format uses CSV after flattening).
    
    Args:
        config_obj: Configuration object
        rerun_num: Original training run number (e.g., 24)
        out_base_dir: Base output directory (e.g., './output')
        debug: Enable debug output
    
    Returns:
        Set of event IDs that were used in training (train only)
    """
    training_event_ids = set()
    
    # Construct path to original training run output
    model_fname = config_obj['modelConfig'].get('modelFname', 'model')
    original_run_path = os.path.join(out_base_dir, model_fname, str(rerun_num))
    
    if not os.path.exists(original_run_path):
        print(f"WARNING: Original training run folder not found: {original_run_path}")
        print(f"         No events will be excluded from the original training set.")
        return training_event_ids
    
    if debug:
        print(f"updateTestData: Loading original training run from: {original_run_path}")
    
    # Load trainData (try both JSON and CSV formats)
    train_data_base_name = config_obj['dataFileNames'].get('trainDataFileJson', 'trainData').replace('.json', '')
    train_data_path = os.path.join(original_run_path, train_data_base_name)
    train_ids = load_event_ids_from_file(train_data_path)
    if debug:
        print(f"updateTestData: Found {len(train_ids)} events in original trainData")
    training_event_ids.update(train_ids)
    
    if debug:
        print(f"updateTestData: Total events to exclude from training: {len(training_event_ids)}")
    
    return training_event_ids


def get_original_test_event_ids(config_obj, rerun_num, out_base_dir, debug=False):
    """
    Get event IDs from original testData in the specified rerun output folder.

    Args:
        config_obj: Configuration object
        rerun_num: Original training run number (e.g., 24)
        out_base_dir: Base output directory (e.g., './output')
        debug: Enable debug output

    Returns:
        Set of event IDs from original testData
    """
    model_fname = config_obj['modelConfig'].get('modelFname', 'model')
    original_run_path = os.path.join(out_base_dir, model_fname, str(rerun_num))

    if not os.path.exists(original_run_path):
        if debug:
            print(f"updateTestData: Original run path missing for testData lookup: {original_run_path}")
        return set()

    test_data_base_name = config_obj['dataFileNames'].get('testDataFileJson', 'testData').replace('.json', '')
    test_data_path = os.path.join(original_run_path, test_data_base_name)
    test_ids = load_event_ids_from_file(test_data_path)

    if debug:
        print(f"updateTestData: Found {len(test_ids)} events in original testData")

    return test_ids


def _is_seizure_event(event_type_str):
    if event_type_str is None:
        return False
    return str(event_type_str).strip().lower() == 'seizure'


def _count_ids_by_type(event_ids, event_type_map):
    """Count seizure/non-seizure events for a set of event IDs."""
    seizure_count = 0
    nonseizure_count = 0
    unknown_count = 0

    for event_id in event_ids:
        event_type = event_type_map.get(event_id)
        if event_type is None:
            unknown_count += 1
        elif _is_seizure_event(event_type):
            seizure_count += 1
        else:
            nonseizure_count += 1

    return {
        'seizure': seizure_count,
        'non_seizure': nonseizure_count,
        'unknown': unknown_count,
        'total': len(event_ids)
    }


def update_test_data(config_obj, rerun_num, out_dir=".", db_dir_override=None, debug=False):
    """
    Create a new test dataset with events not used in original training.
    
    Args:
        config_obj: Configuration object containing filter criteria
        rerun_num: Original training run number to identify training data to exclude
        out_dir: Output directory where testDataNew.json will be written
        db_dir_override: Optional override for the database directory (cacheDir)
        debug: Enable debug output
    
    Returns:
        Dictionary with summary information:
        {
            'total_in_db': Number of events in database after filtering,
            'in_training': Number of events in original training set,
            'in_test_new': Number of events in testDataNew,
            'test_data_path': Path to testDataNew.json file
        }
    """
    print("updateTestData: Starting update-test-data operation")
    
    if debug:
        print(f"updateTestData: config keys: {config_obj.keys()}")
    
    # Validate inputs
    if rerun_num <= 0:
        print("ERROR: --rerun must be > 0 to identify the original training run")
        return None
    
    # Get invalid events list if available
    if "invalidEvents" in config_obj['osdbConfig']:
        print("updateTestData: Using invalid events from configObj")
        invalid_events = config_obj['osdbConfig']['invalidEvents']
    else:
        invalid_events = None
    
    # Determine database directory
    if db_dir_override:
        print(f"updateTestData: Using database directory override: {db_dir_override}")
        db_dir = db_dir_override
    elif "cacheDir" in config_obj['osdbConfig']:
        print(f"updateTestData: Using cache directory from configObj: {config_obj['osdbConfig']['cacheDir']}")
        db_dir = config_obj['osdbConfig']['cacheDir']
    else:
        print("updateTestData: Using default cache directory")
        db_dir = None
    
    # Load all available data from database with filtering
    print(f"updateTestData: Loading all data from database (dbDir: {db_dir})")
    osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=db_dir, debug=debug)
    
    for fname in config_obj['osdbConfig']['osdbFiles']:
        if debug:
            print(f"updateTestData: Loading OSDB File: {fname}")
        events_obj_len = osd.loadDbFile(fname)
        print(f"updateTestData: Loaded {events_obj_len} events from {fname}")
    
    # Remove invalid events if specified
    event_ids_list = osd.getEventIds()
    print(f"updateTestData: Total of {len(event_ids_list)} events loaded from database")
    
    if invalid_events is not None:
        print("updateTestData: Removing invalid events...")
        osd.removeEvents(invalid_events)
        event_ids_list = osd.getEventIds()
        print(f"updateTestData: {len(event_ids_list)} events remaining after removing invalid events")
    
    # Apply filtering criteria from configuration
    filter_cfg = config_obj['eventFilters']
    print("updateTestData: Applying event filters from configuration")
    
    event_ids_list = osd.getFilteredEventsLst(
        includeUserIds=filter_cfg['includeUserIds'],
        excludeUserIds=filter_cfg['excludeUserIds'],
        includeTypes=filter_cfg['includeTypes'],
        excludeTypes=filter_cfg['excludeTypes'],
        includeSubTypes=filter_cfg['includeSubTypes'],
        excludeSubTypes=filter_cfg['excludeSubTypes'],
        includeDataSources=filter_cfg['includeDataSources'],
        excludeDataSources=filter_cfg['excludeDataSources'],
        includeText=filter_cfg['includeText'],
        excludeText=filter_cfg['excludeText'],
        require3dData=filter_cfg['require3dData'],
        requireHrData=filter_cfg['requireHrData'],
        requireO2SatData=filter_cfg['requireO2SatData'],
        debug=debug
    )
    
    total_after_filtering = len(event_ids_list)
    print(f"updateTestData: {total_after_filtering} events remaining after applying filters")
    
    # Get the event IDs that were used in the original training
    out_base_dir = config_obj['osdbConfig'].get('outDir', './output')
    training_event_ids = get_training_event_ids(config_obj, rerun_num, out_base_dir, debug=debug)
    original_test_event_ids = get_original_test_event_ids(config_obj, rerun_num, out_base_dir, debug=debug)
    
    # Create testDataNew by excluding training events
    test_data_new_ids = [eid for eid in event_ids_list if eid not in training_event_ids]
    
    print(f"updateTestData: Events in original training set: {len(training_event_ids)}")
    print(f"updateTestData: Events in testDataNew (filtered & not in training): {len(test_data_new_ids)}")

    # Build event type map from currently filtered database events for summary stats
    filtered_events_metadata = osd.getEvents(event_ids_list, includeDatapoints=False)
    event_type_map = {event.get('id'): event.get('type') for event in filtered_events_metadata if event.get('id') is not None}

    # Compare original testData to newly generated testDataNew
    test_data_new_id_set = set(test_data_new_ids)
    added_ids = test_data_new_id_set - original_test_event_ids
    removed_ids = original_test_event_ids - test_data_new_id_set
    train_test_overlap_ids = test_data_new_id_set.intersection(training_event_ids)

    added_counts = _count_ids_by_type(added_ids, event_type_map)
    removed_counts = _count_ids_by_type(removed_ids, event_type_map)

    print("\nupdateTestData: Comparison vs original testData")
    print(f"  Original testData events: {len(original_test_event_ids)}")
    print(f"  New testDataNew events: {len(test_data_new_id_set)}")
    print(
        "  Added events: total={total}, seizure={seizure}, non-seizure={non_seizure}, unknown={unknown}".format(
            **added_counts
        )
    )
    print(
        "  Removed events: total={total}, seizure={seizure}, non-seizure={non_seizure}, unknown={unknown}".format(
            **removed_counts
        )
    )
    print(f"  Train/Test overlap check (must be 0): {len(train_test_overlap_ids)}")
    if len(train_test_overlap_ids) > 0:
        sample_ids = sorted(list(train_test_overlap_ids))[:10]
        print(f"ERROR: Data leakage detected. Example overlapping event IDs: {sample_ids}")
        return None
    
    # Load the full event details for testDataNew
    if len(test_data_new_ids) > 0:
        test_data_new_events = osd.getEvents(test_data_new_ids, includeDatapoints=True)
    else:
        test_data_new_events = []
    
    # Save testDataNew.json
    test_data_new_json_fname = config_obj['dataFileNames'].get('testDataNewFileJson', 'testDataNew.json')
    test_data_new_path = os.path.join(out_dir, test_data_new_json_fname)
    
    print(f"updateTestData: Saving testDataNew to {test_data_new_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Save as JSON (use same format as allData.json)
    try:
        with open(test_data_new_path, 'w') as f:
            json.dump(test_data_new_events, f, indent=2)
        print(f"updateTestData: testDataNew.json saved ({len(test_data_new_events)} events)")
    except Exception as e:
        print(f"ERROR: Failed to save testDataNew.json: {e}")
        return None
    
    # Return summary information
    return {
        'total_in_db': total_after_filtering,
        'in_training': len(training_event_ids),
        'in_original_test': len(original_test_event_ids),
        'in_test_new': len(test_data_new_ids),
        'added_total': added_counts['total'],
        'added_seizure': added_counts['seizure'],
        'added_non_seizure': added_counts['non_seizure'],
        'removed_total': removed_counts['total'],
        'removed_seizure': removed_counts['seizure'],
        'removed_non_seizure': removed_counts['non_seizure'],
        'train_test_overlap': len(train_test_overlap_ids),
        'test_data_path': test_data_new_path,
        'test_json_path': test_data_new_path
    }


def main():
    """Command-line interface for updateTestData."""
    print("updateTestData.main()")
    parser = argparse.ArgumentParser(description='Create new test data excluding training data')
    parser.add_argument('--config', default="nnConfig.json",
                        help='name of json file containing configuration')
    parser.add_argument('--rerun', type=int, required=True,
                        help='Original training run number to identify which data was used for training')
    parser.add_argument('--outDir', default=".",
                        help='directory to write testDataNew.json')
    parser.add_argument('--dbDir', default=None,
                        help='Override the database directory (cacheDir) from config')
    parser.add_argument('--debug', action="store_true",
                        help='Write debugging information to screen')
    
    args_namespace = parser.parse_args()
    args = vars(args_namespace)
    
    if args['debug']:
        print(f"Arguments: {args}")
    
    # Load configuration
    config_obj = libosd.configUtils.loadConfig(args['config'])
    
    # Load separate OSDB config if included
    if "osdbCfg" in config_obj:
        osdb_cfg_fname = libosd.configUtils.getConfigParam("osdbCfg", config_obj)
        print(f"updateTestData: Loading separate OSDB Configuration File: {osdb_cfg_fname}")
        osdb_cfg_obj = libosd.configUtils.loadConfig(osdb_cfg_fname)
        config_obj = config_obj | osdb_cfg_obj
    
    # Run the update
    result = update_test_data(
        config_obj,
        args['rerun'],
        out_dir=args['outDir'],
        db_dir_override=args['dbDir'],
        debug=args['debug']
    )
    
    if result:
        print("\nupdateTestData: Summary")
        print(f"  Total events in database (after filtering): {result['total_in_db']}")
        print(f"  Events in original training set: {result['in_training']}")
        print(f"  Events in original testData: {result['in_original_test']}")
        print(f"  Events in testDataNew: {result['in_test_new']}")
        print(f"  Added events (seizure/non-seizure): {result['added_seizure']}/{result['added_non_seizure']}")
        print(f"  Removed events (seizure/non-seizure): {result['removed_seizure']}/{result['removed_non_seizure']}")
        print(f"  testDataNew saved to: {result['test_data_path']}")
    else:
        print("updateTestData: FAILED")


if __name__ == "__main__":
    main()
