import os
import json
import pandas as pd
import numpy as np

def process_feature_history(input_csv, output_csv, event_col, n_history):
    df = pd.read_csv(input_csv)
    raw_cols = ['x', 'y', 'z', 'magnitude']
    exclude_cols = [event_col, 'eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState',
                   'osdSpecPoewr', 'osdRoiPower', 'startSample', 'endSample']
    feature_cols = [col for col in df.columns if col not in raw_cols and col not in exclude_cols]
    keep_cols = [col for col in df.columns if col not in feature_cols]  # columns to keep as-is
    output_rows = []
    for event_id, group in df.groupby(event_col):
        if len(group) < n_history:
            continue
        group = group.reset_index(drop=True)
        for i in range(n_history-1, len(group)):
            row = {}
            # Copy all non-history columns from the current row
            for col in keep_cols:
                if col in group.columns:
                    row[col] = group.loc[i, col]
            # Add calculated features with history
            for col in feature_cols:
                for h in range(n_history):
                    row[f'{col}_t-{n_history-1-h}'] = group.loc[i-h, col]
            output_rows.append(row)
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)
    print(f'Saved {len(out_df)} rows to {output_csv}')

def add_feature_history(configObj, foldOutFolder=None):
    n_history = configObj.get('dataProcessing', {}).get('nHistory', 1)
    dataFileNames = configObj.get('dataFileNames', {})
    event_col = 'eventId'
    # Use foldOutFolder if provided
    def full_path(fname):
        return os.path.join(foldOutFolder, fname) if foldOutFolder else fname
    # Train features
    train_input = full_path(dataFileNames.get('trainFeaturesFileCsv', 'trainFeatures.csv'))
    train_output = full_path(dataFileNames.get('trainFeaturesHistoryFileCsv', 'trainDataFeaturesHistory.csv'))
    process_feature_history(train_input, train_output, event_col, n_history)
    # Test features
    test_input = full_path(dataFileNames.get('testFeaturesFileCsv', 'testFeatures.csv'))
    test_output = full_path(dataFileNames.get('testFeaturesHistoryFileCsv', 'testDataFeaturesHistory.csv'))
    process_feature_history(test_input, test_output, event_col, n_history)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add feature history columns to features CSV')
    parser.add_argument('--config', default='nnConfig.json', help='Path to config JSON file (default: nnConfig.json)')
    parser.add_argument('--folder', default=None, help='Output folder for fold (default: None)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configObj = json.load(f)
    add_feature_history(configObj, foldOutFolder=args.folder)
