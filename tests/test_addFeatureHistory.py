import os
import pandas as pd
import numpy as np
import json
import tempfile
import sys
from pathlib import Path

# Ensure repo root is on sys.path for imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from user_tools.nnTraining2 import addFeatureHistory


def make_sample_features_csv(path):
    # Create two events with multiple rows each and simple feature columns a,b
    rows = []
    for eventId in [1, 2]:
        for i in range(5):
            rows.append({
                'eventId': eventId,
                'userId': 10 + eventId,
                'type': 1 if eventId == 1 else 0,
                'dataTime': 1000 + i,
                'a': i + eventId * 10,
                'b': (i * 2) + eventId * 100,
                'x': 0,
                'y': 0,
                'z': 0,
                'magnitude': 0,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_add_feature_history_stream_and_df(tmp_path):
    # Prepare input file
    in_csv = tmp_path / 'features.csv'
    out_csv = tmp_path / 'features_hist.csv'
    make_sample_features_csv(str(in_csv))

    # Config object with small batch to force streaming writes
    config = {
        'dataProcessing': {
            'nHistory': 3,
            'batch_size': 2,
            'stream_chunksize': 2
        },
        'dataFileNames': {}
    }

    # Run streaming (filename input)
    addFeatureHistory.process_feature_history(str(in_csv), str(out_csv), 'eventId', 3, configObj=config)

    df_out = pd.read_csv(str(out_csv))

    # Check that history columns for 'a' and 'b' exist
    expected_cols = ['a_t-2', 'a_t-1', 'a_t-0', 'b_t-2', 'b_t-1', 'b_t-0']
    for c in expected_cols:
        assert c in df_out.columns, f"Missing column {c}"

    # Build expected rows from input file using same semantics as the code
    df_in = pd.read_csv(str(in_csv))
    expected_rows = []
    raw_cols = ['x', 'y', 'z', 'magnitude']
    exclude_cols = ['eventId', 'userId', 'typeStr', 'type', 'dataTime', 'osdAlarmState',
                    'osdSpecPoewr', 'osdRoiPower', 'startSample', 'endSample']
    feature_cols = [c for c in df_in.columns if c not in raw_cols and c not in exclude_cols]
    keep_cols = [c for c in df_in.columns if c not in feature_cols]
    n_history = 3
    for event_id, group in df_in.groupby('eventId'):
        g = group.reset_index(drop=True)
        if len(g) < n_history:
            continue
        for i in range(n_history - 1, len(g)):
            row = {col: g.loc[i, col] for col in keep_cols if col in g.columns}
            for col in feature_cols:
                for h in range(n_history):
                    row[f'{col}_t-{n_history-1-h}'] = g.loc[i-h, col]
            expected_rows.append(row)

    df_expected = pd.DataFrame(expected_rows)
    # Read output and compare relevant columns
    df_out = pd.read_csv(str(out_csv))
    # Ensure same number of rows
    assert len(df_out) == len(df_expected)
    # Compare feature history columns and keep cols
    compare_cols = [c for c in df_expected.columns if c in df_out.columns]
    pd.testing.assert_frame_equal(df_out[compare_cols].reset_index(drop=True), df_expected[compare_cols].reset_index(drop=True))

    # Also test in-memory DataFrame path produces same columns
    out_csv2 = tmp_path / 'features_hist2.csv'
    addFeatureHistory.process_feature_history(df_in, str(out_csv2), 'eventId', 3)
    df_out2 = pd.read_csv(str(out_csv2))
    assert 'a_t-0' in df_out2.columns
