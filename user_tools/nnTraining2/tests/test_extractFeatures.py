import os
import shutil
import pandas as pd
import pytest

def test_extractFeatures(tmp_path):
    # Copy flattened CSV to temp dir
    # Generate test_flattened.csv from simulated_events.json
    src = os.path.join(os.path.dirname(__file__), "simulated_events.json")
    dst = tmp_path / "simulated_events.json"
    shutil.copyfile(src, dst)
    configObj_flat = {
        'dataFileNames': {
            'allDataFileJson': str(dst),
            'testDataFileJson': str(dst),
            'testDataFileCsv': str(tmp_path / "test_flattened.csv")
        }
    }
    import user_tools.nnTraining2.flattenData as flattenData
    out_flat_csv = tmp_path / "test_flattened.csv"
    flattenData.flattenOsdb(str(dst), str(out_flat_csv), configObj_flat)
    # Minimal configObj for extractFeatures
    configObj = {
        "dataProcessing": {
            "window": 125,
            "step": 125,
            "features": ["mean_x", "mean_y", "mean_z"],
            "highPassFreq": 0.5,
            "highPassOrder": 2
        }
    }
    import user_tools.nnTraining2.extractFeatures as extractFeatures
    df_flat = pd.read_csv(out_flat_csv)
    df_feat = extractFeatures.extract_features(df_flat, configObj)
    out_csv = tmp_path / "test_features.csv"
    df_feat.to_csv(out_csv, index=False)
    assert out_csv.exists()
    df_out = pd.read_csv(out_csv)
    # Print columns for debug if assertion fails
    expected_cols = ["mean_x", "mean_y", "mean_z"]
    missing = [col for col in expected_cols if col not in df_out.columns]
    if missing:
        print("Missing columns:", missing)
        print("Available columns:", df_out.columns)
    for col in expected_cols:
        if col not in df_out.columns:
            print("Available columns:", df_out.columns)
        assert col in df_out.columns
    test_rows = df_out[df_out["eventId"].str.startswith("T")]
    assert not test_rows.empty
    # For axis test events, mean_x, mean_y, mean_z should match the sequence
    for axis, col in enumerate(["mean_x", "mean_y", "mean_z"]):
        axis_rows = test_rows[test_rows["eventId"] == f"T00{axis+1}"]
        assert (axis_rows[col] != 0).all()
        for other_axis, other_col in enumerate(["mean_x", "mean_y", "mean_z"]):
            if other_axis != axis:
                assert (axis_rows[other_col] == 0).all()
    #assert (False, )  # Temporary fail to inspect output


# Event 8420 style timestamps: naive 'YYYY-MM-DD HH:MM:SS' and ISO
# 'YYYY-MM-DDTHH:MM:SSZ' rows interleaved in chronological order.
MIXED_FORMAT_INPUT = [
    '2022-07-12 16:21:29',
    '2022-07-12 16:21:33',
    '2022-07-12 16:23:18',
    '2022-07-12T16:22:14Z',
    '2022-07-12T16:22:19Z',
]
MIXED_FORMAT_CHRONOLOGICAL = [
    '2022-07-12 16:21:29',
    '2022-07-12 16:21:33',
    '2022-07-12T16:22:14Z',
    '2022-07-12T16:22:19Z',
    '2022-07-12 16:23:18',
]


def test_sort_event_by_time_mixed_formats():
    """Mixed timestamp formats must be ordered by time, not by string."""
    from user_tools.nnTraining2.extractFeatures import sort_event_by_time
    # Sanity check: string sorting gives a different (wrong) order, so this
    # test really exercises the fix.
    assert sorted(MIXED_FORMAT_INPUT) != MIXED_FORMAT_CHRONOLOGICAL

    df = pd.DataFrame({'eventId': ['8420'] * len(MIXED_FORMAT_INPUT),
                       'dataTime': MIXED_FORMAT_INPUT})
    out = sort_event_by_time(df)
    assert out['dataTime'].tolist() == MIXED_FORMAT_CHRONOLOGICAL
    # Input frame must not be modified in place.
    assert df['dataTime'].tolist() == MIXED_FORMAT_INPUT


def test_sort_event_by_time_uniform_formats_unchanged():
    from user_tools.nnTraining2.extractFeatures import sort_event_by_time
    times = ['2022-07-12 16:21:29', '2022-07-12 16:21:33', '2022-07-12 16:21:38']
    df = pd.DataFrame({'eventId': ['1'] * 3, 'dataTime': times})
    out = sort_event_by_time(df)
    assert out['dataTime'].tolist() == times


def test_process_event_simple_orders_rows_chronologically():
    """Feature rows must come out in time order for the model's rolling buffer."""
    from user_tools.nnTraining2.extractFeatures import process_event_simple

    n = len(MIXED_FORMAT_INPUT)
    data = {
        'eventId': ['8420'] * n,
        'userId': [45] * n,
        'typeStr': ['Seizure/Tonic-Clonic'] * n,
        'type': [1] * n,
        'dataTime': MIXED_FORMAT_INPUT,
        'osdAlarmState': [0] * n,
        'osdSpecPower': [0.0] * n,
        'osdRoiPower': [0.0] * n,
        'hr': [60.0] * n,
        'o2sat': [-1] * n,
    }
    # One distinct constant magnitude per row so we can identify row order.
    for i in range(125):
        data[f'M{i:03d}'] = [1000.0 + row for row in range(n)]
    event_df = pd.DataFrame(data)

    rows = process_event_simple(('8420', event_df, 125, 125, ['acc_magnitude']))
    out = pd.DataFrame(rows)
    assert out['dataTime'].tolist() == MIXED_FORMAT_CHRONOLOGICAL
    # M000 carries the row identity: expect rows 0,1,3,4,2 in time order.
    assert out['M000'].tolist() == [1000.0, 1001.0, 1003.0, 1004.0, 1002.0]
