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
            "features": ["mean_X", "mean_Y", "mean_Z"],
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
    expected_cols = ["mean_X", "mean_Y", "mean_Z"]
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
    # For axis test events, mean_X, mean_Y, mean_Z should match the sequence
    for axis, col in enumerate(["mean_X", "mean_Y", "mean_Z"]):
        axis_rows = test_rows[test_rows["eventId"] == f"T00{axis+1}"]
        assert (axis_rows[col] != 0).all()
        for other_axis, other_col in enumerate(["mean_X", "mean_Y", "mean_Z"]):
            if other_axis != axis:
                assert (axis_rows[other_col] == 0).all()
    #assert (False, )  # Temporary fail to inspect output
