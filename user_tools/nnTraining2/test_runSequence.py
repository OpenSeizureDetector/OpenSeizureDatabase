import os
import json
import shutil
import pytest
import pandas as pd
from user_tools.nnTraining2 import runSequence

TEST_SIM_EVENTS = "simulated_events.json"
OUTPUT_FLATTENED = "output/test_flattened.csv"
OUTPUT_FEATURES = "output/test_features.csv"

@pytest.fixture(scope="module")
def setup_simulated_events(tmp_path_factory):
    # Copy the simulated_events.json to a temp directory
    tmpdir = tmp_path_factory.mktemp("data")
    src = os.path.join(os.path.dirname(__file__), TEST_SIM_EVENTS)
    dst = os.path.join(tmpdir, TEST_SIM_EVENTS)
    shutil.copyfile(src, dst)
    return tmpdir


def test_flatten_and_features(setup_simulated_events):
    tmpdir = setup_simulated_events
    os.chdir(tmpdir)
    # Run flattenData and extractFeatures on the simulated events
    import user_tools.nnTraining2.flattenData as flattenData
    import user_tools.nnTraining2.extractFeatures as extractFeatures
    # Flatten
    output_dir = os.path.dirname(OUTPUT_FLATTENED)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    flattenData.flattenOsdb(TEST_SIM_EVENTS, OUTPUT_FLATTENED, {})
    assert os.path.exists(OUTPUT_FLATTENED)
    df_flat = pd.read_csv(OUTPUT_FLATTENED)
    # Check expected columns
    assert "eventId" in df_flat.columns
    # Extract features
    configObj = {"dataProcessing": {"window": 125, "step": 125, "features": ["mean_X", "mean_Y", "mean_Z"]}}
    df_feat = extractFeatures.extract_features(df_flat, configObj)
    df_feat.to_csv(OUTPUT_FEATURES, index=False)
    assert os.path.exists(OUTPUT_FEATURES)
    df_out = pd.read_csv(OUTPUT_FEATURES)
    # Check that the output contains the expected feature columns
    for col in ["mean_X", "mean_Y", "mean_Z"]:
        assert col in df_out.columns
    # Check that test events produce predictable output
    test_rows = df_out[df_out["eventId"].str.startswith("T")]
    assert not test_rows.empty
    # For axis test events, mean_X, mean_Y, mean_Z should match the sequence
    for axis, col in enumerate(["mean_X", "mean_Y", "mean_Z"]):
        axis_rows = test_rows[test_rows["eventId"] == f"T00{axis+1}"]
        # The mean for the axis with sequence should be nonzero, others should be zero
        assert (axis_rows[col] != 0).all()
        for other_axis, other_col in enumerate(["mean_X", "mean_Y", "mean_Z"]):
            if other_axis != axis:
                assert (axis_rows[other_col] == 0).all()
