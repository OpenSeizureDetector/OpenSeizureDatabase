import os
import shutil
import pandas as pd
import pytest

def test_flattenData(tmp_path):
    # Copy simulated_events.json to temp dir
    # Copy simulated_events.json to temp dir
    src = os.path.join(os.path.dirname(__file__), "simulated_events.json")
    dst = tmp_path / "simulated_events.json"
    shutil.copyfile(src, dst)
    # Minimal configObj for flattenData
    configObj = {
        'dataFileNames': {
            'allDataFileJson': str(dst),
            'testDataFileJson': str(dst),
            'testDataFileCsv': str(tmp_path / "test_flattened.csv")
        }
    }
    import user_tools.nnTraining2.flattenData as flattenData
    out_csv = tmp_path / "test_flattened.csv"
    flattenData.flattenOsdb(str(dst), str(out_csv), configObj)
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "eventId" in df.columns
    # Check that test events are present
    assert any(df["eventId"].str.startswith("T"))
