import os
import shutil
import pytest

def test_splitData(tmp_path):
    # Copy selected_events.json to temp dir
    # Generate selected_events.json from simulated_events.json
    src = os.path.join(os.path.dirname(__file__), "simulated_events.json")
    dst = tmp_path / "simulated_events.json"
    shutil.copyfile(src, dst)
    configObj_select = {
        'osdbConfig': {
            'osdbFiles': [str(dst)],
            'cacheDir': str(tmp_path),
            'invalidEvents': [],
        },
        'dataFileNames': {
            'allDataFileJson': str(tmp_path / "selected_events.json")
        },
        'eventFilters': {
            'includeUserIds': [],
            'excludeUserIds': [],
            'includeTypes': [],
            'excludeTypes': [],
            'includeSubTypes': [],
            'excludeSubTypes': [],
            'includeDataSources': [],
            'excludeDataSources': [],
            'includeText': [],
            'excludeText': [],
            'require3dData': False,
            'requireHrData': False,
            'requireO2SatData': False
        }
    }
    import user_tools.nnTraining2.selectData as selectData
    selectData.selectData(configObj_select, outDir=str(tmp_path), debug=False)
    selected_json = tmp_path / "selected_events.json"
    # Minimal configObj for splitData
    configObj = {
        'dataFileNames': {
            'trainDataFileJson': str(tmp_path / "train_events.json"),
            'testDataFileJson': str(tmp_path / "test_events.json"),
            'allDataFileJson': str(selected_json),
            'valDataFileJson': str(tmp_path / "val_events.json")
        },
        'dataProcessing': {
            'testProp': 0.2,
            'validationProp': 0.0,
            'fixedTestEvents': [],
            'fixedTrainEvents': []
        },
        'osdbConfig': {
            'cacheDir': str(tmp_path)
        }
    }
    import user_tools.nnTraining2.splitData as splitData
    splitData.splitData(configObj, kFold=1, outDir=str(tmp_path), debug=False)
    out_train = tmp_path / "train_events.json"
    out_test = tmp_path / "test_events.json"
    assert out_train.exists()
    assert out_test.exists()
    import json
    with open(out_train) as f:
        train_events = json.load(f)
    with open(out_test) as f:
        test_events = json.load(f)
    # Check that all events are split
    total = len(train_events) + len(test_events)
    with open(selected_json) as f:
        all_events = json.load(f)
    assert total == len(all_events)
    # Check that test events are present in either train or test
    assert any(e['id'].startswith('T') for e in train_events + test_events)
