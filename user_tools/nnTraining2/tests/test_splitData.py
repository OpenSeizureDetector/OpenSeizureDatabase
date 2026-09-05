import os
import shutil
import json
import pandas as pd
import pytest


def test_splitData(tmp_path):
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
    with open(selected_json) as f:
        all_events = json.load(f)

    df = pd.DataFrame(
        [{
            'eventId': event['id'],
            'type': event.get('type', 'nda'),
            'userId': event.get('userId', 'test'),
            'dataTime': event.get('dataTime', '')
        } for event in all_events]
    )
    all_csv = tmp_path / "allData.csv"
    df.to_csv(all_csv, index=False)

    configObj = {
        'dataFileNames': {
            'trainDataFileCsv': 'train_events.csv',
            'testDataFileCsv': 'test_events.csv',
            'allDataFileCsv': str(all_csv),
            'valDataFileCsv': 'val_events.csv'
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
    splitData.splitCsvData(configObj, str(all_csv), outDir=str(tmp_path), kFold=1, nestedKfold=1, debug=False)

    out_train = tmp_path / "train_events.csv"
    out_test = tmp_path / "test_events.csv"
    assert out_train.exists()
    assert out_test.exists()

    train_df = pd.read_csv(out_train)
    test_df = pd.read_csv(out_test)
    total = len(train_df) + len(test_df)
    assert total == len(df)
    assert any(str(e).startswith('T') for e in list(train_df['eventId']) + list(test_df['eventId']))
