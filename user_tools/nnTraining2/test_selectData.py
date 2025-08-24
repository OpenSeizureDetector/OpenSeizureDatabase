import os
import shutil
import pytest

def test_selectData(tmp_path):
    # Copy simulated_events.json to temp dir
    src = os.path.join(os.path.dirname(__file__), "simulated_events.json")
    dst = tmp_path / "simulated_events.json"
    shutil.copyfile(src, dst)
    # Minimal configObj for selectData
    configObj = {
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
    selectData.selectData(configObj, outDir=str(tmp_path), debug=False)
    out_json = tmp_path / "selected_events.json"
    assert out_json.exists()
    import json
    with open(out_json) as f:
        events = json.load(f)
    # Should contain all events
    assert len(events) > 0
    # Should contain test events
    assert any(e['id'].startswith('T') for e in events)
