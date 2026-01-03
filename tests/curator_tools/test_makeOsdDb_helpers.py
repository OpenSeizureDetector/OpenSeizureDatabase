import os
import sys
import json
import tempfile
import importlib
import pytest

# Skip these tests if pandas is not installed
if importlib.util.find_spec('pandas') is None:
    pytest.skip('pandas not installed; skipping makeOsdDb helper tests', allow_module_level=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from curator_tools import makeOsdDb


def make_sample_json(path):
    events = [
        {"id": "1", "userId": "u1", "type": "Seizure", "subType": "A", "dataTime": "2020-01-01T12:00:00Z", "desc": "one"},
        {"id": "2", "userId": "u2", "type": "Seizure", "subType": "B", "dataTime": "2020-01-02T12:00:00Z", "desc": "two"}
    ]
    with open(path, 'w') as f:
        json.dump(events, f)


def test_save_index_for_file(tmp_path):
    jf = tmp_path / "test.json"
    make_sample_json(jf)
    # call save_index_for_file
    makeOsdDb.save_index_for_file(str(jf), useCacheDir=False, debug=False)
    csvf = str(jf).replace('.json', '.csv')
    assert os.path.exists(csvf)


def test_update_seizure_times_file_creates_backup_and_index(tmp_path):
    jf = tmp_path / "test2.json"
    make_sample_json(jf)
    # create a minimal config file required by update_seizure_times_file
    cfg = tmp_path / "cfg.json"
    cfg_obj = {"osdbDir": str(tmp_path), "seizureTimesFname": ""}
    with open(cfg, 'w') as f:
        json.dump(cfg_obj, f)
    # Call update_seizure_times_file
    makeOsdDb.update_seizure_times_file(str(jf), str(cfg), backup=True, debug=False)
    assert os.path.exists(str(jf) + '.bak')
    csvf = str(jf).replace('.json', '.csv')
    assert os.path.exists(csvf)
