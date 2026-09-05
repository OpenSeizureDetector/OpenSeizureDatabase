import os
import sys
import json
import importlib

import pytest

# Skip these tests if pandas is not installed
if importlib.util.find_spec('pandas') is None:
    pytest.skip('pandas not installed; skipping makeOsdDb helper tests', allow_module_level=True)

# Repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Current helper modules (non-refactor path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'curator_tools')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'curator_tools', 'src')))

from makeIndex import make_index
from database_utils import backup_database
from osdb_sqlite import OsdWorkingDb


def make_sample_json(path):
    events = [
        {
            "id": "1",
            "userId": "u1",
            "type": "Seizure",
            "subType": "A",
            "dataTime": "2020-01-01T12:00:00Z",
            "desc": "one",
            "osdAlarmState": 2,
            "dataSourceName": "Watch",
            "phoneAppVersion": "1.0",
            "watchSdVersion": "1.0",
            "alarmFreqMin": 0,
            "alarmFreqMax": 0,
            "alarmThresh": 0,
            "alarmRatioThresh": 0,
        },
        {
            "id": "2",
            "userId": "u2",
            "type": "Seizure",
            "subType": "B",
            "dataTime": "2020-01-02T12:00:00Z",
            "desc": "two",
            "osdAlarmState": 2,
            "dataSourceName": "Watch",
            "phoneAppVersion": "1.0",
            "watchSdVersion": "1.0",
            "alarmFreqMin": 0,
            "alarmFreqMax": 0,
            "alarmThresh": 0,
            "alarmRatioThresh": 0,
        },
    ]
    with open(path, 'w') as f:
        json.dump(events, f)


def test_make_index_writes_csv(tmp_path):
    jf = tmp_path / 'test.json'
    make_sample_json(jf)

    out_csv = make_index(str(jf), debug=False)

    assert os.path.exists(out_csv)
    assert out_csv.endswith('.csv')


def test_backup_database_and_make_index(tmp_path):
    jf = tmp_path / 'test2.json'
    make_sample_json(jf)

    # New helper for backups operates on SQLite DB files.
    db_path = tmp_path / 'test.db'
    db = OsdWorkingDb(str(db_path))
    db.add_events([
        {
            'id': 'e1',
            'userId': 1,
            'dataTime': '2020-01-01T12:00:00Z',
            'type': 'Seizure',
            'subType': 'A',
            'osdAlarmState': 2,
            'datapoints': []
        }
    ])
    db.conn.close()

    backup_path = backup_database(str(db_path))
    out_csv = make_index(str(jf), debug=False)

    assert os.path.exists(backup_path)
    assert '.backup.' in backup_path
    assert os.path.exists(out_csv)
