import os
import sys
import tempfile
import csv
import importlib
import pytest

# Skip these tests if pandas is not installed in the environment
if importlib.util.find_spec('pandas') is None:
    pytest.skip('pandas not installed; skipping osd_index_utils tests', allow_module_level=True)

# make repo root importable for pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from curator_tools.osd_index_utils import write_osd_index


def make_sample_events():
    return [
        {"id": "e1", "userId": "u1", "type": "Seizure", "subType": "A", "dataTime": "2020-01-01T12:00:00Z", "desc": "one"},
        {"id": "e2", "userId": "u2", "type": "Seizure", "subType": "B", "dataTime": "2020-01-02T12:00:00Z", "desc": "two"},
        {"id": "e3", "userId": "u3", "type": "Fall", "subType": "X", "dataTime": None, "desc": "three"}
    ]


def test_write_osd_index_basic(tmp_path):
    events = make_sample_events()
    out_csv = tmp_path / "index.csv"
    write_osd_index(events, str(out_csv))
    assert out_csv.exists()
    # verify header columns present
    with open(out_csv, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert 'id' in header
        assert 'dataTime' in header
