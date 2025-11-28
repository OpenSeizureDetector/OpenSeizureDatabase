import json
import os
import subprocess


def run_cmd(args, input_text=None, cwd=None):
    return subprocess.run(args, input=input_text, text=True, capture_output=True, cwd=cwd)


def test_strip_phone_array(tmp_path):
    test_json = tmp_path / "events.json"
    data = [
        {"id": 1, "dataSourceName": "Phone"},
        {"id": 2, "dataSourceName": "Watch"},
        {"id": 3, "dataSourceName": "Phone"},
        {"id": 4}
    ]
    test_json.write_text(json.dumps(data, indent=2))

    out_path = tmp_path / "out.json"
    script = os.path.join(os.path.dirname(__file__), '..', 'strip_phone_datasource.py')
    res = run_cmd(['python3', script, str(test_json), '-o', str(out_path)])
    assert res.returncode == 0, res.stderr

    post = json.loads(out_path.read_text())
    ids = [e.get('id') for e in post]
    assert ids == [2, 4]


def test_strip_phone_ndjson(tmp_path):
    test_ndjson = tmp_path / "events.ndjson"
    lines = [
        json.dumps({"id": 11, "dataSourceName": "Phone"}),
        json.dumps({"id": 12, "dataSourceName": "Watch"}),
        json.dumps({"id": 13}),
    ]
    test_ndjson.write_text("\n".join(lines) + "\n")

    out_path = tmp_path / "out.json"
    script = os.path.join(os.path.dirname(__file__), '..', 'strip_phone_datasource.py')
    res = run_cmd(['python3', script, str(test_ndjson), '-o', str(out_path)])
    assert res.returncode == 0, res.stderr

    post = json.loads(out_path.read_text())
    ids = [e.get('id') for e in post]
    assert ids == [12, 13]
