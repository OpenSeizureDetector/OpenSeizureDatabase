import json
import os
import subprocess


def run_cmd(args, input_text=None, cwd=None):
    return subprocess.run(args, input=input_text, text=True, capture_output=True, cwd=cwd)


def test_update_single_event_preserves_other_fields(tmp_path):
    test_json = tmp_path / "events.json"
    data = [
        {
            "id": 101,
            "eventId": "E101",
            "userId": "U1",
            "type": "seizure",
            "subType": "tonic",
            "desc": "initial",
            "datasource": "Watch",
            "dataTime": "2024-01-02T12:00:00Z",
            "flags": {"reviewed": False, "confidence": 0.73},
            "notes": ["a", "b"],
            "datapoints": [
                {"t": "2024-01-02T12:00:01Z", "simpleSpec": [1.1], "rawData": [10], "rawData3D": [[1,2,3]]}
            ]
        },
        {
            "id": 202,
            "eventId": "E202",
            "userId": "U2",
            "type": "false_alarm",
            "subType": "movement",
            "desc": "fa",
            "datasource": "Phone",
            "dataTime": "2024-01-02T12:05:00Z",
            "datapoints": [
                {"t": "2024-01-02T12:05:01Z", "simpleSpec": [0.1], "rawData": [1], "rawData3D": [[0,0,0]]}
            ]
        }
    ]
    test_json.write_text(json.dumps(data, indent=2))

    # Run update with --yes
    script = os.path.join(os.path.dirname(__file__), '..', 'edit_event.py')
    res = run_cmd([
        'python3', script, str(test_json), '--id', '101', '--type', 'seizure', '--subType', 'clonic', '--userId', 'U9', '--desc', 'updated desc', '--yes'
    ])
    assert res.returncode == 0, res.stderr

    # Verify updates and preserved fields
    post = json.loads(test_json.read_text())
    e101 = next(e for e in post if str(e.get('id')) == '101')
    assert e101['userId'] == 'U9'
    assert e101['subType'] == 'clonic'
    assert e101['desc'] == 'updated desc'
    assert e101['type'] == 'seizure'
    # unchanged
    assert e101['datasource'] == 'Watch'
    assert e101['flags'] == {"reviewed": False, "confidence": 0.73}
    assert e101['notes'] == ["a", "b"]
    assert isinstance(e101['datapoints'], list) and len(e101['datapoints']) == 1


def test_delete_multiple_ids(tmp_path):
    test_json = tmp_path / "events.json"
    data = [
        {"id": 1}, {"id": 2}, {"id": 3}
    ]
    test_json.write_text(json.dumps(data))

    script = os.path.join(os.path.dirname(__file__), '..', 'edit_event.py')
    res = run_cmd(['python3', script, str(test_json), '--id', '1', '3', '--delete', '--yes'])
    assert res.returncode == 0, res.stderr
    post = json.loads(test_json.read_text())
    assert [e['id'] for e in post] == [2]
    # Verify removed events file exists and contains the deleted events
    # Find backup timestamp from created backup filename
    import glob, os as _os
    backups = sorted(glob.glob(str(test_json) + '.*'))
    assert backups, 'No backup file created'
    # Extract timestamp from backup name: events.json.<timestamp>
    ts = backups[-1].split('.')[-1]
    removed_path = _os.fspath(tmp_path / f"events.json_removed_{ts}.json")
    assert _os.path.exists(removed_path), 'Removed events file not found'
    removed = json.loads(open(removed_path).read())
    removed_ids = [e.get('id') for e in removed]
    assert set(removed_ids) == {1, 3}


def test_ndjson_support_update(tmp_path):
    test_ndjson = tmp_path / "events.ndjson"
    lines = [
        json.dumps({"id": 10, "userId": "U", "type": "x"}),
        json.dumps({"id": 20, "userId": "U", "type": "y"})
    ]
    test_ndjson.write_text("\n".join(lines) + "\n")

    script = os.path.join(os.path.dirname(__file__), '..', 'edit_event.py')
    res = run_cmd(['python3', script, str(test_ndjson), '--id', '20', '--type', 'z', '--yes'])
    assert res.returncode == 0, res.stderr

    post_lines = test_ndjson.read_text().strip().splitlines()
    objs = [json.loads(l) for l in post_lines]
    assert any(o.get('id') == 20 and o.get('type') == 'z' for o in objs)
