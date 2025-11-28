import json
import os
import subprocess


def run_cmd(args, input_text=None, cwd=None):
    return subprocess.run(args, input=input_text, text=True, capture_output=True, cwd=cwd)


def write_events(path, events):
    path.write_text(json.dumps(events, indent=2))


def test_merge_respects_type_subtype_and_reference(tmp_path):
    # Prepare two files: first is reference
    ref = tmp_path / 'ref.json'
    other = tmp_path / 'other.json'

    # Same user, close times (<30s), different subType -> NOT duplicates
    e1 = {
        "id": "A1", "userId": "U1", "type": "seizure", "subType": "tonic",
        "dataTime": "2024-01-02T12:00:00Z", "datapoints": [1]
    }
    e2 = {
        "id": "A2", "userId": "U1", "type": "seizure", "subType": "clonic",
        "dataTime": "2024-01-02T12:00:15Z", "datapoints": [1]
    }

    # Same user, same type/subType, within window -> duplicates; choose reference or tie-break
    e3_ref = {
        "id": "B1", "userId": "U2", "type": "false_alarm", "subType": "movement",
        "dataTime": "2024-01-02T13:00:00Z", "datapoints": [1,2]
    }
    e3_other = {
        "id": "B2", "userId": "U2", "type": "false_alarm", "subType": "movement",
        "dataTime": "2024-01-02T13:00:10Z", "datapoints": [1]
    }

    write_events(ref, [e1, e3_ref])
    write_events(other, [e2, e3_other])

    # Run merge with 30s window, reference is first file
    out_dir = tmp_path / 'out'
    script = os.path.join(os.path.dirname(__file__), '..', 'merge_json.py')
    res = run_cmd(['python3', script, '--inDir', str(tmp_path), '--outDir', str(out_dir), '--outFile', 'merged.json', '--dedupe-window', '30', 'ref.json', 'other.json'])
    assert res.returncode == 0, res.stderr

    merged_path = out_dir / 'merged.json'
    merged = json.loads(merged_path.read_text())

    # Expect e1 and e2 both present (different subType)
    ids = {ev['id'] for ev in merged}
    assert 'A1' in ids and 'A2' in ids

    # For duplicates B1/B2 with same type/subType: keep B1 (from reference, and has more datapoints)
    assert 'B1' in ids and 'B2' not in ids


def test_merge_tie_break_prefers_later_when_equal_datapoints(tmp_path):
    ref = tmp_path / 'ref.json'
    other = tmp_path / 'other.json'

    # Same user/type/subType within window, equal datapoint counts
    a_ref = {
        "id": "T1", "userId": "U9", "type": "seizure", "subType": "tc",
        "dataTime": "2024-01-02T10:00:00Z", "datapoints": [1]
    }
    a_other = {
        "id": "T2", "userId": "U9", "type": "seizure", "subType": "tc",
        "dataTime": "2024-01-02T10:00:05Z", "datapoints": [1]
    }

    # Put the earlier one in ref, later in other; since one is reference, keep reference (T1)
    write_events(ref, [a_ref])
    write_events(other, [a_other])

    out_dir = tmp_path / 'out'
    script = os.path.join(os.path.dirname(__file__), '..', 'merge_json.py')
    res = run_cmd(['python3', script, '--inDir', str(tmp_path), '--outDir', str(out_dir), '--outFile', 'merged.json', '--dedupe-window', '30', 'ref.json', 'other.json'])
    assert res.returncode == 0, res.stderr

    merged = json.loads((out_dir / 'merged.json').read_text())
    ids = {ev['id'] for ev in merged}
    assert 'T1' in ids and 'T2' not in ids


def test_merge_window_boundary_inclusive(tmp_path):
    # Verify that events exactly at the window boundary are considered duplicates
    ref = tmp_path / 'ref.json'
    other = tmp_path / 'other.json'

    e1 = {
        "id": "W1", "userId": "U7", "type": "false_alarm", "subType": "movement",
        "dataTime": "2024-01-02T14:00:00Z", "datapoints": [1]
    }
    # Exactly 30 seconds later
    e2 = {
        "id": "W2", "userId": "U7", "type": "false_alarm", "subType": "movement",
        "dataTime": "2024-01-02T14:00:30Z", "datapoints": [1,2]
    }

    write_events(ref, [e1])
    write_events(other, [e2])

    out_dir = tmp_path / 'out'
    script = os.path.join(os.path.dirname(__file__), '..', 'merge_json.py')
    res = run_cmd(['python3', script, '--inDir', str(tmp_path), '--outDir', str(out_dir), '--outFile', 'merged.json', '--dedupe-window', '30', 'ref.json', 'other.json'])
    assert res.returncode == 0, res.stderr

    merged = json.loads((out_dir / 'merged.json').read_text())
    ids = {ev['id'] for ev in merged}
    # Should dedupe and keep W1 because it's from reference file
    assert 'W1' in ids and 'W2' not in ids
