import pandas as pd
from pathlib import Path


def test_extract_features_integration(tmp_path, monkeypatch):
    # Import inside test to ensure PYTHONPATH is respected when running tests
    from user_tools.nnTraining2 import extractFeatures as ef

    # Monkeypatch accelFeatures.calculate_epoch_features to return deterministic features
    class DummyAccel:
        @staticmethod
        def calculate_epoch_features(epoch_data, sf, freq_bands):
            return {'feat_a': 42, 'feat_b': 7}

    monkeypatch.setattr(ef, 'accelFeatures', DummyAccel)

    # Build a minimal flattened CSV with a single event consisting of two
    # successive rows (so 250 samples) so that overlapping windows are
    # produced when step < window.
    cols = ['eventId','dataTime','userId','typeStr','type','osdAlarmState','osdSpecPower','osdRoiPower','hr','o2sat']
    for prefix in ['M','X','Y','Z']:
        for i in range(125):
            cols.append(f"{prefix}{i:03d}")

    rows = []
    for row_idx in [0, 1]:
        row = {}
        for c in cols:
            if c == 'eventId':
                row[c] = 'E1'
            elif c == 'dataTime':
                row[c] = row_idx * 125
            elif c == 'userId':
                row[c] = 'U1'
            elif c == 'typeStr':
                row[c] = 'T'
            elif c == 'type':
                row[c] = 1
            elif c in ['osdAlarmState','osdSpecPower','osdRoiPower','hr','o2sat']:
                row[c] = 0
            else:
                # give each sample a distinguishable value based on row index
                row[c] = float(row_idx)
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)

    in_file = tmp_path / 'in.csv'
    out_file = tmp_path / 'out.csv'
    df.to_csv(in_file, index=False)

    # Use overlapping windows: window=125, step=100 -> two windows (start 0 and 100)
    configObj = {
        'dataProcessing': {
            'window': 125,
            'step': 100,
            'features': []
        },
        'dataFileNames': {}
    }

    # Run extractor (uses streaming path but input is small)
    ef.extractFeatures(str(in_file), str(out_file), configObj)

    out = pd.read_csv(out_file)

    meta_cols = ['eventId','userId','typeStr','type','dataTime','osdAlarmState','osdSpecPower','osdRoiPower','hr','o2sat','startSample','endSample']

    # Check metadata columns are first and in expected order
    assert list(out.columns[:len(meta_cols)]) == meta_cols

    # Check calculated features appear after metadata
    assert 'feat_a' in out.columns
    assert 'feat_b' in out.columns

    # Check raw sample columns exist at the end and are in the expected order
    n = 125
    expected_raw_cols = []
    for prefix in ['M','X','Y','Z']:
        for i in range(n):
            expected_raw_cols.append(f"{prefix}{i:03d}")

    # The last 4*n columns should match the expected raw ordering
    assert list(out.columns[-(4*n):]) == expected_raw_cols

    # Overlapping windows: with 250 samples, window=125, step=100 we expect 2 windows
    assert out.shape[0] == 2
    # Check eventId present in both rows
    assert all(out['eventId'] == 'E1')
    # Check startSample values are 0 and 100 (order may vary depending on processing)
    starts = sorted(list(out['startSample'].astype(int)))
    assert starts == [0, 100]

    # Verify numerical values in the raw sample windows to ensure correct slicing:
    # For the first window starting at 0, all samples should come from the first row (value 0.0)
    row0 = out[out['startSample'] == 0].iloc[0]
    m_vals_row0 = [float(row0[f'M{i:03d}']) for i in range(125)]
    assert all(v == 0.0 for v in m_vals_row0)

    # For the second window starting at 100, first 25 samples are from row0 (0.0), next 100 from row1 (1.0)
    row1 = out[out['startSample'] == 100].iloc[0]
    m_vals_row1 = [float(row1[f'M{i:03d}']) for i in range(125)]
    assert all(m_vals_row1[i] == 0.0 for i in range(25))
    assert all(m_vals_row1[i] == 1.0 for i in range(25, 125))
