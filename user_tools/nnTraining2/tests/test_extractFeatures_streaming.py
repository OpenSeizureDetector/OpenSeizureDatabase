#!/usr/bin/env python3
"""Tests for chunked streaming post-processing in extractFeatures.

The temp-file post-processing step must not load multi-GB files into one
DataFrame (swap-thrash stall). _postprocess_streamed_tmp transforms in chunks
with O(chunk) peak memory while producing byte-equivalent output to the legacy
whole-frame logic: same column order, same header-row cleanup, same type
coercion, same row order.
"""
import os

import pandas as pd

from user_tools.nnTraining2 import extractFeatures as ef


def _tmp_csv(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_ordered_feature_columns_layout():
    cols = ['Z001', 'eventId', 'myFeatB', 'M000', 'type', 'myFeatA',
            'dataTime', 'zzz_unknown', 'X000']
    ordered = ef._ordered_feature_columns(cols)
    # meta first (in canonical order), then sorted features, then raw
    assert ordered[:3] == ['eventId', 'type', 'dataTime']
    assert 'myFeatA' in ordered and 'myFeatB' in ordered
    assert ordered.index('myFeatA') < ordered.index('myFeatB')
    # raw columns keep input order (legacy behaviour, not sorted)
    assert ordered[-3:] == ['Z001', 'M000', 'X000']
    # unknown columns are kept as (sorted) calculated features (legacy)
    assert ordered.index('zzz_unknown') > ordered.index('myFeatB')


def test_clean_postprocess_chunk_drops_header_rows_and_coerces():
    df = pd.DataFrame({
        'eventId': ['E1', 'eventId', 'E2'],
        'type': ['1', 'type', '0'],
        'dataTime': ['2022-01-01', 'dataTime', '2022-01-02'],
        'M000': [1000.0, 'M000', 900.0],
    })
    cleaned, n_seiz, n_non = ef._clean_postprocess_chunk(df)
    assert len(cleaned) == 2
    assert list(cleaned['eventId']) == ['E1', 'E2']
    assert pd.api.types.is_numeric_dtype(cleaned['type'])
    assert (n_seiz, n_non) == (1, 1)


def test_postprocess_streamed_tmp_equivalence(tmp_path):
    # shuffled columns + one header-row + string types, as the worker writer
    # can emit them
    columns = ['M001', 'eventId', 'type', 'M000', 'dataTime', 'hr']
    rows = [
        [1001.0, 'E1', '1', 1000.0, '2022-01-01 00:00:00', 70],
        [900.0, 'E2', '0', 901.0, '2022-01-01 00:00:05', 71],
        ['M001', 'eventId', 'type', 'M000', 'dataTime', 'hr'],  # header-row
        [1002.0, 'E1', '1', 1003.0, '2022-01-01 00:00:10', 72],
    ]
    tmp = str(tmp_path / 'tmp.csv')
    final = str(tmp_path / 'final.csv')
    _tmp_csv(tmp, columns, rows)
    n_rows, n_seiz, n_non = ef._postprocess_streamed_tmp(
        tmp, final, chunk_rows=2)  # tiny chunks to exercise chunk boundaries
    assert (n_rows, n_seiz, n_non) == (3, 2, 1)
    out = pd.read_csv(final)
    # meta first, then raw in input order (legacy preserves input order)
    assert list(out.columns) == ['eventId', 'type', 'dataTime', 'hr',
                                 'M001', 'M000']
    assert list(out['eventId']) == ['E1', 'E2', 'E1']
    assert list(out['M000']) == [1000.0, 901.0, 1003.0]
    assert list(out['type']) == [1, 0, 1]


def test_postprocess_empty_writes_header_only(tmp_path):
    tmp = str(tmp_path / 'tmp.csv')
    final = str(tmp_path / 'final.csv')
    pd.DataFrame(columns=['eventId', 'type', 'M000']).to_csv(tmp, index=False)
    n_rows, n_seiz, n_non = ef._postprocess_streamed_tmp(tmp, final)
    assert (n_rows, n_seiz, n_non) == (0, 0, 0)
    out = pd.read_csv(final)
    assert list(out.columns) == ['eventId', 'type', 'M000']
    assert len(out) == 0


def _tiny_flattened_csv(path, n_events=3, rows_per_event=2):
    cols = ['eventId', 'dataTime', 'userId', 'typeStr', 'type',
            'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat']
    for prefix in ['M', 'X', 'Y', 'Z']:
        cols.extend(f"{prefix}{i:03d}" for i in range(125))
    rows = []
    for e in range(n_events):
        for r in range(rows_per_event):
            row = {'eventId': f'E{e}',
                   'dataTime': f'2022-01-01 00:00:{e * 10 + r:02d}',
                   'userId': 'U1', 'typeStr': 'T',
                   'type': 1 if e == 0 else 0,
                   'osdAlarmState': 0, 'osdSpecPower': 0, 'osdRoiPower': 0,
                   'hr': 70, 'o2sat': 98}
            for prefix in ['M', 'X', 'Y', 'Z']:
                for i in range(125):
                    row[f"{prefix}{i:03d}"] = 1000.0 + r
            rows.append(row)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_streaming_extract_writes_final_and_returns_path(tmp_path):
    src = str(tmp_path / 'flat.csv')
    out = str(tmp_path / 'feat.csv')
    _tiny_flattened_csv(src)
    config = {
        'dataProcessing': {'window': 125, 'step': 125,
                           'features': ['acc_magnitude'],
                           'simpleMagnitudeOnly': True,
                           'worker_count': 1,
                           'postprocess_chunksize': 2},
        'dataFileNames': {},
    }
    result = ef.extractFeatures(src, out, config, debug=False)
    assert result == out
    assert os.path.exists(out)
    df = pd.read_csv(out)
    assert len(df) == 3 * 2  # all rows preserved
    assert list(df.columns[:4]) == ['eventId', 'userId', 'typeStr', 'type']
    assert 'M000' in df.columns
    # tmp file cleaned up
    assert not os.path.exists(out + '.tmp.csv')
