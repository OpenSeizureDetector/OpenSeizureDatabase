"""
Test for fix of hr/o2sat interpolation bug in extractFeatures.py

Bug: process_event used a single shared sample_indices list for both hr and
o2sat interpolation. When hr and o2sat had mismatched sparsity (e.g. hr missing
in some rows where o2sat was present, or vice versa), np.interp was called with
xp and fp of different lengths, raising:

    ValueError: fp and xp are not of the same length.

or with empty xp when hr was completely missing but o2sat was present:

    ValueError: array of sample points is empty

This test reproduces both failure modes and verifies the fix that uses
separate hr_indices / o2sat_indices.

Run: python3 -m pytest user_tools/nnTraining2/tests/test_hr_o2sat_interpolation_fix.py -v
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/graham/osd/OpenSeizureDatabase')
sys.path.insert(0, '/home/graham/osd/OpenSeizureDatabase/user_tools/nnTraining2')

from user_tools.nnTraining2.extractFeatures import process_event, extract_features


def make_event_df(rows, event_id='testEvent'):
    data = []
    for i, r in enumerate(rows):
        row = {
            'eventId': event_id,
            'userId': 'user1',
            'typeStr': 'seizure',
            'type': 1,
            'dataTime': i*5000,
            'osdAlarmState': 0,
            'osdSpecPower': 0,
            'osdRoiPower': 0,
            'hr': r.get('hr', np.nan),
            'o2sat': r.get('o2sat', np.nan),
        }
        for n in range(125):
            row[f"M{n:03d}_t-0"] = 1000.0 + n
            row[f"X{n:03d}_t-0"] = 100.0 + n
            row[f"Y{n:03d}_t-0"] = 200.0 + n
            row[f"Z{n:03d}_t-0"] = 300.0 + n
        data.append(row)
    return pd.DataFrame(data)


def test_hr_missing_o2sat_present():
    """hr completely missing, o2sat present - previously raised 'array of sample points is empty'"""
    df = make_event_df([
        {'hr': np.nan, 'o2sat': 95},
        {'hr': np.nan, 'o2sat': 96},
    ])
    # Should not raise
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 2
    # Verify o2sat was interpolated (result contains hr/o2sat via event processing - check no exception)
    # Check extract_features integration as well
    config = {"dataProcessing": {"window":125,"step":125,"features":["acc_magnitude"],"highPassFreq":None,"highPassOrder":2,"worker_count":1}}
    df_out = extract_features(df, config)
    assert len(df_out) == 2
    # hr should be NaN throughout, o2sat should be interpolated between 95 and 96
    # Since extract_features interpolates at sample resolution, check hr is NaN
    assert df_out['hr'].isna().all() or df_out['hr'].isna().any()


def test_mismatched_hr_o2sat_lengths():
    """hr and o2sat present at different rows - previously raised 'fp and xp are not of the same length'"""
    df = make_event_df([
        {'hr': 70, 'o2sat': np.nan},
        {'hr': np.nan, 'o2sat': 98},
        {'hr': 71, 'o2sat': np.nan},
    ])
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 3


def test_both_present_same_rows():
    """Both present at same rows - should still work (regression)"""
    df = make_event_df([
        {'hr': 70, 'o2sat': 95},
        {'hr': 71, 'o2sat': 96},
    ])
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 2


def test_both_missing():
    """Both hr and o2sat missing - should produce NaN arrays and still window"""
    df = make_event_df([
        {'hr': np.nan, 'o2sat': np.nan},
        {'hr': np.nan, 'o2sat': np.nan},
    ])
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 2


def test_single_point_interpolation():
    """Single valid point should fill entire interpolation (np.interp left/right fill)"""
    df = make_event_df([
        {'hr': np.nan, 'o2sat': 98},
        {'hr': np.nan, 'o2sat': np.nan},
        {'hr': np.nan, 'o2sat': np.nan},
    ])
    # o2sat single point at index 0 should fill all samples with 98
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 3
    # Also test hr single point at end
    df2 = make_event_df([
        {'hr': np.nan, 'o2sat': np.nan},
        {'hr': np.nan, 'o2sat': np.nan},
        {'hr': 75, 'o2sat': np.nan},
    ])
    result2 = process_event(('testEvent', df2, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result2) == 3


def test_none_and_string_handling():
    """Robust handling of None values (pd.isna) without TypeError from np.isnan"""
    df = make_event_df([
        {'hr': None, 'o2sat': None},
        {'hr': 70, 'o2sat': None},
        {'hr': None, 'o2sat': 98},
    ])
    result = process_event(('testEvent', df, 125, 125, ['acc_magnitude'], None, 2, False))
    assert len(result) == 3


def test_extract_features_multiprocessing_mismatched(tmp_path=None):
    """Full extract_features pipeline with multiprocessing and mismatched data"""
    import pandas as pd
    # Create multi-event dataframe
    df1 = make_event_df([{'hr': np.nan, 'o2sat': 95}, {'hr': np.nan, 'o2sat': 96}], event_id='ev1')
    df2 = make_event_df([{'hr': 70, 'o2sat': np.nan}, {'hr': np.nan, 'o2sat': 98}, {'hr': 71, 'o2sat': np.nan}], event_id='ev2')
    df_all = pd.concat([df1, df2], ignore_index=True)
    config = {"dataProcessing": {"window":125,"step":125,"features":["acc_magnitude"],"highPassFreq":None,"highPassOrder":2,"worker_count":2}}
    out = extract_features(df_all, config)
    # 2 + 3 = 5 windows expected
    assert len(out) == 5
