#!/usr/bin/env python3
"""
Tests for warm-up masking and data-gap buffer handling.

Background: rolling-buffer models (e.g. CnnLstmModelPyTorch, 45 s buffer = 9
datapoints) score the first datapoints of each buffer segment with a window
that still contains test-time pre-fill. Those "warm-up" datapoints are flagged
(df['is_warm']) and excluded from alarm decisions, but still scored/plotted.

Buffer segments start at event boundaries and at dataTime gaps (missing-data
spans left as discontinuities by flattenData - no synthetic filler rows).
nnTrainer.df2trainingData restarts the buffer at gaps (post-gap warm-up rows
are dropped from training); nnTester restarts, re-prefills realistically, and
flags warm-up rows for decision masking.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_tools.nnTraining2 import nnTester
from user_tools.nnTraining2 import nnTrainer
from user_tools.nnTraining2 import flattenData


# ---------------------------------------------------------------------------
# decision helpers
# ---------------------------------------------------------------------------

def test_split_nonwarm_segments():
    probs = [0.1, 0.9, 0.9, 0.2]
    warm = [True, False, False, True]
    segs = nnTester._split_nonwarm_segments(probs, warm)
    assert len(segs) == 1
    assert list(segs[0]) == [0.9, 0.9]


def test_split_nonwarm_segments_all_warm():
    segs = nnTester._split_nonwarm_segments([0.9, 0.9], [True, True])
    assert segs == []


def test_event_rule_ignores_warm():
    # Only warm datapoint is positive -> masked decision is negative.
    assert nnTester._event_positive_from_probs(
        [0.9, 0.1], 0.5, mode='event', warm_mask=[True, False]) == 0
    # Non-warm datapoint positive -> masked decision still positive.
    assert nnTester._event_positive_from_probs(
        [0.1, 0.9], 0.5, mode='event', warm_mask=[True, False]) == 1
    # Unmasked behaviour unchanged.
    assert nnTester._event_positive_from_probs([0.9, 0.1], 0.5, mode='event') == 1


def test_production_rule_cannot_bridge_masked_region():
    probs = [0.9, 0.9, 0.9, 0.9]
    # Unmasked: run of 4 -> alarm.
    assert nnTester._event_positive_from_probs(
        probs, 0.5, mode='production', consecutive_required=3) == 1
    # Masked middle datapoint splits the run into 2+1 -> no alarm.
    assert nnTester._event_positive_from_probs(
        probs, 0.5, mode='production', consecutive_required=3,
        warm_mask=[False, False, True, False]) == 0
    # Masked trailing datapoint leaves a run of 3 -> alarm.
    assert nnTester._event_positive_from_probs(
        probs, 0.5, mode='production', consecutive_required=3,
        warm_mask=[False, False, False, True]) == 1


def test_threshold_sweep_masked_curves_and_exclusion():
    event_probs = [[0.9, 0.1], [0.1, 0.9], [0.95, 0.95]]
    labels = [0, 1, 0]
    warm = [[True, False], [False, False], [True, True]]
    out = nnTester._threshold_metrics_from_event_probs(
        event_probs, labels, [0.5], mode='event', warm_masks=warm)
    # standard: ev0 FP (0.9), ev1 TP, ev2 FP
    assert out['fp'] == [2] and out['tp'] == [1]
    # masked: ev0 negative (only 0.1 non-warm), ev1 positive, ev2 excluded
    assert out['fp_masked'] == [0]
    assert out['tp_masked'] == [1]
    assert out['n_excluded_warm_only'] == 1


def test_masked_event_metrics_sentinel():
    m = nnTester._masked_event_metrics([0, 1, 0, 1], [1, 0, -1, -1])
    assert (m['tp'], m['fp'], m['tn'], m['fn']) == (0, 1, 0, 1)
    assert m['n_excluded'] == 2
    assert m['tpr'] == 0.0 and m['fpr'] == 1.0


def test_prod_masked_preds_sentinel_and_segments():
    df = pd.DataFrame({
        'event_probs_list': [[0.9, 0.9, 0.9], [0.9, 0.9, 0.9], [0.1, 0.2]],
        'event_warm_list': [[False] * 3, [False, False, True], [True, True]],
    })
    preds = nnTester._prod_masked_preds(df, 0.5, consecutive_required=3)
    assert list(preds) == [1, 0, -1]


def test_segment_rng_deterministic_per_event():
    r1 = nnTester._segment_rng(42, '100')
    r2 = nnTester._segment_rng(42, '100')
    assert r1.normal(0, 1, 10).tolist() == r2.normal(0, 1, 10).tolist()
    r3 = nnTester._segment_rng(42, '101')
    assert r1.normal(0, 1, 10).tolist() != r3.normal(0, 1, 10).tolist()
    assert nnTester._segment_rng(None, '100') is not None


# ---------------------------------------------------------------------------
# training-side gap reset (df2trainingData)
# ---------------------------------------------------------------------------

class _TinyBufferModel:
    """2-datapoint magnitude buffer: first dp of each segment -> None."""
    accel_input_mode = 'magnitude'

    def __init__(self):
        self.accBuf = []

    def resetAccBuf(self):
        self.accBuf = []

    def dp2vector(self, dpObj, normalise=False):
        raw = dpObj.get('rawData', None)
        if raw is None:
            return None
        self.accBuf.extend(list(raw))
        if len(self.accBuf) > 250:
            self.accBuf = self.accBuf[-250:]
        if len(self.accBuf) < 250:
            return None
        return np.array(self.accBuf[-250:], dtype=float)


def _gap_df():
    m_cols = [f"M{i:03d}" for i in range(125)]
    times = ["2022-01-01T00:00:00Z", "2022-01-01T00:00:05Z",
             # 30 s gap here (missing-data span, no filler rows)
             "2022-01-01T00:00:40Z", "2022-01-01T00:00:45Z"]
    rows = []
    for t in times:
        row = {"eventId": "E001", "type": 0, "userId": "u", "dataTime": t, "hr": 70}
        for mc in m_cols:
            row[mc] = 1000.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_df2training_data_restarts_buffer_at_gap():
    df = _gap_df()
    model = _TinyBufferModel()
    out, classes, used = nnTrainer.df2trainingData(df, model, return_row_indices=True)
    # dp0: cold (None); dp1: first full window; gap resets; dp2: cold (None);
    # dp3: first full post-gap window.
    assert used == [1, 3]
    assert len(out) == 2


def test_df2training_data_no_reset_without_gap():
    df = _gap_df()
    # contiguous 5 s spacing: no reset mid-event
    df.loc[2, 'dataTime'] = "2022-01-01T00:00:10Z"
    df.loc[3, 'dataTime'] = "2022-01-01T00:00:15Z"
    model = _TinyBufferModel()
    out, classes, used = nnTrainer.df2trainingData(df, model, return_row_indices=True)
    assert used == [1, 2, 3]


# ---------------------------------------------------------------------------
# flattenData: gaps are discontinuities, never synthetic rows
# ---------------------------------------------------------------------------

def _gap_event():
    def dp(t, val=1000.0):
        return {'id': 1, 'dataTime': t, 'hr': 70, 'o2Sat': 98,
                'rawData': [val] * 125, 'rawData3D': [0.0, 0.0, val] * 125,
                'specPower': 1.0, 'roiPower': 1.0, 'alarmState': 0}
    return {'id': 'G1', 'userId': 'u', 'type': 'False Alarm', 'subType': '',
            'datapoints': [dp("2022-01-01T00:00:00Z"),
                           dp("2022-01-01T00:00:05Z"),
                           # 35 s gap: previously filled with zero rows
                           dp("2022-01-01T00:00:45Z")]}


def test_flatten_gaps_produce_no_synthetic_rows():
    rows = flattenData.process_event_obj(_gap_event(), debug=False, validate=True,
                                         config=None)
    # only the 3 real datapoints; header order: eventId,userId,typeStr,type,
    # dataTime,osdAlarmState,osdSpecPower,osdRoiPower,hr,o2sat,M000.. + XYZ
    assert len(rows) == 3
    m_start = 10
    for row in rows:
        mags = row[m_start:m_start + 125]
        assert not all(v == 0 for v in mags), "synthetic zero row emitted"
    times = [r[4] for r in rows]
    assert times[0] < times[1] < times[2]


def test_flatten_contiguous_event_unchanged():
    ev = _gap_event()
    ev['datapoints'] = ev['datapoints'][:2]
    rows = flattenData.process_event_obj(ev, debug=False, validate=True, config=None)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# alarm latency ignores warm-up datapoints
# ---------------------------------------------------------------------------

def test_latency_ignores_warm_datapoints():
    event_stats_df = pd.DataFrame([{
        'eventId': 'E1', 'true_label': 1, 'subType': 'Tonic-Clonic'}])
    df = pd.DataFrame({
        'eventId': ['E1'] * 3,
        'dataTime': ["2022-01-01 00:00:00", "2022-01-01 00:00:05",
                     "2022-01-01 00:00:10"],
        'is_warm': [True, False, False],
    })
    proba = np.array([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    details = {'E1': {'dataTime': "2022-01-01 00:00:00",
                      'seizureTimes': [0, 30],
                      'userId': 'u', 'subType': 'Tonic-Clonic'}}
    lat, per_event = nnTester._compute_alarm_latency(
        event_stats_df, df, proba, details, [0.5])
    # unmasked first crossing would be dp0 (latency 0); the warm dp0 must not
    # trigger, so the alarm is at dp2 -> 10 s after onset.
    assert per_event.loc[0, 'latency_th_0.5'] == 10.0
