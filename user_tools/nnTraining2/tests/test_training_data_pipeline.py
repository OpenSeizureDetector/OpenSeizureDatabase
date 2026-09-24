#!/usr/bin/env python3
"""
Generic tests for the training data pipeline.

Covers the hot path that previously caused OOM:
  augmentData.loadCsv -> df2trainingData -> cnnLstmModel dp2vector (750-sample LSTM buffer)

These tests do NOT require torch or tensorflow and are intended to run on
machines without GPU libraries. They lock the current pipeline semantics so
that any future memory/performance refactor (e.g. float32 early, vectorised
loop, streaming Dataset) can be proved to be output-equivalent.

Each test uses lightweight dummy models that replicate the buffer behaviour
of CnnLstmModelPyTorch without importing torch.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_tools.nnTraining2 import nnTrainer


# ---------------------------------------------------------------------------
# Dummy models — no torch dependency, same interface as nnModel subclasses
# ---------------------------------------------------------------------------

class DummyLstmMagnitude:
    """Mimics CnnLstmModelPyTorch magnitude mode: 750-sample buffer (6x125)."""
    accel_input_mode = 'magnitude'
    bufferSamples = 750  # 30 s @ 25 Hz

    def __init__(self):
        self.accBuf = []

    def resetAccBuf(self):
        self.accBuf = []

    def accData2vector(self, accData, normalise=False):
        self.accBuf.extend(accData)
        if len(self.accBuf) > self.bufferSamples:
            self.accBuf = self.accBuf[-self.bufferSamples:]
        if len(self.accBuf) < self.bufferSamples:
            return None
        vec = np.array(self.accBuf[-self.bufferSamples:], dtype=float) / 1000.0
        if normalise:
            std = vec.std()
            vec = (vec - vec.mean()) / std if std != 0 else vec - vec.mean()
        return vec.tolist()

    def dp2vector(self, dpObj, normalise=False):
        raw = dpObj.get('rawData', None)
        if raw is None:
            return None
        return self.accData2vector(raw, normalise)


class DummyLstmXYZ:
    """Mimics CnnLstmModelPyTorch xyz mode: 750 rows x 3 channels."""
    accel_input_mode = 'xyz'
    bufferSamples = 750

    def __init__(self):
        self.accBuf3D = []

    def resetAccBuf(self):
        self.accBuf3D = []

    def accData3D2vector(self, accData3D, normalise=False):
        arr = np.asarray(accData3D, dtype=float)
        if arr.ndim == 1:
            if len(arr) % 3 != 0:
                return None
            arr = arr.reshape(-1, 3)
        self.accBuf3D.extend(arr.tolist())
        if len(self.accBuf3D) > self.bufferSamples:
            self.accBuf3D = self.accBuf3D[-self.bufferSamples:]
        if len(self.accBuf3D) < self.bufferSamples:
            return None
        vec = np.array(self.accBuf3D[-self.bufferSamples:], dtype=float) / 1000.0
        return vec

    def dp2vector(self, dpObj, normalise=False):
        raw3d = dpObj.get('rawData3D', None)
        if raw3d is None:
            return None
        return self.accData3D2vector(raw3d, normalise)


class DummyFilterModel:
    """Returns None for low-amplitude rows to test filtering."""
    accel_input_mode = 'magnitude'

    def resetAccBuf(self):
        pass

    def dp2vector(self, dpObj, normalise=False):
        if dpObj['rawData'][0] < 0.2:
            return None
        return np.array(dpObj['rawData'], dtype=float)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mag_df(n_events=2, rows_per_event=10, start_val=1000, use_suffix=True, with_hr=True):
    """Create a synthetic trainFeatures-style DataFrame with M* cols."""
    suffix = "_t-0" if use_suffix else ""
    m_cols = [f"M{i:03d}{suffix}" for i in range(125)]
    cols = ["eventId", "type", "userId", "dataTime"] + m_cols
    if with_hr:
        cols.append("hr")
    rows = []
    eid = 0
    for e in range(n_events):
        event_id = f"E{e:03d}"
        typ = 1 if e % 2 == 0 else 0
        for r in range(rows_per_event):
            row = {
                "eventId": event_id,
                "type": typ,
                "userId": "test_user",
                "dataTime": f"2022-01-01T00:00:{r:02d}Z",
            }
            # Distinct value per row so we can detect cross-event leakage
            base = float(start_val + e * 10000 + r * 100)
            for mc in m_cols:
                row[mc] = base
            if with_hr:
                row["hr"] = 70 + r
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    return df


def _make_xyz_df(n_events=1, rows_per_event=6, with_hr=False):
    """Create DataFrame with M*, X*, Y*, Z* cols for xyz mode."""
    m_cols = [f"M{i:03d}_t-0" for i in range(125)]
    x_cols = [f"X{i:03d}_t-0" for i in range(125)]
    y_cols = [f"Y{i:03d}_t-0" for i in range(125)]
    z_cols = [f"Z{i:03d}_t-0" for i in range(125)]
    cols = ["eventId", "type", "userId", "dataTime"] + m_cols + x_cols + y_cols + z_cols
    if with_hr:
        cols.append("hr")
    rows = []
    for e in range(n_events):
        event_id = f"EX{e:03d}"
        typ = 1
        for r in range(rows_per_event):
            row = {"eventId": event_id, "type": typ, "userId": "u", "dataTime": "2022-01-01T00:00:00Z"}
            base = float(1000 + r * 10)
            for mc in m_cols:
                row[mc] = base
            for xc in x_cols:
                row[xc] = base + 1
            for yc in y_cols:
                row[yc] = base + 2
            for zc in z_cols:
                row[zc] = base + 3
            if with_hr:
                row["hr"] = 80
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lstm_magnitude_750_window_stitching():
    """
    LSTM magnitude model needs 6 consecutive 125-sample rows (750) per vector.
    For 2 events x 10 rows each, expect (10-5)=5 vectors per event = 10 total.
    Also checks output shape and /1000 scaling.
    """
    df = _make_mag_df(n_events=2, rows_per_event=10, start_val=1000)
    model = DummyLstmMagnitude()
    x, y = nnTrainer.df2trainingData(df, model)
    # 5 valid windows per event (first 5 rows per event are buffering)
    assert len(x) == 10
    assert len(y) == 10
    # Each vector length 750, values are mG/1000
    assert len(x[0]) == 750
    # First valid vector for E000 should be rows 0-5 (values 1000,1100,...1500)/1000
    # After reset for E001, second event's vectors isolated
    assert np.isclose(x[0][0], 1.0)  # 1000/1000
    assert np.isclose(x[5][0], 1.0 + 10000/1000.0)  # start of second event base
    # Labels: E000 type 1, E001 type 0
    assert y[:5] == [1]*5
    assert y[5:] == [0]*5


def test_event_boundary_resets_buffer_no_cross_leakage():
    """
    Buffer must reset on eventId change. An event with distinctive values
    must not leak into the next event's vectors.
    """
    # Event A rows all 100, event B rows all 9999
    df_a = _make_mag_df(n_events=1, rows_per_event=10, start_val=100)
    df_b = _make_mag_df(n_events=1, rows_per_event=10, start_val=9999)
    # Rename second event id to be distinct sequential
    df_b["eventId"] = "E001"
    df_b["type"] = 0
    df = pd.concat([df_a, df_b], ignore_index=True)

    model = DummyLstmMagnitude()
    x, y, used = nnTrainer.df2trainingData(df, model, return_row_indices=True)

    assert len(x) == 10  # 5+5
    # Last vector of first event should contain only 100-series values (<2.0 after /1000)
    assert max(x[4]) < 2.0
    # First vector of second event should contain only 9999-series values (>9.0)
    assert min(x[5]) > 9.0
    # Used indices: first 5 rows of each event are dropped
    assert used == [5, 6, 7, 8, 9, 15, 16, 17, 18, 19]


def test_nan_handling_replaces_with_zero():
    """
    Rows with NaN in M* should be replaced with 0 (via nan_to_num) and not
    propagate NaN into xTrain — mimics stale 3D handling.
    """
    df = _make_mag_df(n_events=1, rows_per_event=6, start_val=1000)
    # Inject NaNs into first row's first 10 M columns
    df.loc[0, "M000_t-0"] = np.nan
    df.loc[0, "M001_t-0"] = float("inf")
    df.loc[1, "M002_t-0"] = float("-inf")
    model = DummyLstmMagnitude()
    x, y = nnTrainer.df2trainingData(df, model)
    # 6 rows -> 1 valid vector (rows 0-5)
    assert len(x) == 1
    arr = np.array(x[0], dtype=float)
    assert not np.isnan(arr).any(), "NaN propagated into output"
    assert not np.isinf(arr).any(), "Inf propagated into output"
    # The corrupted positions should be 0 after nan_to_num
    assert arr[0] == 0.0  # M000 of row 0 -> 0
    assert arr[1] == 0.0


def test_float32_dtype_parity():
    """
    Reading the same CSV as float64 vs float32 should produce vectors that
    agree within 1e-6 after /1000 scaling. Ensures early float32 is safe.
    """
    df64 = _make_mag_df(n_events=1, rows_per_event=6, start_val=1234)
    # Simulate float32 ingestion by casting M cols to float32 then back
    df32 = df64.copy()
    m_cols = [c for c in df32.columns if c.startswith("M")]
    for c in m_cols:
        df32[c] = df32[c].astype(np.float32)

    model64 = DummyLstmMagnitude()
    model32 = DummyLstmMagnitude()
    x64, _ = nnTrainer.df2trainingData(df64, model64)
    x32, _ = nnTrainer.df2trainingData(df32, model32)

    assert len(x64) == len(x32) == 1
    assert np.allclose(np.array(x64[0]), np.array(x32[0]), atol=1e-6)


def test_xyz_mode_produces_750x3_vectors():
    """XYZ mode should produce (750,3) arrays via 3*125 per row, 6 rows = 750."""
    df = _make_xyz_df(n_events=1, rows_per_event=6)
    model = DummyLstmXYZ()
    x, y = nnTrainer.df2trainingData(df, model)
    assert len(x) == 1
    arr = np.array(x[0])
    assert arr.shape == (750, 3)
    # Values scaled /1000
    assert np.isclose(arr[0, 0], 1.001)  # X base 1001/1000
    assert np.isclose(arr[0, 1], 1.002)
    assert np.isclose(arr[0, 2], 1.003)


def test_xyz_mode_missing_columns_raises():
    """Requesting xyz mode without X/Y/Z cols should raise ValueError."""
    df = _make_mag_df(n_events=1, rows_per_event=6)
    model = DummyLstmXYZ()
    try:
        nnTrainer.df2trainingData(df, model)
        assert False, "Expected ValueError for missing XYZ cols"
    except ValueError as e:
        assert "XYZ" in str(e)


def test_return_row_indices_alignment():
    """
    return_row_indices=True must give indices that align with labels and
    can be used to slice the original DataFrame for subtype weighting.
    """
    df = _make_mag_df(n_events=2, rows_per_event=8, start_val=500)
    # Use filter model that drops first value <0.2 — but our base is 500 so none dropped
    # Instead use DummyFilterModel on a tiny base
    df_small = pd.DataFrame(
        {
            "eventId": [1, 1, 2, 2],
            "type": [1, 1, 0, 0],
            "M000_t-0": [0.05, 0.3, 0.9, 0.05],
            "M001_t-0": [0.10, 0.4, 1.1, 0.10],
            "hr": [90, 91, 70, 69],
        }
    )
    vectors, labels, used_rows = nnTrainer.df2trainingData(df_small, DummyFilterModel(), return_row_indices=True)
    assert used_rows == [1, 2]
    assert labels == [1, 0]
    # Alignment check: slicing original df by used_rows gives correct types
    assert list(df_small.iloc[used_rows]["type"]) == labels


def test_hr_missing_handled_gracefully():
    """DataFrames without hr column should not error; hr defaults to None."""
    df = _make_mag_df(n_events=1, rows_per_event=6, with_hr=False)
    assert "hr" not in df.columns
    model = DummyLstmMagnitude()
    x, y = nnTrainer.df2trainingData(df, model)
    assert len(x) == 1  # still produces vector


def test_suffix_variants_both_accepted():
    """
    Both M000_t-0 and M000 (no suffix) naming must be accepted — mirrors
    flattenData vs feature CSV conventions (nnTrainer.py:84-95).
    """
    df_suffix = _make_mag_df(n_events=1, rows_per_event=6, use_suffix=True)
    df_no_suffix = _make_mag_df(n_events=1, rows_per_event=6, use_suffix=False)
    # Rename cols to verify _collect_axis_cols handles both
    m1 = DummyLstmMagnitude()
    m2 = DummyLstmMagnitude()
    x1, _ = nnTrainer.df2trainingData(df_suffix, m1)
    x2, _ = nnTrainer.df2trainingData(df_no_suffix, m2)
    assert len(x1) == len(x2) == 1
    assert np.allclose(x1[0], x2[0])


def test_insufficient_rows_per_event_yields_no_vectors():
    """An event with fewer than 6 rows should produce zero vectors (LSTM)."""
    df = _make_mag_df(n_events=1, rows_per_event=3, start_val=1000)
    model = DummyLstmMagnitude()
    x, y = nnTrainer.df2trainingData(df, model)
    assert len(x) == 0
    assert len(y) == 0


def test_single_row_filter_model_drops_correctly():
    """When dp2vector returns None for a row, that row is excluded and indices reflect it."""
    df = pd.DataFrame(
        {
            "eventId": ["E1", "E1", "E1"],
            "type": [1, 1, 1],
            "M000_t-0": [0.05, 0.5, 0.5],
            "M001_t-0": [0.05, 0.5, 0.5],
            "hr": [70, 70, 70],
        }
    )
    x, y, used = nnTrainer.df2trainingData(df, DummyFilterModel(), return_row_indices=True)
    # Row 0 drops (0.05 <0.2), rows 1,2 keep
    assert used == [1, 2]
    assert len(x) == 2
