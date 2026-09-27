#!/usr/bin/env python3
"""Tests for DC-offset augmentation (dcOffsetAug) and userAug scope.

dcOffsetAug adds per-event random sensor-bias shifts to EVERY event (seizure
and non-seizure alike) so the model cannot use absolute DC level as a seizure
cue. userAug must only ever duplicate seizure events (verified here because
several users contribute little usable false-alarm data).
"""
import os
import glob
import json
import numpy as np
import pandas as pd

from user_tools.nnTraining2 import augmentData


def _make_df(events, with_3d=False, base_key="baseVal"):
    """Minimal flattened-CSV dataframe.

    events: list of dicts {eventId, type, nDp, baseVal}.
    """
    cols = ["eventId", "userId", "typeStr", "type", "dataTime"]
    m_cols = [f"M{i:03d}" for i in range(125)]
    cols.extend(m_cols)
    x_cols = [f"X{i:03d}" for i in range(125)] if with_3d else []
    y_cols = [f"Y{i:03d}" for i in range(125)] if with_3d else []
    z_cols = [f"Z{i:03d}" for i in range(125)] if with_3d else []
    cols.extend(x_cols + y_cols + z_cols)
    rows = []
    for ev in events:
        eid = str(ev["eventId"])
        typ = ev["type"]
        typeStr = "Seizure/Other" if typ == 1 else "False Alarm/Sorting"
        for _ in range(ev.get("nDp", 2)):
            row = {"eventId": eid, "userId": ev.get("userId", "u1"),
                   "typeStr": typeStr, "type": typ,
                   "dataTime": "2022-01-01T00:00:00Z"}
            base = ev.get(base_key, 1000.0)
            for mc in m_cols:
                row[mc] = base
            if with_3d:
                for xc in x_cols:
                    row[xc] = 100.0
                for yc in y_cols:
                    row[yc] = 200.0
                for zc in z_cols:
                    row[zc] = 900.0
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _mag(df, eid):
    m_cols = [f"M{i:03d}" for i in range(125)]
    return df[df["eventId"] == eid][m_cols].to_numpy(dtype=float)


def test_dc_offset_augments_both_classes():
    df = _make_df([
        {"eventId": "S1", "type": 1, "nDp": 2},
        {"eventId": "N1", "type": 0, "nDp": 3},
    ])
    out = augmentData.dcOffsetAug(df, 50.0, 1, config={"randomSeed": 42})
    counts = out.groupby("eventId").size().to_dict()
    # originals kept, one shifted copy each - including non-seizure
    assert counts == {"S1": 2, "N1": 3, "S1-dc1": 2, "N1-dc1": 3}
    # labels preserved on copies
    assert set(out[out["eventId"] == "S1-dc1"]["type"]) == {1}
    assert set(out[out["eventId"] == "N1-dc1"]["type"]) == {0}


def test_dc_offset_constant_per_event_within_band():
    df = _make_df([
        {"eventId": "S1", "type": 1, "nDp": 2, "baseVal": 1000.0},
        {"eventId": "S2", "type": 1, "nDp": 2, "baseVal": 1030.0},
        {"eventId": "N1", "type": 0, "nDp": 2, "baseVal": 985.0},
        {"eventId": "N2", "type": 0, "nDp": 2, "baseVal": 1010.0},
    ])
    out = augmentData.dcOffsetAug(df, 50.0, 1, config={"randomSeed": 7})
    offsets = []
    for eid in ("S1", "S2", "N1", "N2"):
        diff = _mag(out, f"{eid}-dc1") - _mag(out, eid)
        # one offset per event: identical across all dps and samples
        assert np.allclose(diff, diff.flat[0])
        offsets.append(float(diff.flat[0]))
        assert abs(diff.flat[0]) <= 50.0
    # offsets vary across events (not a single global shift)
    assert len(set(np.round(offsets, 6))) > 1


def test_dc_offset_clips_at_zero():
    df = _make_df([{"eventId": "S1", "type": 1, "nDp": 2, "baseVal": 5.0}])
    out = augmentData.dcOffsetAug(df, 50.0, 2, config={"randomSeed": 3})
    assert _mag(out, "S1-dc1").min() >= 0.0
    assert _mag(out, "S1-dc2").min() >= 0.0


def test_dc_offset_3d_bias_vector_consistent():
    df = _make_df([{"eventId": "S1", "type": 1, "nDp": 2}], with_3d=True)
    out = augmentData.dcOffsetAug(df, 50.0, 1, config={"randomSeed": 11})
    x_cols = [f"X{i:03d}" for i in range(125)]
    y_cols = [f"Y{i:03d}" for i in range(125)]
    z_cols = [f"Z{i:03d}" for i in range(125)]
    g = out[out["eventId"] == "S1-dc1"]
    x = g[x_cols].to_numpy(dtype=float)
    y = g[y_cols].to_numpy(dtype=float)
    z = g[z_cols].to_numpy(dtype=float)
    # XYZ actually changed (bias vector applied)
    assert not np.allclose(x, 100.0)
    # magnitude recomputed from shifted XYZ
    recomputed = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    assert np.allclose(_mag(out, "S1-dc1"), recomputed, atol=1e-6)
    # bias magnitude bounded by the configured max
    shift = np.stack([x - 100.0, y - 200.0, z - 900.0], axis=-1)
    assert np.linalg.norm(shift, axis=-1).max() <= 50.0 + 1e-6


def test_dc_offset_deterministic_with_seed():
    df = _make_df([
        {"eventId": "S1", "type": 1, "nDp": 3},
        {"eventId": "N1", "type": 0, "nDp": 3},
    ])
    cfg = {"randomSeed": 123}
    a = augmentData.dcOffsetAug(df, 50.0, 2, config=cfg)
    b = augmentData.dcOffsetAug(df, 50.0, 2, config=cfg)
    pd.testing.assert_frame_equal(a, b)


def test_dc_offset_disabled_returns_input():
    df = _make_df([{"eventId": "S1", "type": 1, "nDp": 2}])
    assert len(augmentData.dcOffsetAug(df, 50.0, 0, config={"randomSeed": 1})) == len(df)
    assert len(augmentData.dcOffsetAug(df, 0.0, 2, config={"randomSeed": 1})) == len(df)


def test_user_aug_only_touches_seizures():
    df = _make_df([
        {"eventId": "S1", "type": 1, "nDp": 2, "userId": "uA", "baseVal": 1000.0},
        {"eventId": "S2", "type": 1, "nDp": 2, "userId": "uA", "baseVal": 1100.0},
        {"eventId": "S3", "type": 1, "nDp": 2, "userId": "uB", "baseVal": 1200.0},
        {"eventId": "N1", "type": 0, "nDp": 3, "userId": "uC", "baseVal": 900.0},
    ])
    before_nonseiz = df[df["type"] == 0].reset_index(drop=True)
    cfg = {"randomSeed": 42, "dataProcessing": {"userAugmentationThreshold": 1}}
    out = augmentData.userAug(df, config=cfg)
    # non-seizure rows byte-identical: same count and same values
    after_nonseiz = out[out["type"] == 0].reset_index(drop=True)
    assert len(after_nonseiz) == len(before_nonseiz) == 3
    pd.testing.assert_frame_equal(after_nonseiz, before_nonseiz)
    # seizure side balanced across users: uA has 2, uB has 1 -> 1 duplicate
    seiz_counts = out[out["type"] == 1].groupby("eventId").size().to_dict()
    assert sum(seiz_counts.values()) == 8  # 3 originals x2 dps + 1 dup x2 dps
    assert any(str(eid).startswith("S3-dup") for eid in seiz_counts)


def test_dc_offset_config_keys_present_everywhere():
    cfg_dir = os.path.join(os.path.dirname(__file__), '..')
    paths = sorted(glob.glob(os.path.join(cfg_dir, 'nnConfig*.json')))
    assert len(paths) >= 10
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        dp = d.get('dataProcessing', {})
        for key in ('dcOffsetAugmentation', 'dcOffsetAugmentationFactor',
                    'dcOffsetAugmentationMax'):
            assert key in dp, f"{os.path.basename(p)} missing {key}"
    # dcNormalisation is now a surrogate-only rescale preserving the 1000 mg
    # DC offset, so dcOffsetAugmentation remains meaningful alongside it; the
    # rerun target keeps the keys with DC-offset disabled and normalisation on.
    with open(os.path.join(cfg_dir, 'nnConfig_lstm_1d_45s.json')) as f:
        d45 = json.load(f)['dataProcessing']
    assert d45['dcOffsetAugmentation'] is False
    assert d45['dcNormalisation'] is True
