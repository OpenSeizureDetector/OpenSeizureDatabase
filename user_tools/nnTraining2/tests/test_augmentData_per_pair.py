#!/usr/bin/env python3
"""Tests for per-pair noiseAugmentationNonSeizure factor/value (option 1).

Covers hard-negative mining style selective augmentation where each
type/subType pair can specify its own ``factor`` and ``value``.
"""
import os
import numpy as np
import pandas as pd
import pytest

from user_tools.nnTraining2 import augmentData


def _make_df(events):
    """Create a minimal flattened CSV dataframe.

    events: list of dicts {eventId:str, type:int, subType:str, typeStr:str|None, nDp:int}
    Each datapoint gets M000-M124 values (125 columns) and minimal metadata.
    """
    cols = ["eventId", "userId", "typeStr", "type", "dataTime"]
    m_cols = [f"M{i:03d}" for i in range(125)]
    cols.extend(m_cols)
    rows = []
    for ev in events:
        eid = str(ev["eventId"])
        typ = ev["type"]
        sub = ev.get("subType", "")
        typeStr = ev.get("typeStr")
        if typeStr is None:
            # emulate flattenData's typeStr: "Seizure/Other" or "False Alarm/Sorting"
            type_name = "Seizure" if typ == 1 else "False Alarm"
            typeStr = f'"{type_name}/{sub}"' if sub else type_name
        nDp = ev.get("nDp", 2)
        for _ in range(nDp):
            row = {
                "eventId": eid,
                "userId": "test_user",
                "typeStr": typeStr,
                "type": typ,
                "dataTime": "2022-01-01T00:00:00Z",
            }
            # Fill magnitude values with deterministic constant per event for value tests
            base = ev.get("baseVal", 1000.0)
            for mc in m_cols:
                row[mc] = base
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    return df


def _count_events(df):
    """Return dict eventId -> n rows"""
    return df.groupby("eventId").size().to_dict()


def test_normalise_parses_factor_value_aliases():
    pairs = [
        {"type": 0, "subType": "Sorting", "factor": 5, "value": 8.0},
        {"type": 0, "subType": "Motor Vehicle", "noiseAugmentationFactor": "3", "noiseAugmentationValue": "12.5"},
        {"type": 0, "subType": "Unknown", "noiseFactor": 2, "noiseValue": 4},
    ]
    norm = augmentData._normalise_type_subtype_pairs(pairs)
    assert len(norm) == 3
    assert norm[0]["factor"] == 5 and norm[0]["value"] == 8.0
    assert norm[1]["factor"] == 3 and norm[1]["value"] == 12.5
    assert norm[2]["factor"] == 2 and norm[2]["value"] == 4.0


def test_normalise_missing_factor_value_is_none():
    pairs = [{"type": 0, "subType": "Sorting"}]
    norm = augmentData._normalise_type_subtype_pairs(pairs)
    assert norm[0]["factor"] is None
    assert norm[0]["value"] is None


def test_extract_subtype_from_typestr_fallback():
    # DataFrame without subType column, only typeStr
    df = _make_df([{"eventId": "E1", "type": 0, "subType": "Sorting", "nDp": 1}])
    # Remove subType if present, ensure only typeStr exists
    assert "subType" not in df.columns
    row = df.iloc[0]
    sub = augmentData._extract_subtype_from_row(row, None)
    assert sub == "Sorting"
    # Direct subType column should take precedence
    df2 = df.copy()
    df2["subType"] = "Motor Vehicle"
    row2 = df2.iloc[0]
    sub2 = augmentData._extract_subtype_from_row(row2, "subType")
    assert sub2 == "Motor Vehicle"


def test_noiseAugNonSeizure_global_factor():
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 2},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 2},
        {"eventId": "U1", "type": 0, "subType": "Unknown", "nDp": 2},
        {"eventId": "SE1", "type": 1, "subType": "Tonic-Clonic", "nDp": 2},
    ])
    orig_counts = _count_events(df)
    # Only Sorting and Motor selected, global factor 2
    out = augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=2,
                                         targetTypeSubTypePairs=[
                                             {"type": 0, "subType": "Sorting"},
                                             {"type": 0, "subType": "Motor Vehicle"},
                                         ])
    counts = _count_events(out)
    # Original events kept
    assert counts["S1"] == orig_counts["S1"]  # original rows kept? Actually aug adds new eventIds
    # Check number of distinct eventIds: S1 original + 2 augmented, M1 +2, U1 unchanged, SE1 unchanged
    assert "S1-nns1" in counts and "S1-nns2" in counts
    assert "M1-nns1" in counts and "M1-nns2" in counts
    assert "U1-nns1" not in counts
    assert "SE1" in counts
    # Total rows: original 8 + 2*2*2 (2 events *2 copies*2 rows each) = 16
    assert len(out) == 16
    # Total distinct events: 4 original + 4 augmented =8
    assert len(counts) == 8


def test_per_pair_factor_overrides_global():
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 1},
        {"eventId": "U1", "type": 0, "subType": "Unknown", "nDp": 1},
    ])
    out = augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=2,
                                         targetTypeSubTypePairs=[
                                             {"type": 0, "subType": "Sorting", "factor": 5},
                                             {"type": 0, "subType": "Motor Vehicle", "factor": 1},
                                             {"type": 0, "subType": "Unknown", "factor": 0},
                                         ])
    counts = _count_events(out)
    # Sorting gets 5 copies
    for i in range(1, 6):
        assert f"S1-nns{i}" in counts
    assert "S1-nns6" not in counts
    # Motor gets 1
    assert "M1-nns1" in counts
    assert "M1-nns2" not in counts
    # Unknown suppressed even though global 2
    assert "U1-nns1" not in counts
    # total events = 3 original +5+1 =9
    assert len(counts) == 9
    # total rows = 3 +5+1 =9 rows (1 dp each)
    assert len(out) == 9


def test_per_pair_value_overrides():
    # Two events with same base value 1000, different per-pair noise values
    # value 0 -> augmented rows identical to original; value large -> rows differ
    np.random.seed(0)
    df_zero = _make_df([{"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1, "baseVal": 1000.0}])
    out_zero = augmentData.noiseAugNonSeizure(df_zero, noiseAugVal=5.0, noiseAugFac=1,
                                              targetTypeSubTypePairs=[{"type": 0, "subType": "Sorting", "value": 0.0}])
    # Find augmented row
    aug_row_zero = out_zero[out_zero["eventId"] == "S1-nns1"].iloc[0]
    # All M values should be exactly 1000 (noise 0)
    m_vals_zero = pd.to_numeric(aug_row_zero[[f"M{i:03d}" for i in range(125)]], errors='coerce').values.astype(float)
    assert np.allclose(m_vals_zero, 1000.0)

    np.random.seed(0)
    df_big = _make_df([{"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1, "baseVal": 1000.0}])
    out_big = augmentData.noiseAugNonSeizure(df_big, noiseAugVal=0.0, noiseAugFac=1,
                                             targetTypeSubTypePairs=[{"type": 0, "subType": "Sorting", "value": 10.0}])
    aug_row_big = out_big[out_big["eventId"] == "S1-nns1"].iloc[0]
    m_vals_big = pd.to_numeric(aug_row_big[[f"M{i:03d}" for i in range(125)]], errors='coerce').values.astype(float)
    # With value 10, augmented values should differ from 1000 and have variance
    assert not np.allclose(m_vals_big, 1000.0)
    assert np.std(m_vals_big - 1000.0) > 2.0


def test_mixed_global_and_per_pair():
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 1},
    ])
    # Global factor 2, but Sorting overrides to 4, Motor has no override -> uses global 2
    out = augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=2,
                                         targetTypeSubTypePairs=[
                                             {"type": 0, "subType": "Sorting", "factor": 4},
                                             {"type": 0, "subType": "Motor Vehicle"},
                                         ])
    counts = _count_events(out)
    assert sum(1 for k in counts if k.startswith("S1-nns")) == 4
    assert sum(1 for k in counts if k.startswith("M1-nns")) == 2


def test_global_zero_but_per_pair_positive():
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 1},
    ])
    # Global 0, but Sorting per-pair 3 should still augment
    out = augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=0,
                                         targetTypeSubTypePairs=[
                                             {"type": 0, "subType": "Sorting", "factor": 3},
                                             {"type": 0, "subType": "Motor Vehicle", "factor": 0},
                                         ])
    counts = _count_events(out)
    assert sum(1 for k in counts if k.startswith("S1-nns")) == 3
    assert sum(1 for k in counts if k.startswith("M1-nns")) == 0
    assert len(out) == 2 + 3  # 2 original +3 augmented


def test_backward_compat_no_per_pair_keys():
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 1},
    ])
    # Old config without factor/value keys should still use global factor
    out = augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=2,
                                         targetTypeSubTypePairs=[{"type": 0, "subType": "Sorting"}])
    counts = _count_events(out)
    assert sum(1 for k in counts if k.startswith("S1-nns")) == 2


def test_augmentSeizureData_with_per_pair_config(tmp_path):
    # End-to-end via augmentSeizureData reading/writing CSV
    # Create train CSV
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 2},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 2},
        {"eventId": "SE1", "type": 1, "subType": "Tonic-Clonic", "nDp": 2},
    ])
    train_csv = tmp_path / "trainData.csv"
    train_aug_csv = tmp_path / "trainDataAugmented.csv"
    df.to_csv(train_csv, index=False)

    configObj = {
        "dataFileNames": {
            "trainDataFileCsv": str(train_csv.name),
            "trainAugmentedFileCsv": str(train_aug_csv.name),
        },
        "dataProcessing": {
            "noiseAugmentation": False,
            "noiseAugmentationFactor": 0,
            "noiseAugmentationValue": 0,
            "phaseAugmentation": False,
            "phaseAugmentationStep": 5,
            "sampleRateAugmentation": False,
            "sampleRateAugmentationFactors": [],
            "userAugmentation": False,
            "noiseAugmentationNonSeizure": True,
            "noiseAugmentationNonSeizureFactor": 1,  # global default 1
            "noiseAugmentationNonSeizureValue": 5.0,
            "noiseAugmentationNonSeizurePairs": [
                {"type": 0, "subType": "Sorting", "factor": 3, "value": 2.0},
                {"type": 0, "subType": "Motor Vehicle", "factor": 1, "value": 10.0},
            ],
            "oversample": "none",
            "undersample": "none",
        }
    }
    # augmentSeizureData expects dataDir to contain trainData.csv
    augmentData.augmentSeizureData(configObj, dataDir=str(tmp_path), debug=False)
    assert train_aug_csv.exists()
    out_df = pd.read_csv(train_aug_csv)
    counts = _count_events(out_df)
    # Sorting should have 3 augmented events, Motor 1, originals kept
    assert sum(1 for k in counts if str(k).startswith("S1-nns")) == 3
    assert sum(1 for k in counts if str(k).startswith("M1-nns")) == 1
    # Seizure event unchanged
    assert "SE1" in counts


def test_noiseAugNonSeizure_console_output(capsys):
    df = _make_df([
        {"eventId": "S1", "type": 0, "subType": "Sorting", "nDp": 2},
        {"eventId": "M1", "type": 0, "subType": "Motor Vehicle", "nDp": 1},
        {"eventId": "U1", "type": 0, "subType": "Unknown", "nDp": 1},
    ])
    augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=2,
                                   targetTypeSubTypePairs=[
                                       {"type": 0, "subType": "Sorting", "factor": 3, "value": 7.5},
                                       {"type": 0, "subType": "Motor Vehicle", "factor": 1, "value": 10.0},
                                       {"type": 0, "subType": "Unknown", "factor": 0},
                                   ])
    out = capsys.readouterr().out
    # Should echo global factor/value and per-pair details including matched counts and values
    assert "noiseAugNonSeizure(): Starting with" in out
    assert "3 non-seizure events" in out
    assert "Global factor=2" in out
    assert "value=5.0" in out
    assert "subtype='sorting'" in out.lower()  # case-insensitive check via lower()
    assert "factor=3" in out and "value=7.5" in out
    assert "factor=1" in out and "value=10.0" in out
    assert "matched 1 event" in out
    assert "Suffix for augmented events: '-nns" in out
    assert "Completed" in out and "non-seizure events before: 3, after:" in out
    assert "suffix: '-nns" in out.lower() or "augmented eventid suffix" in out.lower()


def test_sampleRateAug_console_output(capsys):
    df = _make_df([
        {"eventId": "SE1", "type": 1, "subType": "Tonic-Clonic", "nDp": 3},
        {"eventId": "SE2", "type": 1, "subType": "Other", "nDp": 2},
        {"eventId": "N1", "type": 0, "subType": "Unknown", "nDp": 2},
    ])
    out_df = augmentData.sampleRateAug(df, sampleRateFactors=[0.9, 1.1, 1.15])
    out = capsys.readouterr().out
    assert "sampleRateAug(): Starting with 2 seizure events" in out
    assert "datapoints" in out.lower()
    assert "Sample-rate factors: [0.9, 1.1, 1.15]" in out or "0.9" in out
    assert "suffix" in out.lower()
    assert "-sr0p900" in out or "-sr0p9" in out
    assert "-sr1p100" in out or "-sr1p1" in out
    assert "-sr1p150" in out or "-sr1p15" in out
    assert "Completed – seizure events before: 2, after:" in out
    assert "added" in out.lower() and "events" in out.lower()
    assert "suffix: '-sr" in out.lower() or "suffix: '-sr" in out
    # Verify suffix is actually used in output df
    counts = _count_events(out_df)
    assert any("-sr" in k for k in counts.keys())

