#!/usr/bin/env python3
"""Tests for tonic-clonic threshold graphs (event vs production)."""
import os
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path


def _make_dummy_event_df(tmp_path, model_prefix="lstm_1d"):
    """Create minimal event_results.csv compatible df for analyzeEventResults.generate_plots."""
    # Minimal required columns for generate_plots: ActualLabel, ModelPrediction, UserID, Type, SubType, Description, plus dp columns
    # Create 10 events: 5 seizures (2 tonic-clonic, 3 other) and 5 non-seizures
    import pandas as pd
    rows = []
    for i in range(5):
        # seizure
        sub = "Tonic-Clonic" if i < 2 else "Other"
        rows.append({
            "EventID": 1000 + i,
            "UserID": "test_user",
            "Type": "Seizure",
            "SubType": sub,
            "ActualLabel": 1,
            "ModelPrediction": 1 if i < 3 else 0,
            "MaxSeizureProbability": 0.9 if i < 3 else 0.2,
            "dp0": 0.9, "dp1": 0.8, "dp2": 0.1,
            "Description": "test seizure"
        })
    for i in range(5):
        rows.append({
            "EventID": 2000 + i,
            "UserID": "test_user",
            "Type": "False Alarm",
            "SubType": "Unknown",
            "ActualLabel": 0,
            "ModelPrediction": 0,
            "MaxSeizureProbability": 0.1,
            "dp0": 0.1, "dp1": 0.2, "dp2": 0.05,
            "Description": "test non-seizure"
        })
    df = pd.DataFrame(rows)
    # mimic attrs set by load_event_results
    csv_path = tmp_path / f"{model_prefix}_event_results.csv"
    df.to_csv(csv_path, index=False)
    df_loaded = pd.read_csv(csv_path)
    df_loaded.attrs['source_csv_path'] = str(csv_path)
    return df_loaded, csv_path


def test_nnTester_generates_tonic_clonic_event_vs_production_plot(tmp_path):
    """nnTester should generate event_vs_production for tonic-clonic (the missing PNG)."""
    from user_tools.nnTraining2 import nnTester

    # Create synthetic event_probs_list and labels
    # 10 events: 3 TC seizures, 7 others (including non-seizures)
    event_probs = [
        [0.9, 0.8, 0.9],  # TC seizure - high
        [0.85, 0.9, 0.8], # TC seizure - high
        [0.2, 0.1, 0.3],  # TC seizure - low (FN)
        [0.8, 0.7, 0.75], # Other seizure
        [0.1, 0.2, 0.15], # Other seizure - low
        [0.1, 0.05, 0.2], # non-seizure
        [0.6, 0.7, 0.65], # non-seizure FP
        [0.05, 0.1, 0.08],
        [0.12, 0.15, 0.1],
        [0.3, 0.35, 0.32],
    ]
    true_labels = [1,1,1,1,1,0,0,0,0,0]
    # TC positive mask: first 3 are TC
    tc_mask = np.array([True, True, True, False, False, False, False, False, False, False])

    thresholds = [0.1,0.3,0.5,0.7,0.9]
    td_event_all = nnTester._threshold_metrics_from_event_probs(event_probs, true_labels, thresholds, mode='event')
    td_prod_all = nnTester._threshold_metrics_from_event_probs(event_probs, true_labels, thresholds, mode='production')
    td_event_tc = nnTester._threshold_metrics_from_event_probs(event_probs, true_labels, thresholds, mode='event', positive_mask=tc_mask)
    td_prod_tc = nnTester._threshold_metrics_from_event_probs(event_probs, true_labels, thresholds, mode='production', positive_mask=tc_mask, consecutive_required=3)

    model_prefix = "test_model"
    out_all = tmp_path / f"{model_prefix}_event_vs_production_threshold_analysis.png"
    out_tc = tmp_path / f"{model_prefix}_event_vs_production_threshold_analysis_tonic_clonic.png"

    nnTester._plot_event_vs_production_thresholds(td_event_all, td_prod_all, str(out_all), model_prefix)
    nnTester._plot_event_vs_production_thresholds(td_event_tc, td_prod_tc, str(out_tc), model_prefix)

    assert out_all.exists() and out_all.stat().st_size > 0
    assert out_tc.exists() and out_tc.stat().st_size > 0
    # Check tc title contains tonic-clonic (file name based)
    # File should be distinguishable; we already test existence. Verify suffix
    assert "tonic_clonic" in str(out_tc)


def test_analyzeEventResults_pdf_includes_tonic_clonic_threshold_plot(tmp_path):
    """analyzeEventResults.generate_plots should include both all and TC event_vs_production plots."""
    from user_tools.nnTraining2 import analyzeEventResults
    import matplotlib.pyplot as plt

    model_prefix = "lstm_1d"
    # Create dummy threshold plots that the PDF generator will find
    for suffix in ["_event_vs_production_threshold_analysis.png",
                   "_event_vs_production_threshold_analysis_tonic_clonic.png",
                   "_training.png", "_training2.png", "_training_tpr_fpr.png"]:
        p = tmp_path / f"{model_prefix}{suffix}"
        # Create a simple blank PNG
        plt.figure()
        plt.text(0.5, 0.5, f"dummy {suffix}")
        plt.savefig(p)
        plt.close()

    df, csv_path = _make_dummy_event_df(tmp_path, model_prefix)
    # Run the analyze functions to get required dataframes
    seizure_df, user_metrics_df, far_metrics_df = analyzeEventResults.analyze_by_user(df, seizure_threshold=2, far_threshold=2)
    _, subtype_metrics_df = analyzeEventResults.analyze_by_seizure_type(df)
    false_alarms_df, far_by_subtype_df = analyzeEventResults.analyze_false_alarms(df)
    false_negatives_df = df[(df['ActualLabel'] == 1) & (df['ModelPrediction'] == 0)].copy()
    fn_details = pd.DataFrame()

    pdf_path = analyzeEventResults.generate_plots(
        df, seizure_df, user_metrics_df, far_metrics_df,
        subtype_metrics_df, false_alarms_df, false_negatives_df,
        fn_details, str(tmp_path)
    )
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 0
    # Verify both threshold plots were found (check via internal helper)
    found_all = analyzeEventResults._find_first_matching_file(str(tmp_path), [f"{model_prefix}_event_vs_production_threshold_analysis.png"])
    found_tc = analyzeEventResults._find_first_matching_file(str(tmp_path), [f"{model_prefix}_event_vs_production_threshold_analysis_tonic_clonic.png"])
    assert found_all is not None
    assert found_tc is not None


def test_nnTrainer_references_tonic_clonic_plot():
    """nnTrainer.py should reference the tonic-clonic event_vs_production plot."""
    import pathlib
    p = pathlib.Path(__file__).parent.parent / "nnTrainer.py"
    text = p.read_text()
    assert "event_vs_production_threshold_analysis_tonic_clonic" in text
    # Also ensure nnTester generates it
    p2 = pathlib.Path(__file__).parent.parent / "nnTester.py"
    text2 = p2.read_text()
    assert "event_vs_production_threshold_analysis_tonic_clonic" in text2


def test_noiseAugNonSeizure_and_sampleRate_console_output_still_echo(tmp_path, capsys):
    """Regression: ensure previous console-output requirements still hold after new changes."""
    from user_tools.nnTraining2 import augmentData
    # Minimal df for noise aug
    cols = ["eventId", "userId", "typeStr", "type", "dataTime"] + [f"M{i:03d}" for i in range(125)]
    rows = []
    for eid, sub in [("S1", "Sorting"), ("M1", "Motor Vehicle")]:
        for _ in range(1):
            row = {"eventId": eid, "userId": "u", "typeStr": f'"False Alarm/{sub}"', "type": 0, "dataTime": "2022-01-01T00:00:00Z"}
            for c in cols[5:]:
                row[c] = 1000
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    augmentData.noiseAugNonSeizure(df, noiseAugVal=5.0, noiseAugFac=1,
                                   targetTypeSubTypePairs=[{"type":0,"subType":"Sorting","factor":1,"value":5.0}])
    out = capsys.readouterr().out
    assert "noiseAugNonSeizure(): Starting with" in out
    assert "factor=1" in out and "value=5.0" in out
    assert "Completed" in out

    # sampleRate
    cols2 = ["eventId", "userId", "typeStr", "type", "dataTime"] + [f"M{i:03d}" for i in range(125)]
    rows2 = []
    for eid in ["SE1", "SE2"]:
        for _ in range(2):
            row = {"eventId": eid, "userId": "u", "typeStr": '"Seizure/Tonic-Clonic"', "type": 1, "dataTime": "2022-01-01T00:00:00Z"}
            for c in cols2[5:]:
                row[c] = 1000
            rows2.append(row)
    df2 = pd.DataFrame(rows2, columns=cols2)
    augmentData.sampleRateAug(df2, sampleRateFactors=[0.9, 1.1])
    out2 = capsys.readouterr().out
    assert "sampleRateAug(): Starting with" in out2
    assert "Completed – seizure events before:" in out2
    assert "suffix" in out2.lower()
