import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_tools' / 'nnTraining2'))

import analyzeEventResults


def test_summarize_production_metrics_detects_three_consecutive_probabilities():
    df = pd.DataFrame([
        {
            'EventID': 1,
            'UserID': 10,
            'Type': 'Seizure',
            'SubType': 'Tonic-Clonic',
            'ActualLabel': 1,
            'ModelPrediction': 0,
            'MaxSeizureProbability': 0.75,
            'dp0': 0.05,
            'dp1': 0.60,
            'dp2': 0.70,
            'dp3': 0.80,
            'dp4': 0.10,
        },
        {
            'EventID': 2,
            'UserID': 11,
            'Type': 'False Alarm',
            'SubType': '',
            'ActualLabel': 0,
            'ModelPrediction': 1,
            'MaxSeizureProbability': 0.90,
            'dp0': 0.10,
            'dp1': 0.90,
            'dp2': 0.80,
            'dp3': 0.70,
            'dp4': 0.10,
        },
    ])

    summary = analyzeEventResults.summarize_production_metrics(df, threshold=0.5, consecutive_required=3)

    assert summary['tp'] == 1
    assert summary['fp'] == 1
    assert summary['fn'] == 0
    assert summary['event_prod_tp'] == 1
    assert summary['event_prod_fp'] == 1
