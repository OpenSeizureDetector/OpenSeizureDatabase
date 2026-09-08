#!/usr/bin/env python3

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from user_tools.nnTraining2 import nnTester
from user_tools.nnTraining2 import nnTrainer
from user_tools.nnTraining2.subtype_weighting import create_subtype_weighted_sampler


class _DummyModel:
    accel_input_mode = 'magnitude'

    def resetAccBuf(self):
        return None

    def dp2vector(self, dpDict, normalise=False):
        # Drop rows with very small signal to simulate filtered datapoints.
        if dpDict['rawData'][0] < 0.2:
            return None
        return np.array(dpDict['rawData'], dtype=float)


def _minimal_config(use_subtype_weighting=False, subtype_weights=None):
    return {
        'dataFileNames': {
            'trainFeaturesHistoryFileCsv': 'train.csv',
            'valDataFileCsv': 'val.csv',
            'testFeaturesHistoryFileCsv': 'test.csv',
        },
        'dataProcessing': {
            'validationProp': 0.15,
        },
        'modelConfig': {
            'framework': 'pytorch',
            'modelFname': 'lstm_1d',
            'modelClass': 'user_tools.nnTraining2.cnnLstmModel_torch.CnnLstmModelPyTorch',
            'batchSize': 32,
            'epochs': 3,
            'lrStart': 1e-4,
            'lrMin': 1e-6,
            'lrPeak': 5e-4,
            'lrMainEnd': 1e-4,
            'warmupSteps': 10,
            'mainSteps': 50,
            'cooldownSteps': 10,
            'totalTrainingSteps': 70,
            'evalEverySteps': 10,
            'useLrSchedule': True,
            'useAdamW': True,
            'useBalancedBatches': True,
            'useSubtypeWeighting': use_subtype_weighting,
            'subtypeWeights': subtype_weights or {},
        },
    }


def test_load_config_params_reads_subtype_weighting_keys():
    cfg = _minimal_config(
        use_subtype_weighting=True,
        subtype_weights={'Tonic-Clonic': 2.0, 'Aura': 0.4, 'Other': 1.0},
    )

    params = nnTrainer.load_config_params(cfg)

    assert params['use_subtype_weighting'] is True
    assert params['subtype_weights']['Tonic-Clonic'] == 2.0
    assert params['subtype_weights']['Aura'] == 0.4


def test_df2trainingData_returns_aligned_row_indices():
    df = pd.DataFrame(
        {
            'eventId': [1, 1, 2, 2],
            'type': [1, 1, 0, 0],
            'M000_t-0': [0.1, 0.3, 0.9, 0.05],
            'M001_t-0': [0.2, 0.4, 1.1, 0.10],
            'hr': [90, 91, 70, 69],
        }
    )

    vectors, labels, used_rows = nnTrainer.df2trainingData(df, _DummyModel(), return_row_indices=True)

    assert used_rows == [1, 2]
    assert len(vectors) == len(labels) == len(used_rows)
    assert labels == [1, 0]


def test_subtype_weighted_sampler_biases_tonic_clonic_over_aura():
    df = pd.DataFrame(
        {
            'eventId': [10, 11, 12, 13],
            'subType': ['Tonic-Clonic', 'Aura', 'Other', 'Other'],
        }
    )
    y = np.array([1, 1, 1, 0])

    sampler = create_subtype_weighted_sampler(
        df=df,
        y_values=y,
        subtype_weights={'Tonic-Clonic': 2.0, 'Aura': 0.4, 'Other': 1.0},
        debug=False,
    )

    weights = sampler.weights.detach().cpu().numpy()

    assert weights[0] > weights[1]
    assert weights[2] > weights[1]


def test_summary_text_includes_subtype_weighting_details():
    cfg = {
        'modelConfig': {
            'useSubtypeWeighting': True,
            'subtypeWeights': {'Tonic-Clonic': 2.0, 'Aura': 0.4, 'Other': 1.0},
        }
    }

    text = nnTester._format_subtype_weighting_summary(cfg)

    assert 'Subtype weighting enabled: True' in text
    assert 'Aura: 0.4' in text
    assert 'Tonic-Clonic: 2.0' in text
