#!/usr/bin/env python3
"""
Regression tests for generic .pt -> .pte export inputs.

The converter (convertPt2Pte) must not assume a fixed input shape: each model
wrapper declares its own trace inputs via export_example_inputs(), in the exact
(batch, channels, length) layout the .pte runtime is fed at inference time.
These tests lock that contract for every torch wrapper (DeepEpiCNN, CNN-LSTM
magnitude + xyz, arbitrary window lengths) without running the slow
torch.export / flatc steps.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

torch = pytest.importorskip("torch")

from user_tools.nnTraining2 import nnModel
from user_tools.nnTraining2 import cnnLstmModel_torch
from user_tools.nnTraining2 import deepEpiCnnModel_torch


def test_base_model_has_no_export_layout():
    m = nnModel.NnModel(configObj=None, debug=False)
    assert m.export_example_inputs() is None


def test_cnn_lstm_45s_export_inputs():
    cfg = {'sampleFreq': 25, 'cnnWindowSeconds': 1.0, 'lstmWindowSeconds': 45.0}
    m = cnnLstmModel_torch.CnnLstmModelPyTorch(cfg, debug=False)
    (x,) = m.export_example_inputs()
    assert tuple(x.shape) == (1, 1, 1125)
    assert x.dtype == torch.float32


def test_cnn_lstm_30s_export_inputs():
    cfg = {'sampleFreq': 25, 'cnnWindowSeconds': 1.0, 'lstmWindowSeconds': 30.0}
    m = cnnLstmModel_torch.CnnLstmModelPyTorch(cfg, debug=False)
    (x,) = m.export_example_inputs()
    assert tuple(x.shape) == (1, 1, 750)


def test_cnn_lstm_xyz_export_inputs():
    cfg = {'sampleFreq': 25, 'cnnWindowSeconds': 1.0, 'lstmWindowSeconds': 30.0,
           'accelInputMode': 'xyz', 'inputChannels': 3}
    m = cnnLstmModel_torch.CnnLstmModelPyTorch(cfg, debug=False)
    (x,) = m.export_example_inputs()
    assert tuple(x.shape) == (1, 3, 750)


def test_cnn_lstm_export_inputs_run_through_forward():
    # The exact failure from output/lstm_1d/19: tracing must accept the
    # channels-first layout the .pte runtime feeds at inference time.
    cfg = {'sampleFreq': 25, 'cnnWindowSeconds': 1.0, 'lstmWindowSeconds': 45.0}
    m = cnnLstmModel_torch.CnnLstmModelPyTorch(cfg, debug=False)
    m.makeModel(num_classes=2)
    m.model.eval().cpu()
    (x,) = m.export_example_inputs()
    with torch.no_grad():
        out = m.model(x)
    assert tuple(out.shape) == (1, 2)


def test_deep_epicnn_export_inputs_default():
    m = deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch(None, debug=False)
    # no config -> 30 s default
    (x,) = m.export_example_inputs()
    assert tuple(x.shape) == (1, 1, 750)


def test_deep_epicnn_export_inputs_configured():
    cfg = {'sampleFreq': 25, 'bufferSeconds': 45.0}
    m = deepEpiCnnModel_torch.DeepEpiCnnModelPyTorch(cfg, debug=False)
    (x,) = m.export_example_inputs()
    assert tuple(x.shape) == (1, 1, 1125)
    m.makeModel(num_classes=2)
    m.model.eval().cpu()
    with torch.no_grad():
        out = m.model(x)
    assert tuple(out.shape) == (1, 2)
