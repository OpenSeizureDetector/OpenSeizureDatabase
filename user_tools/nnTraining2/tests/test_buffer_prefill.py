#!/usr/bin/env python3
"""
Tests for test-time acceleration buffer pre-fill (modelConfig.testBufferPrefill).

The rolling buffer models (e.g. CnnLstmModelPyTorch) return None from dp2vector()
until bufferSamples samples have been accumulated, so the first ~45 s of every
buffer segment used to be dropped during testing and the event charts started late.
prefillAccBuf() fills the buffer at the start of each segment (event start, or
restart after a data gap) so that every datapoint is scored. Training is unaffected.

Modes: 'repeat' (default - tile the segment's first real datapoint),
'noise' (Gaussian matched to the first datapoint, optionally seeded),
'stationary' (legacy flat 1 g fill), 'none' (disabled, rows dropped).
Warm-up datapoints (window still containing pre-fill) are flagged downstream
and excluded from alarm decisions - see test_warmup_masking.py.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_tools.nnTraining2 import nnModel
from user_tools.nnTraining2 import nnTester
from user_tools.nnTraining2 import cnnLstmModel_torch


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------

def test_prefill_mode_defaults_to_repeat():
    assert nnTester.get_test_prefill_mode({'modelConfig': {}}) == 'repeat'


def test_prefill_mode_missing_config_defaults_to_repeat():
    assert nnTester.get_test_prefill_mode({}) == 'repeat'


def test_prefill_mode_disabled_values():
    for val in ('none', 'off', 'false', 'no', 'NONE', ''):
        cfg = {'modelConfig': {'testBufferPrefill': val}}
        assert nnTester.get_test_prefill_mode(cfg) is None, val


def test_prefill_mode_explicit_stationary():
    cfg = {'modelConfig': {'testBufferPrefill': 'stationary'}}
    assert nnTester.get_test_prefill_mode(cfg) == 'stationary'


def test_prefill_mode_repeat_and_noise():
    for val in ('repeat', 'noise', 'REPEAT', 'Noise'):
        cfg = {'modelConfig': {'testBufferPrefill': val}}
        assert nnTester.get_test_prefill_mode(cfg) == val.strip().lower(), val


def test_prefill_mode_unknown_value_disables(capsys):
    cfg = {'modelConfig': {'testBufferPrefill': 'chocolate'}}
    assert nnTester.get_test_prefill_mode(cfg) is None
    assert 'unknown testBufferPrefill' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# base class implementation (used by all buffered models)
# ---------------------------------------------------------------------------

class _BufferedStub(nnModel.NnModel):
    def __init__(self, buffer_samples=None):
        super().__init__(configObj=None, debug=False)
        if buffer_samples is not None:
            self.bufferSamples = buffer_samples
            self.accBuf = []


class _UnbufferedStub(nnModel.NnModel):
    pass


def test_base_prefill_no_buffer():
    m = _UnbufferedStub()
    assert m.getAccBufSize() == 0
    assert m.prefillAccBuf('stationary') is False


def test_base_prefill_fills_magnitudes():
    m = _BufferedStub(buffer_samples=10)
    assert m.getAccBufSize() == 10
    assert m.prefillAccBuf('stationary') is True
    assert m.accBuf == [1000.0] * 10


def test_base_prefill_analysis_samp_attr():
    m = nnModel.NnModel(configObj=None, debug=False)
    m.analysisSamp = 4
    m.accBuf = []
    assert m.getAccBufSize() == 4
    assert m.prefillAccBuf('stationary') is True
    assert m.accBuf == [1000.0] * 4


def test_base_prefill_unknown_mode_leaves_buffer_alone():
    m = _BufferedStub(buffer_samples=10)
    assert m.prefillAccBuf('bogus') is False
    assert m.accBuf == []


def test_base_prefill_repeat_tiles_reference():
    m = _BufferedStub(buffer_samples=10)
    assert m.prefillAccBuf('repeat', ref=[1.0, 2.0, 3.0]) is True
    assert m.accBuf == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]


def test_base_prefill_repeat_without_ref_falls_back_to_stationary():
    m = _BufferedStub(buffer_samples=4)
    assert m.prefillAccBuf('repeat') is True
    assert m.accBuf == [1000.0] * 4


def test_base_prefill_noise_matches_reference_stats():
    rng = np.random.default_rng(7)
    ref = list(rng.normal(1005.0, 6.0, 125))
    m = _BufferedStub(buffer_samples=1000)
    assert m.prefillAccBuf('noise', ref=ref, rng=np.random.default_rng(7)) is True
    assert len(m.accBuf) == 1000
    assert abs(np.mean(m.accBuf) - 1005.0) < 5.0
    assert abs(np.std(m.accBuf) - 6.0) < 3.0
    # perfectly flat fill must NOT occur (that is the OOD trigger this replaces)
    assert np.std(m.accBuf) > 0


def test_base_prefill_noise_deterministic_with_seed():
    ref = [1000.0 + (i % 7) for i in range(125)]
    m1 = _BufferedStub(buffer_samples=250)
    m2 = _BufferedStub(buffer_samples=250)
    assert m1.prefillAccBuf('noise', ref=ref, rng=1234) is True
    assert m2.prefillAccBuf('noise', ref=ref, rng=1234) is True
    assert m1.accBuf == m2.accBuf


def test_base_prefill_noise_without_ref_falls_back_to_stationary():
    m = _BufferedStub(buffer_samples=4)
    assert m.prefillAccBuf('noise') is True
    assert m.accBuf == [1000.0] * 4


def test_get_warmup_datapoints():
    assert _BufferedStub(buffer_samples=1125).get_warmup_datapoints(125) == 8
    assert _BufferedStub(buffer_samples=150).get_warmup_datapoints(125) == 1
    assert _BufferedStub(buffer_samples=125).get_warmup_datapoints(125) == 0
    assert _UnbufferedStub().get_warmup_datapoints(125) == 0


def test_warmup_datapoints_for_model_helper():
    # Same call signature as nnTester.testModel's warm-up setup.
    assert nnTester._warmup_datapoints_for_model(
        _BufferedStub(buffer_samples=1125), samples_per_datapoint=125) == 8
    assert nnTester._warmup_datapoints_for_model(
        _UnbufferedStub(), samples_per_datapoint=125) == 0


# ---------------------------------------------------------------------------
# CnnLstmModelPyTorch (the model whose warm-up rows were being dropped)
# ---------------------------------------------------------------------------

def _lstm_model(**cfg):
    config = {'sampleFreq': 25, 'lstmWindowSeconds': 6.0}  # bufferSamples = 150
    config.update(cfg)
    return cnnLstmModel_torch.CnnLstmModelPyTorch(config, debug=False)


def _mag_dp(n_samp=125):
    return {'rawData': [float(i) for i in range(n_samp)], 'hr': 70}


def test_lstm_warmup_without_prefill_returns_none():
    m = _lstm_model()
    m.resetAccBuf()
    assert m.getAccBufSize() == 150
    assert m.dp2vector(_mag_dp(), normalise=False) is None


def test_lstm_prefill_magnitude_scores_first_datapoint():
    m = _lstm_model()
    m.resetAccBuf()
    assert m.prefillAccBuf('stationary') is True

    vec = m.dp2vector(_mag_dp(), normalise=False)
    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (150,)
    # 150-sample buffer, 125 new samples -> oldest 25 samples are the pre-fill
    assert np.allclose(vec[:25], 1.0)            # 1000 milli-g / 1000 = 1 g
    assert np.allclose(vec[25:], np.arange(125) / 1000.0)


def test_lstm_prefill_resets_between_events():
    m = _lstm_model()
    m.resetAccBuf()
    m.prefillAccBuf('stationary')
    assert m.dp2vector(_mag_dp(), normalise=False) is not None
    # new event: reset clears, prefill restores, first datapoint scored again
    m.resetAccBuf()
    assert m.accBuf == []
    m.prefillAccBuf('stationary')
    assert m.dp2vector(_mag_dp(), normalise=False) is not None


def test_lstm_prefill_unknown_mode():
    m = _lstm_model()
    m.resetAccBuf()
    assert m.prefillAccBuf('bogus') is False
    assert m.accBuf == []


def test_lstm_prefill_xyz_mode():
    m = _lstm_model(accelInputMode='xyz', inputChannels=3)
    m.resetAccBuf()
    assert m.prefillAccBuf('stationary') is True
    assert len(m.accBuf3D) == 150
    assert all(s == [0.0, 0.0, 1000.0] for s in m.accBuf3D)

    raw3d = []
    for i in range(125):
        raw3d.extend([float(i), float(i) + 1.0, float(i) + 2.0])
    vec = m.dp2vector({'rawData3D': raw3d, 'hr': 70}, normalise=False)
    assert vec is not None
    assert vec.shape == (150, 3)
    assert np.allclose(vec[:25], [0.0, 0.0, 1.0])
    assert np.allclose(vec[25, 0], 0.0)
    assert np.allclose(vec[125, 0], 100.0 / 1000.0)


def test_lstm_prefill_repeat_scores_first_datapoint():
    m = _lstm_model()
    m.resetAccBuf()
    ref = [1000.0 + float(i % 5) for i in range(125)]
    assert m.prefillAccBuf('repeat', ref=ref) is True
    vec = m.dp2vector(_mag_dp(), normalise=False)
    assert vec is not None
    assert vec.shape == (150,)
    # oldest 25 samples are the tiled reference, not a flat line
    assert np.allclose(vec[:25], np.array(ref[:25]) / 1000.0)


def test_lstm_prefill_noise_xyz():
    m = _lstm_model(accelInputMode='xyz', inputChannels=3)
    m.resetAccBuf()
    ref = np.zeros((125, 3))
    ref[:, 0] = 10.0 + np.arange(125) % 3
    ref[:, 1] = -5.0 + np.arange(125) % 3
    ref[:, 2] = 1000.0 + np.arange(125) % 3
    assert m.prefillAccBuf('noise', ref=ref, rng=42) is True
    assert len(m.accBuf3D) == 150
    arr = np.asarray(m.accBuf3D)
    assert arr.shape == (150, 3)
    assert abs(arr[:, 2].mean() - 1001.0) < 5.0
    assert arr.std() > 0


def test_stationary_fill_value_respects_dc_normalisation():
    # Legacy (no config / flag off): stationary fill = 1 g = 1000 milli-g.
    m = _BufferedStub(buffer_samples=4)
    assert m._stationary_fill_value() == 1000.0
    assert m.prefillAccBuf('stationary') is True
    assert m.accBuf == [1000.0] * 4

    # dcNormalisation on: flattened data is per-datapoint zero-mean, so a
    # stationary sensor reads 0 milli-g.
    m.configObj = {'dataProcessing': {'dcNormalisation': True}}
    assert m._stationary_fill_value() == 0.0
    assert m.prefillAccBuf('stationary') is True
    assert m.accBuf == [0.0] * 4
    # Fallback path (repeat without reference) uses the same value.
    m.accBuf = []
    assert m.prefillAccBuf('repeat') is True
    assert m.accBuf == [0.0] * 4

    # Flag off again -> back to 1000.
    m.configObj = {'dataProcessing': {'dcNormalisation': False}}
    assert m.prefillAccBuf('stationary') is True
    assert m.accBuf == [1000.0] * 4
