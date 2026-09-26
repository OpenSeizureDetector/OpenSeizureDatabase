#!/usr/bin/env python3
"""
Tests for the subtype source used by subtype-weighted sampling.

The flattened/feature CSVs (flattenData.py:333) carry the subtype inside typeStr
as 'Type/SubType' - e.g. 'Seizure/Tonic-Clonic' - and have no dedicated subType
column, so nnTrainer.ensure_subtype_column() derives it before building the
subtype-aware sampler. Without this, subtype weighting silently falls back to
class-balanced sampling ("eventId/subType columns are unavailable").
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from user_tools.nnTraining2 import nnTrainer


def test_derives_sub_type_from_type_str():
    df = pd.DataFrame({
        'eventId': [1, 2, 3],
        'typeStr': ['Seizure/Tonic-Clonic', 'Seizure/Other', 'False Alarm/'],
    })
    out = nnTrainer.ensure_subtype_column(df)
    assert list(out['subType']) == ['Tonic-Clonic', 'Other', '']


def test_existing_sub_type_column_untouched():
    df = pd.DataFrame({'eventId': [1], 'subType': ['Focal'], 'typeStr': ['Seizure/Focal']})
    out = nnTrainer.ensure_subtype_column(df)
    assert out is df
    assert list(out['subType']) == ['Focal']


def test_event_type_column_used_when_no_type_str():
    df = pd.DataFrame({'eventId': [1, 2], 'eventType': ['Seizure/Tonic-Clonic', 'Seizure/Aura']})
    out = nnTrainer.ensure_subtype_column(df)
    assert list(out['subType']) == ['Tonic-Clonic', 'Aura']


def test_returns_none_without_any_source_column():
    df = pd.DataFrame({'eventId': [1], 'type': [1]})
    assert nnTrainer.ensure_subtype_column(df) is None


def test_original_dataframe_not_modified():
    df = pd.DataFrame({'eventId': [1], 'typeStr': ['Seizure/Tonic-Clonic']})
    nnTrainer.ensure_subtype_column(df)
    assert 'subType' not in df.columns


def test_sampler_applies_tonic_clonic_weight():
    pytest.importorskip('torch')
    from user_tools.nnTraining2.subtype_weighting import create_subtype_weighted_sampler

    df = pd.DataFrame({
        'eventId': ['tc'] * 4 + ['other'] * 4 + ['nda'] * 4,
        'typeStr': (['Seizure/Tonic-Clonic'] * 4 + ['Seizure/Other'] * 4
                    + ['False Alarm/'] * 4),
    })
    df = nnTrainer.ensure_subtype_column(df)
    y = [1] * 4 + [1] * 4 + [0] * 4
    weights = {'Tonic-Clonic': 2.0, 'Other': 1.0}

    sampler = create_subtype_weighted_sampler(df=df, y_values=y, subtype_weights=weights)
    w = sampler.weights.tolist()
    # class weights: 1/4 (seizure) and 1/4 (non-seizure) -> TC gets x2 multiplier
    assert w[0] == pytest.approx(1.0)      # tonic-clonic seizure, normalised
    assert w[4] == pytest.approx(0.5)      # other seizure
    assert w[8] == pytest.approx(1.0)      # non-seizure (no subtype multiplier)


def test_sampler_without_sub_type_falls_back_to_class_weights():
    pytest.importorskip('torch')
    from user_tools.nnTraining2.subtype_weighting import create_subtype_weighted_sampler

    df = pd.DataFrame({'eventId': ['tc'] * 4 + ['other'] * 4, 'type': [1] * 8})
    y = [1] * 8
    sampler = create_subtype_weighted_sampler(
        df=df, y_values=y, subtype_weights={'Tonic-Clonic': 2.0})
    # no subType column -> class-based fallback -> all weights equal
    assert len(set(round(float(x), 6) for x in sampler.weights)) == 1
