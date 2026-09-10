#!/usr/bin/env python3
"""Unit tests for timezone-safe datetime normalization."""

import os
import sys


# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from datetime_normalization import normalize_datetime_string, normalize_event_datetimes


def test_normalize_datetime_string_keeps_naive_wall_time_as_utc():
    assert normalize_datetime_string('2022-04-23 14:31:56') == '2022-04-23T14:31:56Z'


def test_normalize_datetime_string_converts_offset_to_utc():
    assert normalize_datetime_string('2022-04-23T15:32:16+01:00') == '2022-04-23T14:32:16Z'


def test_normalize_event_datetimes_normalizes_event_and_datapoints():
    event = {
        'id': 5595,
        'dataTime': '2022-04-23 14:31:56',
        'datapoints': [
            {'dataTime': '2022-04-23 14:32:02'},
            {'dataTime': '2022-04-23T15:32:16+01:00'},
            {'time': '2022-04-23 14:32:24'},
        ],
    }

    out = normalize_event_datetimes(event, normalize_datapoints=True, in_place=False)
    assert out['dataTime'] == '2022-04-23T14:31:56Z'
    assert out['datapoints'][0]['dataTime'] == '2022-04-23T14:32:02Z'
    assert out['datapoints'][1]['dataTime'] == '2022-04-23T14:32:16Z'
    assert out['datapoints'][2]['time'] == '2022-04-23T14:32:24Z'
