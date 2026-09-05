#!/usr/bin/env python3

import csv
import os
import sys
import tempfile

import libosd.dpTools


# Ensure legacy testRunner modules resolve correctly in tests.
TESTRUNNER_DIR = os.path.join(os.path.dirname(__file__), '..', 'user_tools', 'testRunner')
if TESTRUNNER_DIR not in sys.path:
    sys.path.insert(0, TESTRUNNER_DIR)
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from user_tools.testRunner.io_utils import loadCsvFile


def _build_headers():
    return [
        'eventId', 'userId', 'typeStr', 'dataTime',
        'osdAlarmState', 'osdSpecPower', 'osdRoiPower', 'hr', 'o2sat'
    ]


def test_mixed_naive_and_z_timestamps_are_normalized_to_utc():
    rows = [
        ['5595', '55', 'Seizure/Other', '2022-04-23 14:31:56', '0', '0', '0', '60', '98'],
        ['5595', '55', 'Seizure/Other', '2022-04-23 14:32:02', '0', '0', '0', '60', '98'],
        ['5595', '55', 'Seizure/Other', '2022-04-23T14:32:16Z', '0', '0', '0', '60', '98'],
        ['5595', '55', 'Seizure/Other', '2022-04-23 14:32:24', '0', '0', '0', '60', '98'],
        ['5595', '55', 'Seizure/Other', '2022-04-23T14:33:33Z', '1', '0', '0', '60', '98'],
    ]

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'mixed_times.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(_build_headers())
            writer.writerows(rows)

        events = loadCsvFile(csv_path, debug=False)
        assert '5595' in events

        event = events['5595']
        assert event['dataTime'] == '2022-04-23T14:31:56Z'

        dp_times = [dp['dataTime'] for dp in event['datapoints']]
        assert all(t.endswith('Z') for t in dp_times)

        base = libosd.dpTools.dateStr2secs(event['dataTime'])
        offsets = [libosd.dpTools.dateStr2secs(t) - base for t in dp_times]

        # Ensure no accidental +3600 second jumps from mixed timezone parsing.
        assert max(offsets) < 300
        assert offsets == [0.0, 6.0, 20.0, 28.0, 97.0]
