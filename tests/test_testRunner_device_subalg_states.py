#!/usr/bin/env python3

import os
import sys
import json
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import libosd.osdDbConnection
import user_tools.testRunner.testRunner as testRunner


class DeviceAlg:
    """Fake DeviceAlg that emits per-subalgorithm state fields."""

    def __init__(self, settingsStr=None, debug=False):
        self.debug = debug

    def resetAlg(self):
        return

    def processDp(self, rawDataStr, eventId):
        # Mimic the updated /data JSON structure
        ret = {
            'valid': True,
            # Overall voted alarm state
            'alarmState': 0,
            # Internal algorithm states
            'osdAlgState': 2,
            'flapAlgState': 0,
            'hrAlgState': 1,
            'cnnAlgState': 0,
            # ML model arrays
            'mlNumModels': 1,
            'mlModelNames': ['ML1'],
            'mlModelStates': [2],
            'mlModelActive': [True],
        }
        return json.dumps(ret)


def _make_event(event_id):
    dp = {
        'dataTime': '2022-01-01T00:00:00Z',
        'alarmState': 0,
        'hr': 60,
        'rawData': [0] * 125,
        'rawData3D': [0] * 375,
        'specPower': 0,
        'roiPower': 0,
    }
    return {
        'id': event_id,
        'userId': 1,
        'type': 'seizure',
        'subType': 'test',
        'dataTime': '2022-01-01T00:00:00Z',
        'desc': 'test',
        'datapoints': [dp],
    }


class TestDeviceSubAlgStates(unittest.TestCase):
    def test_expands_subalg_slots_and_counts(self):
        osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=False)
        osd.addEvent(_make_event(1))

        event_ids = [1]
        algs = [DeviceAlg()]
        alg_names = ['Phone']

        results, results_strs, expanded = testRunner.testEachEvent(event_ids, osd, algs, alg_names, debug=False)

        self.assertIn('Phone', expanded)
        self.assertIn('Phone.osdAlgState', expanded)
        self.assertIn('Phone.flapAlgState', expanded)
        self.assertIn('Phone.hrAlgState', expanded)
        self.assertIn('Phone.cnnAlgState', expanded)
        self.assertIn('Phone.ml.ML1', expanded)

        idx_osd = expanded.index('Phone.osdAlgState')
        idx_hr = expanded.index('Phone.hrAlgState')
        idx_ml = expanded.index('Phone.ml.ML1')

        # osdAlgState=2 and mlModelStates[0]=2 should count as ALARM
        self.assertEqual(results[0][idx_osd][2], 1)
        self.assertEqual(results[0][idx_ml][2], 1)

        # hrAlgState=1 should count as WARN
        self.assertEqual(results[0][idx_hr][1], 1)

        # Ensure status strings were recorded
        self.assertTrue(results_strs[0][idx_osd].endswith('2'))
        self.assertTrue(results_strs[0][idx_hr].endswith('1'))


if __name__ == '__main__':
    unittest.main()
