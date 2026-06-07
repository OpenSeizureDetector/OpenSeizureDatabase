#!/usr/bin/env python3

import os
import sys
import unittest
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import libosd.osdDbConnection
import user_tools.testRunner.testRunner as testRunner


def _make_event(event_id, type_str, sub_type):
    return {
        'id': event_id,
        'userId': 1,
        'type': type_str,
        'subType': sub_type,
        'dataTime': '2022-01-01T00:00:00Z',
        'desc': 'test',
        'datapoints': [],
    }


class TestTonicClonicStatsAppend(unittest.TestCase):
    def test_appends_tonic_clonic_stats_to_all_seizures_output(self):
        osd = libosd.osdDbConnection.OsdDbConnection(cacheDir=None, debug=False)
        # Two seizures: one tonic-clonic and one other seizure subtype
        osd.addEvent(_make_event(1, 'seizure', 'Tonic-Clonic'))
        osd.addEvent(_make_event(2, 'seizure', 'other'))
        # One non-seizure
        osd.addEvent(_make_event(3, 'false alarm', ''))

        event_ids = [1, 2, 3]
        alg_names = ['Alg1']

        # results[event][alg][status]
        # status index 2 is ALARM; generate alarm only for event 1
        results = np.zeros((3, 1, 5), dtype=float)
        results[0, 0, 2] = 1  # tonic-clonic seizure -> alarm
        results[1, 0, 0] = 1  # other seizure -> no alarm
        results[2, 0, 0] = 1  # false alarm -> no alarm

        results_str = [["_2"], ["_0"], ["_0"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                testRunner.saveResults2('output', results, results_str, event_ids, osd, alg_names)

                with open('output_allSeizures.csv', 'r') as f:
                    contents = f.read()

                self.assertIn('#Total', contents)
                self.assertIn('#Correct Count', contents)
                self.assertIn('#Correct Prop', contents)

                # Tonic-clonic appended block
                self.assertIn('#TonicClonic Total', contents)
                self.assertIn('#TonicClonic Correct Count', contents)
                self.assertIn('#TonicClonic Correct Prop', contents)

                # Totals: allSeizures has 2 seizure events; tonic-clonic has 1
                lines = [ln.strip() for ln in contents.splitlines() if ln.strip()]
                tc_total = next(ln for ln in lines if ln.startswith('#TonicClonic Total'))
                tc_correct = next(ln for ln in lines if ln.startswith('#TonicClonic Correct Count'))
                tc_prop = next(ln for ln in lines if ln.startswith('#TonicClonic Correct Prop'))

                # Last two columns are Alg1 and reported
                tc_total_vals = [v.strip() for v in tc_total.split(',') if v.strip().isdigit()]
                self.assertEqual(tc_total_vals, ['1', '1'])

                tc_correct_vals = [v.strip() for v in tc_correct.split(',') if v.strip().isdigit()]
                # Alg1 alarms correctly; reported is 0 because the event has no datapoints.
                self.assertEqual(tc_correct_vals, ['1', '0'])

                # Prop should be 1.00 for Alg1 and 0.00 for reported
                self.assertIn('1.00', tc_prop)
                self.assertIn('0.00', tc_prop)
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
