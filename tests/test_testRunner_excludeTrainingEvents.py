#!/usr/bin/env python3

import os
import sys
import json
import csv
import tempfile
import unittest

# Ensure repo root is on path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import user_tools.testRunner.testRunner as testRunner


def _make_event(event_id, user_id=1, event_type='seizure'):
    return {
        'id': event_id,
        'userId': user_id,
        'type': event_type,
        'subType': 'test',
        'dataTime': '2022-01-01T00:00:00Z',
        'desc': f'event {event_id}',
        'datapoints': [
            {
                'dataTime': '2022-01-01T00:00:00Z',
                'alarmState': 0,
                'specPower': 0,
                'roiPower': 0,
                'hr': 60,
                'o2Sat': 98,
                'rawData': [0] * 125,
                'rawData3D': [0] * 375,
            }
        ],
    }


class TestExcludeTrainingEvents(unittest.TestCase):
    def test_exclude_training_events_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'db.json')
            train_path = os.path.join(tmp, 'trainData.json')

            # Database has 3 events
            events = [_make_event(1), _make_event(2), _make_event(3)]
            with open(db_path, 'w') as f:
                json.dump(events, f)

            # Training data used events 2 and 3
            train_events = [_make_event(2), _make_event(3)]
            with open(train_path, 'w') as f:
                json.dump(train_events, f)

            # Load DB via testRunner helper
            osd = testRunner.loadDataFiles(['db.json'], dbDir=tmp, debug=False)
            self.assertEqual(set(osd.getEventIds()), {1, 2, 3})

            removed = testRunner.exclude_training_events_from_osd(
                osd,
                'trainData.json',
                search_dirs=[tmp],
                debug=False,
            )
            self.assertEqual(set(removed), {2, 3})
            self.assertEqual(set(osd.getEventIds()), {1})

    def test_exclude_training_events_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'db.json')
            train_csv_path = os.path.join(tmp, 'trainData.csv')

            # Database has 3 events
            events = [_make_event(10), _make_event(20), _make_event(30)]
            with open(db_path, 'w') as f:
                json.dump(events, f)

            # Training CSV: multiple rows per event, with header containing eventId
            with open(train_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['eventId', 'userId', 'typeStr', 'dataTime'])
                writer.writerow([10, 1, 'Seizure/test', '01-01-2022 00:00:00'])
                writer.writerow([10, 1, 'Seizure/test', '01-01-2022 00:00:05'])  # duplicate event
                writer.writerow([30, 2, 'Seizure/test', '01-01-2022 00:00:10'])

            osd = testRunner.loadDataFiles(['db.json'], dbDir=tmp, debug=False)
            self.assertEqual(set(osd.getEventIds()), {10, 20, 30})

            removed = testRunner.exclude_training_events_from_osd(
                osd,
                'trainData.csv',
                search_dirs=[tmp],
                debug=False,
            )
            self.assertEqual(set(removed), {10, 30})
            self.assertEqual(set(osd.getEventIds()), {20})


if __name__ == '__main__':
    unittest.main()
