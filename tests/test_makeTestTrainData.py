#!/usr/bin/env python3
"""Compatibility tests for data splitting and flattening in nnTraining2.

This file replaces legacy nnTraining-based tests with nnTraining2 equivalents.
"""

import json
import os
import sys
import unittest

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import user_tools.nnTraining2.splitData as splitData
import user_tools.nnTraining2.flattenData as flattenData


def _make_raw_data(base):
    return [base + i for i in range(125)]


class TestSplitCsvData(unittest.TestCase):
    def test_splitCsvData_single_split(self):
        # Build a synthetic flattened allData.csv with 10 events and 2 rows per event.
        rows = []
        for ev in range(10):
            ev_type = 1 if ev < 5 else 0
            for dp in range(2):
                rows.append({
                    "eventId": f"E{ev}",
                    "userId": ev % 3,
                    "type": ev_type,
                    "dataTime": f"2022-05-09T02:37:{10 + ev + dp:02d}Z",
                    "M000": float(ev),
                })

        with self.subTest("split csv"):
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                all_data_csv = os.path.join(tmpdir, "allData.csv")
                train_csv = "trainData.csv"
                test_csv = "testData.csv"
                pd.DataFrame(rows).to_csv(all_data_csv, index=False)

                config = {
                    "randomSeed": 42,
                    "dataFileNames": {
                        "trainDataFileCsv": train_csv,
                        "testDataFileCsv": test_csv,
                    },
                    "dataProcessing": {
                        "testProp": 0.3,
                    },
                }

                splitData.splitCsvData(config, all_data_csv, outDir=tmpdir, kFold=1, nestedKfold=1, debug=False)

                train_path = os.path.join(tmpdir, train_csv)
                test_path = os.path.join(tmpdir, test_csv)
                self.assertTrue(os.path.exists(train_path))
                self.assertTrue(os.path.exists(test_path))

                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                train_events = set(train_df["eventId"].astype(str).unique())
                test_events = set(test_df["eventId"].astype(str).unique())

                self.assertEqual(len(train_events | test_events), 10)
                self.assertEqual(len(train_events & test_events), 0)
                self.assertEqual(len(test_events), 3)


class TestFlattenData(unittest.TestCase):
    def test_flattenOsdb(self):
        events = []
        for event_no in range(4):
            datapoints = []
            for dp_no in range(3):
                datapoints.append({
                    "id": dp_no,
                    "dataTime": f"2022-05-09T02:37:{10 + event_no + dp_no:02d}Z",
                    "hr": 75,
                    "rawData": _make_raw_data(event_no),
                })

            events.append({
                "id": f"EV{event_no}",
                "userId": event_no % 2,
                "dataTime": f"2022-05-09T02:37:{10 + event_no:02d}Z",
                "type": "seizure" if event_no < 2 else "False Alarm",
                "subType": "test",
                "desc": "test",
                "datapoints": datapoints,
            })

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            in_json = os.path.join(tmpdir, "trainData.json")
            out_csv = os.path.join(tmpdir, "trainDataRaw.csv")

            with open(in_json, "w") as f:
                json.dump(events, f)

            flattenData.flattenOsdb(in_json, out_csv, debug=False, validate_datapoints=False, config=None)
            self.assertTrue(os.path.exists(out_csv))

            df_raw = pd.read_csv(out_csv)
            self.assertEqual(len(df_raw), 12)  # 4 events * 3 datapoints

            # Validate that key fields transferred correctly.
            self.assertIn("eventId", df_raw.columns)
            self.assertIn("userId", df_raw.columns)
            self.assertIn("hr", df_raw.columns)
            self.assertIn("M000", df_raw.columns)
            self.assertEqual(df_raw.iloc[0]["eventId"], "EV0")
            self.assertEqual(df_raw.iloc[0]["userId"], 0)
            self.assertEqual(df_raw.iloc[0]["hr"], 75)
            self.assertEqual(df_raw.iloc[0]["M000"], 0)


if __name__ == "__main__":
    unittest.main()
