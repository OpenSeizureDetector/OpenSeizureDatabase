#!/usr/bin/env python3
"""Tests for augmentation helpers in user_tools.nnTraining2.augmentData."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import user_tools.nnTraining2.augmentData as augmentData


class TestAug(unittest.TestCase):
    def setUp(self):
        row_lst = []

        # Seizure events: user1 has 3 events, user2 has 1 event.
        row_lst.append(self.make_row("s1", 1, 1))
        row_lst.append(self.make_row("s2", 1, 1))
        row_lst.append(self.make_row("s3", 1, 1))
        row_lst.append(self.make_row("s4", 2, 1))

        # Non-seizure events.
        row_lst.append(self.make_row("n1", 3, 0))
        row_lst.append(self.make_row("n2", 3, 0))

        columns_lst = ["eventId", "userId", "type", "dataTime", "hr", "o2sat"]
        for n in range(0, 125):
            columns_lst.append("M%03d" % n)
        self.df = pd.DataFrame(row_lst, columns=columns_lst)

    def make_row(self, event_id, user_id, event_type):
        row = [event_id, user_id, event_type, "2022-05-09T02:37:25Z", "70", None]
        for _ in range(0, 125):
            row.append(1000.0)
        return row

    def test_analyseDf(self):
        props = augmentData.getUserCounts(self.df)
        # 2 rows of 6 belong to user 3.
        self.assertAlmostEqual(props[3], 2.0 / 6.0)

    def test_userAug(self):
        """Check that user augmentation balances seizure events between users."""
        config = {"dataProcessing": {"userAugmentationThreshold": 1}}
        aug_df = augmentData.userAug(self.df, config=config)
        seizures_df, _ = augmentData.getSeizureNonSeizureDfs(aug_df)

        user_event_counts = seizures_df.groupby("userId")["eventId"].nunique()
        self.assertEqual(user_event_counts[1], user_event_counts[2])

    def test_noiseAug(self):
        """Check that noise augmentation creates expected seizure copies."""
        noise_val = 10.0
        noise_fac = 3

        seizures_df, non_seizure_df = augmentData.getSeizureNonSeizureDfs(self.df)
        n_seizure_events = seizures_df["eventId"].nunique()

        np.random.seed(0)
        aug_df = augmentData.noiseAug(self.df, noise_val, noise_fac, debug=False)
        seizures_df_aug, non_seizure_df_aug = augmentData.getSeizureNonSeizureDfs(aug_df)

        self.assertEqual(seizures_df_aug["eventId"].nunique(), n_seizure_events * (1 + noise_fac))
        self.assertEqual(non_seizure_df_aug["eventId"].nunique(), non_seizure_df["eventId"].nunique())

    def test_phaseAug(self):
        """Check phase augmentation event count with phase_step=25."""
        # Build a two-row-per-event seizure dataset for deterministic phase count.
        rows = []
        for eid, base in [("a", 1.0), ("b", 2.0)]:
            rows.append([eid, 1, 1, "2022-05-09T02:37:25Z", "70", None] + [base + i * 0.001 for i in range(125)])
            rows.append([eid, 1, 1, "2022-05-09T02:37:30Z", "70", None] + [base + 0.125 + i * 0.001 for i in range(125)])

        columns_lst = ["eventId", "userId", "type", "dataTime", "hr", "o2sat"] + [f"M{n:03d}" for n in range(125)]
        phase_df = pd.DataFrame(rows, columns=columns_lst)

        aug_df = augmentData.phaseAug(phase_df, phase_step=25, debug=False)
        seizures_df, _ = augmentData.getSeizureNonSeizureDfs(aug_df)

        # 2 originals + 5 phase events per source event = 12 unique seizure event IDs.
        self.assertEqual(seizures_df["eventId"].nunique(), 12)


if __name__ == "__main__":
    unittest.main()
