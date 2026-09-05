#!/usr/bin/env python3
"""Minimal-event tests for user_tools.nnTraining2.augmentData
using the small synthetic CSV at user_tools/nnTraining2/tests/data/aug_test_small.csv.
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import user_tools.nnTraining2.augmentData as augmentData

TEST_DATA = os.path.join(
    ROOT,
    'user_tools',
    'nnTraining2',
    'tests',
    'data',
    'aug_test_small.csv',
)


class TestAugmentDataMinimal(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(TEST_DATA)
        # Normalize eventId to string for consistent comparisons
        self.df['eventId'] = self.df['eventId'].astype(str)
        self.m_cols = [f"M{n:03d}" for n in range(125)]

    def test_noiseAug_event_counts_and_changes(self):
        """Noise aug should duplicate seizure events with suffixes and perturb magnitudes."""
        np.random.seed(0)
        noise_val = 0.5
        noise_fac = 3
        aug_df = augmentData.noiseAug(self.df, noise_val, noise_fac, debug=False)

        seizures_df, nonseizures_df = augmentData.getSeizureNonSeizureDfs(aug_df)
        # Original seizures: 2 events -> expect 2 * (1 + noise_fac) events
        unique_events = sorted(seizures_df['eventId'].astype(str).unique())
        self.assertEqual(len(unique_events), 2 * (1 + noise_fac))
        # Each event should retain two rows
        for eid in unique_events:
            self.assertEqual(len(seizures_df[seizures_df['eventId'] == eid]), 2)

        # Check magnitudes changed for one augmented event compared to its source
        orig_row = self.df[self.df['eventId'] == '1001'].iloc[0][self.m_cols]
        aug_row = seizures_df[seizures_df['eventId'] == '1001-1'].iloc[0][self.m_cols]
        diff = np.abs(aug_row.to_numpy() - orig_row.to_numpy())
        self.assertGreater(diff.sum(), 0.0)

        # Non-seizure event should remain single and unchanged in count
        self.assertEqual(len(nonseizures_df['eventId'].unique()), 1)
        self.assertEqual(nonseizures_df['eventId'].astype(str).unique()[0], '2001')
        self.assertEqual(len(nonseizures_df), 2)

    def test_phaseAug_event_counts_step1(self):
        """Phase aug with step=1 should generate sliding windows across the concatenated event."""
        aug_df = augmentData.phaseAug(self.df, phase_step=1, debug=False)
        seizures_df, nonseizures_df = augmentData.getSeizureNonSeizureDfs(aug_df)

        unique_events = seizures_df['eventId'].astype(str).unique()
        # originals (2) + 125 augmented per seizure event (125 windows, skipping offset 0) => 2 + 125*2 = 252 events total
        self.assertEqual(len(unique_events), 2 + 125*2)

        # Each augmented event should have 1 row
        for i in range(1, 126):
            for orig_eid in ['1001', '1002']:
                aug_eid = f'{orig_eid}-phase{i}'
                self.assertEqual(len(seizures_df[seizures_df['eventId'] == aug_eid]), 1)

        # Originals remain at 2 rows each
        for eid in ['1001', '1002']:
            self.assertEqual(len(seizures_df[seizures_df['eventId'] == eid]), 2)

        # Non-seizure untouched
        self.assertEqual(len(nonseizures_df['eventId'].unique()), 1)
        self.assertEqual(nonseizures_df['eventId'].astype(str).unique()[0], '2001')

    def test_phaseAug_step_applied_counts_and_shift(self):
        """Phase aug uses the configured phase_step for sliding windows and correct counts."""
        phase_step = 25
        aug_df = augmentData.phaseAug(self.df, phase_step=phase_step, debug=False)
        seizures_df, _ = augmentData.getSeizureNonSeizureDfs(aug_df)

        unique_events = seizures_df['eventId'].astype(str).unique()
        # originals (2) + 5 augmented per seizure event (skipping offset 0) => 2 + 5*2 = 12 events total
        self.assertEqual(len(unique_events), 2 + 5*2)

        # Each augmented event should have 1 row
        for i in range(1, 6):
            for orig_eid in ['1001', '1002']:
                aug_eid = f'{orig_eid}-phase{i}'
                self.assertEqual(len(seizures_df[seizures_df['eventId'] == aug_eid]), 1)

        # Check shift correctness
        # Original event 1001 has 2 rows: [1.0-1.124] and [1.125-1.249]
        # Concatenated: [1.0, 1.001, ..., 1.124, 1.125, ..., 1.249]
        # Window 1 (1001-phase1) starts at sample 1*25=25: [1.025, 1.026, ..., 1.149]
        # Window 2 (1001-phase2) starts at sample 2*25=50: [1.050, 1.051, ..., 1.174]
        
        # Check first window (offset by 25)
        row1 = seizures_df[seizures_df['eventId'] == '1001-phase1'].iloc[0]
        mag1 = row1[self.m_cols].astype(float).to_numpy()
        expected1 = np.array([1.025 + i*0.001 for i in range(125)])
        np.testing.assert_allclose(mag1, expected1, rtol=1e-5)
        
        # Check second window  
        row2 = seizures_df[seizures_df['eventId'] == '1001-phase2'].iloc[0]
        mag2 = row2[self.m_cols].astype(float).to_numpy()
        expected2 = np.array([1.050 + i*0.001 for i in range(125)])
        np.testing.assert_allclose(mag2, expected2, rtol=1e-5)

    def test_sampleRateAug_resamples_and_rebuilds_125_windows(self):
        """Sample-rate aug should resample concatenated seizure data and rebuild 125-sample rows."""
        aug_df = augmentData.sampleRateAug(self.df, sampleRateFactors=[0.5, 1.5], debug=False)
        seizures_df, nonseizures_df = augmentData.getSeizureNonSeizureDfs(aug_df)

        unique_events = set(seizures_df['eventId'].astype(str).unique())
        expected_ids = {
            '1001', '1002',
            '1001-sr0p5', '1002-sr0p5',
            '1001-sr1p5', '1002-sr1p5',
        }
        self.assertTrue(expected_ids.issubset(unique_events))

        # 250*0.5 -> 125 samples -> 1 row, 250*1.5 -> 375 samples -> 3 rows.
        self.assertEqual(len(seizures_df[seizures_df['eventId'] == '1001-sr0p5']), 1)
        self.assertEqual(len(seizures_df[seizures_df['eventId'] == '1001-sr1p5']), 3)

        # Verify 0.5x event content via linear interpolation of the concatenated source event.
        orig_concat = self.df[self.df['eventId'] == '1001'][self.m_cols].astype(float).to_numpy().reshape(-1)
        expected_resampled = np.interp(
            np.linspace(0.0, 1.0, 125),
            np.linspace(0.0, 1.0, len(orig_concat)),
            orig_concat,
        )

        sr_event_row = seizures_df[seizures_df['eventId'] == '1001-sr0p5'].iloc[0]
        x_cols = [f"X{n:03d}" for n in range(125)]
        x_row = sr_event_row[x_cols].astype(float).to_numpy()
        np.testing.assert_allclose(x_row, expected_resampled, rtol=1e-6, atol=1e-8)

        # In 3D mode magnitude is recomputed from XYZ, so expected M is sqrt(3) * channel value for this fixture.
        sr_mag = sr_event_row[self.m_cols].astype(float).to_numpy()
        np.testing.assert_allclose(sr_mag, np.sqrt(3.0) * expected_resampled, rtol=1e-6, atol=1e-8)

        self.assertEqual(len(nonseizures_df[nonseizures_df['eventId'].astype(str) == '2001']), 2)

    def test_noiseAugNonSeizure_filters_by_type_subtype_pairs(self):
        """Non-seizure noise aug should duplicate only events matching configured selector pairs."""
        np.random.seed(0)
        df = self.df.copy()
        df['eventId'] = df['eventId'].astype(str)
        df['subType'] = ''

        # Selected non-seizure event.
        df.loc[df['eventId'] == '2001', 'subType'] = 'Check'

        # Non-selected non-seizure event.
        extra = df[df['eventId'] == '2001'].copy()
        extra['eventId'] = '2002'
        extra['subType'] = 'Other'
        df = pd.concat([df, extra], ignore_index=True)

        aug_df = augmentData.noiseAugNonSeizure(
            df,
            noiseAugVal=0.5,
            noiseAugFac=2,
            targetTypeSubTypePairs=[{'type': 0, 'subType': 'check'}],
            debug=False,
        )

        seizures_df, nonseizures_df = augmentData.getSeizureNonSeizureDfs(aug_df)
        self.assertEqual(set(seizures_df['eventId'].astype(str).unique()), {'1001', '1002'})

        ns_event_ids = set(nonseizures_df['eventId'].astype(str).unique())
        self.assertIn('2001-nns1', ns_event_ids)
        self.assertIn('2001-nns2', ns_event_ids)
        self.assertNotIn('2002-nns1', ns_event_ids)

        self.assertEqual(len(nonseizures_df[nonseizures_df['eventId'].astype(str) == '2001-nns1']), 2)
        self.assertEqual(len(nonseizures_df[nonseizures_df['eventId'].astype(str) == '2002']), 2)

        orig_row = nonseizures_df[nonseizures_df['eventId'].astype(str) == '2001'].iloc[0][self.m_cols].astype(float).to_numpy()
        aug_row = nonseizures_df[nonseizures_df['eventId'].astype(str) == '2001-nns1'].iloc[0][self.m_cols].astype(float).to_numpy()
        self.assertGreater(np.abs(aug_row - orig_row).sum(), 0.0)

    def test_userAug_balanced_no_new_events(self):
        """Dataset already balanced by user; userAug should not add seizure events."""
        aug_df = augmentData.userAug(self.df)
        seizures_df, nonseizures_df = augmentData.getSeizureNonSeizureDfs(aug_df)

        self.assertEqual(set(seizures_df['eventId'].astype(str).unique()), {'1001', '1002'})
        # Row counts unchanged: 4 seizure rows and 2 non-seizure rows
        self.assertEqual(len(seizures_df), 4)
        self.assertEqual(len(nonseizures_df), 2)


if __name__ == '__main__':
    unittest.main()
