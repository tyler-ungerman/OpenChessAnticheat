"""Regression gates for research-only scoring and empirical null calibration."""
import unittest

import numpy as np
import pandas as pd

from backend.src.detectors.base import MIN_SIGMA
from backend.src.detectors.statistical_z import _combine_z_scores, _robust_z, aggregate_player_scores
from backend.src.detectors.calibration import _lookup_cohort_z
from backend.src.detectors.validation import empirical_account_tail, validate_label_manifest


class SafetyTests(unittest.TestCase):
    def test_degenerate_choice_z_bounded(self):
        series = pd.Series([0.04] * 12, name="choice_sensitivity_s")
        self.assertLessEqual(abs(_robust_z(-10.0, series)), 6.0)
        self.assertEqual(MIN_SIGMA["choice_sensitivity_s"], 0.08)

    def test_cohort_zero_variance_is_bounded(self):
        cal = {"buckets": {}, "global_tc": {"blitz": {"choice_sensitivity_s": (0.5, 0.0001)}}}
        self.assertEqual(_lookup_cohort_z(-10.0, "choice_sensitivity_s", "blitz", 1400, cal, False), 6.0)

    def test_correlated_channels_do_not_inflate(self):
        score, count = _combine_z_scores([(2.0, 0.5), (2.0, 0.5)])
        self.assertEqual((score, count), (2.0, 2))

    def test_account_summary_never_reports_cheat_probability(self):
        df = pd.DataFrame([
            {"game_id": str(i), "date": f"2026-11-{i+1:02d}", "time_control": "blitz",
             "player": "test", "color": "white", "composite_z": 5.0,
             "status": "exploratory_outlier"} for i in range(5)
        ])
        row = aggregate_player_scores(df).iloc[0].to_dict()
        self.assertNotIn("cheat_probability", row)
        self.assertEqual(row["account_verdict"], "manual_review_recommended")
        self.assertIsNone(row["null_p_value"])
        self.assertEqual(row["calibration_status"], "uncalibrated")

    def test_empirical_null_refuses_small_sample(self):
        controls = pd.DataFrame({"player": [f"p{i}" for i in range(10)],
                                 "time_control": ["blitz"] * 10, "max_sprt_llr": [1.] * 10})
        result = empirical_account_tail(10., controls, account_id="target", time_control="blitz")
        self.assertIsNone(result["null_p_value"])

    def test_empirical_null_add_one_and_ties(self):
        controls = pd.DataFrame({"player": [f"p{i}" for i in range(99)],
                                 "time_control": ["blitz"] * 99, "max_sprt_llr": [1.] * 99})
        result = empirical_account_tail(2., controls, account_id="target", time_control="blitz")
        self.assertEqual(result["null_p_value"], 0.01)
        result2 = empirical_account_tail(1., controls, account_id="target", time_control="blitz")
        self.assertEqual(result2["null_p_value"], 1.0)

    def test_control_account_cannot_calibrate_itself(self):
        controls = pd.DataFrame({"player": ["target"] + [f"p{i}" for i in range(99)],
                                 "time_control": ["blitz"] * 100, "max_sprt_llr": [100.] + [1.] * 99})
        result = empirical_account_tail(2., controls, account_id="target", time_control="blitz")
        self.assertEqual(result["n_independent_controls"], 99)

    def test_duplicate_control_account_rejected(self):
        controls = pd.DataFrame({"player": ["a", "a"], "time_control": ["blitz"] * 2,
                                 "max_sprt_llr": [1., 2.]})
        with self.assertRaises(ValueError):
            empirical_account_tail(2., controls, account_id="target", time_control="blitz")

    def test_group_split_leakage_rejected(self):
        manifest = pd.DataFrame([
            {"platform": "lichess", "player": "a", "label": 0,
             "label_source": "public_api", "label_checked_at": "2026-11-15", "split": split}
            for split in ("train", "test")
        ])
        with self.assertRaises(ValueError):
            validate_label_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
