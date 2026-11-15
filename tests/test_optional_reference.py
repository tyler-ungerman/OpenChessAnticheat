"""Opt-in proxy reference checks. No Stockfish or remote data needed."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backend.src.detectors.reference_compare import compare_accounts, read_proxy_reference


def reference():
    return {"schema": "openchess-proxy-null-v1",
            "status": "exploratory_proxy_reference_NOT_validated_clean_null",
            "statistic": "max_sprt_llr", "n_games_exact": 30,
            "elo_bucket_width": 200,
            "cohorts": {"chesscom:blitz:1400": {
                "n_accounts": 99, "sorted_scores": [1.0] * 99,
                "source": "unconfirmed_status_accounts_NOT_proven_clean"}}}


class OptionalReferenceTests(unittest.TestCase):
    def test_optional_is_not_implicit(self):
        # Default detector outputs are deliberately untouched by this opt-in module.
        raw = pd.DataFrame([{"player": "case", "time_control": "blitz", "games": 30,
                             "mean_elo": 1450, "max_sprt_llr": 3.0,
                             "account_verdict": "statistical_outlier"}])
        self.assertNotIn("proxy_tail_p_value", raw)
        out = compare_accounts(raw, reference(), platform="chesscom")
        self.assertEqual(out.iloc[0]["account_verdict"], "statistical_outlier")
        self.assertAlmostEqual(out.iloc[0]["proxy_tail_p_value"], 0.01)
        self.assertNotIn("cheat_probability", out.columns)

    def test_reject_mismatched_game_count(self):
        raw = pd.DataFrame([{"player": "case", "time_control": "blitz", "games": 20,
                             "mean_elo": 1450, "max_sprt_llr": 3.0}])
        out = compare_accounts(raw, reference(), platform="chesscom")
        self.assertEqual(out.iloc[0]["proxy_reference_status"], "game_count_mismatch")
        self.assertTrue(pd.isna(out.iloc[0]["proxy_tail_p_value"]))

    def test_reject_wrong_platform(self):
        raw = pd.DataFrame([{"player": "case", "time_control": "blitz", "games": 30,
                             "mean_elo": 1450, "max_sprt_llr": 3.0}])
        out = compare_accounts(raw, reference(), platform="lichess")
        self.assertEqual(out.iloc[0]["proxy_reference_status"], "no_matching_reference")

    def test_loader_rejects_invalid_cohort(self):
        ref = reference()
        ref["cohorts"]["chesscom:blitz:1400"]["n_accounts"] = 100
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ref.json"
            p.write_text(json.dumps(ref))
            with self.assertRaises(ValueError):
                read_proxy_reference(p)

    def test_ties_are_conservative(self):
        raw = pd.DataFrame([{"player": "case", "time_control": "blitz", "games": 30,
                             "mean_elo": 1450, "max_sprt_llr": 1.0}])
        out = compare_accounts(raw, reference(), platform="chesscom")
        self.assertEqual(out.iloc[0]["proxy_tail_p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
