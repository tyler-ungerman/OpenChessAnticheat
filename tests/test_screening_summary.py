"""Product screening index: no fake cheating probabilities, length robust."""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import pandas as pd

from backend.src.detectors.screening_summary import build_screening_summary

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_pgn_index", ROOT / "examples/analyze_pgn.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def games(n, z, status="no_outlier_detected"):
    return pd.DataFrame([
        {"player": "alice", "time_control": "blitz", "game_id": i,
         "composite_z": z, "status": status,
         "z_accuracy": max(0.0, z), "z_critical": 0.2,
         "z_choice": 0.1, "z_surprise": 0.0, "z_timing": 0.0,
         "z_phase_shift": 0.0, "z_contextual_toggle": 0.0}
        for i in range(n)
    ])


class ScreeningIndexTests(unittest.TestCase):
    def test_ordinary_games_do_not_become_high_risk_from_history_length(self):
        small = build_screening_summary(games(10, .7))[0]
        large = build_screening_summary(games(100, .7))[0]
        self.assertEqual(small["suspicion_index"], large["suspicion_index"])
        self.assertLess(small["suspicion_index"], 25)
        self.assertIsNone(small["probability_of_cheating"])

    def test_weak_peak_not_called_high_risk(self):
        frame = games(50, 0.15)
        frame.loc[49, "composite_z"] = 1.8
        result = build_screening_summary(frame)[0]
        self.assertLess(result["suspicion_index"], 25)
        self.assertEqual(result["outlier_games"], 0)

    def test_multiple_review_games_increase_index(self):
        frame = games(30, 0.1)
        frame.loc[0:8, "composite_z"] = 3.0
        frame.loc[0:8, "status"] = "exploratory_outlier"
        result = build_screening_summary(frame)[0]
        self.assertGreaterEqual(result["suspicion_index"], 75)

    def test_cli_prints_no_probability(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli._print_product_summary(build_screening_summary(games(25, .5)), False)
        value = output.getvalue()
        self.assertIn("Suspicion index:", value)
        self.assertIn("NOT a % chance of cheating", value)
        self.assertIn("Move accuracy", value)


if __name__ == "__main__":
    unittest.main()
