"""Actionable output regression tests, no external Stockfish needed."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_pgn", ROOT / "examples/analyze_pgn.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def test_games_ranked_even_below_outlier_threshold(self):
        scored = pd.DataFrame([
            {"game_id": "a", "date": "2026-10-01", "opponent": "one", "time_control": "rapid",
             "composite_z": 0.3, "capped_acpl": 67.0, "top1_rate": 0.21,
             "z_accuracy": 0.1, "z_critical": 0.0, "z_choice": 0.0, "z_surprise": 0.0,
             "z_timing": 0.0, "z_phase_shift": 0.0, "z_contextual_toggle": 0.0},
            {"game_id": "b", "date": "2026-10-02", "opponent": "two", "time_control": "rapid",
             "composite_z": 1.3, "capped_acpl": 35.0, "top1_rate": 0.58,
             "z_accuracy": 2.1, "z_critical": 1.4, "z_choice": 0.3, "z_surprise": 0.1,
             "z_timing": 0.2, "z_phase_shift": 0.0, "z_contextual_toggle": 0.0},
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report._print_diagnostic_games(scored, 2, False)
        text = output.getvalue()
        self.assertIn("Games to inspect", text)
        self.assertIn("two", text)
        self.assertIn("accuracy 2.1", text)
        self.assertLess(text.index("two"), text.index("one"))

    def test_cumulative_outlier_explained_and_capped_acpl_shown(self):
        rows = pd.DataFrame([{
            "player": "target", "time_control": "rapid",
            "account_verdict": "statistical_outlier", "games": 49,
            "mean_elo": 553, "mean_acpl": 384.6, "mean_capped_acpl": 90.0,
            "median_game_acpl": 80, "top1_rate": .35,
            "max_sprt_llr": 4.32, "max_cusum": 5.21, "max_single_z": 1.76,
            "flagged_games": 0, "flagged_pct": 0, "null_p_value": None
        }])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report._print_player_cards(rows, False)
        text = output.getvalue()
        self.assertIn("Capped ACPL", text)
        self.assertIn("inflated", text)
        self.assertIn("ONLY from accumulated", text)
        self.assertIn("no individual game", text)

    def test_move_quality_detects_extreme_losses(self):
        moves = pd.DataFrame([
            {"player": "target", "centipawn_loss": 3000, "move_time": None},
            {"player": "target", "centipawn_loss": 2, "move_time": 5},
            {"player": "other", "centipawn_loss": 0, "move_time": 5},
        ])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report._print_move_quality(moves, "target", False)
        text = output.getvalue()
        self.assertIn("CPL >1000: 1", text)
        self.assertIn("50%", text)


if __name__ == "__main__":
    unittest.main()
