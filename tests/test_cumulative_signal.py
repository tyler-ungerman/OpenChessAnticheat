"""Cumulative-only warnings are distinct from independently unusual games."""
import unittest

import chess.pgn
import pandas as pd

from backend.src.detectors.statistical_z import aggregate_player_scores
from backend.src.ingestion.pgn import is_bot_game


class CumulativeAlertsTests(unittest.TestCase):
    def test_weak_accumulated_signal_is_not_statistical_outlier(self):
        rows = pd.DataFrame([
            {"game_id": f"g{i}", "date": f"2026-11-{i+1:02d}",
             "time_control": "rapid", "player": "target", "elo": 600,
             "acpl": 70.0, "color": "White", "composite_z": 0.8,
             "status": "no_outlier_detected"}
            for i in range(15)
        ])
        result = aggregate_player_scores(rows).iloc[0]
        self.assertGreater(result["max_sprt_llr"], 4.2)
        self.assertEqual(result["flagged_games"], 0)
        self.assertEqual(result["account_verdict"], "weak_cumulative_signal")

    def test_named_opponent_can_be_excluded_without_guessing(self):
        headers = chess.pgn.Headers({"Event": "Rated Rapid", "White": "target",
                                     "Black": "Chesslowbot", "TimeControl": "600"})
        self.assertFalse(is_bot_game(headers))
        self.assertTrue(is_bot_game(headers, extra_bot_names={"Chesslowbot"}))


if __name__ == "__main__":
    unittest.main()
