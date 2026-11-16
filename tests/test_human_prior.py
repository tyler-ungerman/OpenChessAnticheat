"""Human-choice position features cannot accidentally include move/label leakage."""
import unittest

import chess
import numpy as np
import pandas as pd

from backend.src.detectors.human_prior import (
    FEATURE_NAMES, frame_features, human_surprise_scores,
)


class HumanPriorTests(unittest.TestCase):
    def test_same_position_features_despite_opposite_observed_moves(self):
        b = chess.Board()
        base = {
            "position_before": b.fen(), "move_stockfish_15": "e2e4",
            "move_stockfish_9": "e2e4", "player_elo": 1500,
            "opponent_elo": 1400, "half_move": 1, "time_control": "180+0",
            "eval_before": 0,
        }
        a = {**base, "move_player": "e2e4", "is_cheating_move": True}
        c = {**base, "move_player": "d2d4", "is_cheating_move": False}
        features = frame_features(pd.DataFrame([a,c]))
        self.assertEqual(features.iloc[0].tolist(), features.iloc[1].tolist())
        self.assertEqual(len(FEATURE_NAMES), 10)
        self.assertNotIn("move_player", FEATURE_NAMES)

    def test_human_surprise_sharper_for_rare_engine_match(self):
        scores = human_surprise_scores(np.array([1, 1, 0]), np.array([0.1, 0.8, 0.1]))
        self.assertGreater(scores[0], scores[1])
        self.assertLess(scores[2], scores[1])


if __name__ == "__main__":
    unittest.main()
