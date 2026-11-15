"""Train / inference feature compatibility and leakage tests (offline)."""
import math
import unittest

import numpy as np
import pandas as pd

from backend.src.detectors.pretrained_features import FEATURE_NAMES, from_chessfraud, from_local_game
from backend.src.detectors.pretrained_inference import (
    predict_feature_score, score_pgn_games, eligible_game_diagnostics,
)


class PretrainedContractTests(unittest.TestCase):
    def test_features_equivalent_across_public_and_local_formats(self):
        rows = []
        for i in range(15):
            rows.append({
                "player_id": 11, "game_id": "g1", "is_cheating_player_game": True,
                "time_control": "300+0", "is_used": True,
                "move_player": "e2e4" if i % 2 else "d2d4",
                "move_stockfish_15": "e2e4", "move_stockfish_9": "d2d4",
                "eval_before": 5, "half_move": 2*i+1,
                "centipawn_loss": i*3,
                "move_thinking_time": float(i+1),
            })
        source = pd.DataFrame(rows)
        train = from_chessfraud(source).iloc[0]
        local = pd.DataFrame([{
            "game_id": 7, "player": "alice", "time_control": "300+0",
            "move_uci": r["move_player"], "best_move": r["move_stockfish_15"],
            "shallow_best_move": r["move_stockfish_9"], "eval_before": r["eval_before"],
            "ply": r["half_move"],
            "centipawn_loss": r["centipawn_loss"],
            "time_spent": r["move_thinking_time"],
        } for r in rows])
        infer = from_local_game(local)
        for field in FEATURE_NAMES:
            self.assertAlmostEqual(float(train[field]), float(infer[field]), places=7)

    def test_no_label_as_feature(self):
        self.assertNotIn("is_cheating_player_game", FEATURE_NAMES)
        self.assertNotIn("player_id", FEATURE_NAMES)
        self.assertNotIn("is_cheating_move", FEATURE_NAMES)

    def test_too_few_moves_or_wrong_time_control_rejected(self):
        rows = pd.DataFrame([{
            "game_id": 1, "player": "alice", "time_control": "180+0",
            "move_uci": "e2e4", "best_move": "e2e4",
            "shallow_best_move": "e2e4", "eval_before": 0, "ply": 21,
            "centipawn_loss": 0, "time_spent": 2.0,
        }] * 20)
        model = {"schema": "openchess-chessfraud-game-logistic-v1",
                 "model_features": list(FEATURE_NAMES), "time_control": "300+0",
                 "impute_medians": [0.] * len(FEATURE_NAMES),
                 "scale_means": [0.] * len(FEATURE_NAMES),
                 "scale_std": [1.] * len(FEATURE_NAMES),
                 "coefficients": [0.] * len(FEATURE_NAMES), "intercept": 0.}
        self.assertEqual(score_pgn_games(rows, model), [])
        rows["time_control"] = "300+0"
        self.assertEqual(score_pgn_games(rows, model)[0]["experimental_assistance_score"], 50.)

    def test_tree_export_replay(self):
        from backend.src.detectors.pretrained_inference import predict_feature_score
        features = {name: 0.0 for name in FEATURE_NAMES}
        model = {"model_family": "random_forest",
                 "impute_medians": [0.] * len(FEATURE_NAMES),
                 "scale_means": [0.] * len(FEATURE_NAMES),
                 "scale_std": [1.] * len(FEATURE_NAMES),
                 "trees": [{"left": [1, -1, -1], "right": [2, -1, -1],
                            "feature": [0, -2, -2], "threshold": [0.5, -2., -2.],
                            "positive": [0.5, 0.1, 0.9]}]}
        self.assertAlmostEqual(predict_feature_score(features, model), 10.)
        features[FEATURE_NAMES[0]] = 1.0
        self.assertAlmostEqual(predict_feature_score(features, model), 90.)

    def test_raw_chesscom_time_controls_are_reported_and_not_conflated(self):
        sample = pd.DataFrame([{
            "game_id": index, "player": "alice", "time_control": tc,
            "move_uci": "e2e4", "best_move": "e2e4",
            "shallow_best_move": "e2e4", "eval_before": 0,
            "ply": ply, "centipawn_loss": 0, "time_spent": 2.0,
        } for index, tc, count in ((1, "180+0", 20), (2, "300", 20),
                                   (3, "300+0", 6), (4, "180+2", 20))
          for ply in range(1, count + 1)])
        summary = eligible_game_diagnostics(sample, player="alice")
        self.assertEqual(summary["game_time_controls"],
                         {"180+0": 1, "300": 1, "300+0": 1, "180+2": 1})
        self.assertEqual(summary["eligible_5min_games"], 1)
        self.assertEqual(summary["insufficient_moves_5min_games"], 1)
        self.assertEqual(summary["other_time_control_games"], 2)

    def test_logistic_export_replay(self):
        model = {"impute_medians": [0.] * len(FEATURE_NAMES),
                 "scale_means": [0.] * len(FEATURE_NAMES),
                 "scale_std": [1.] * len(FEATURE_NAMES),
                 "coefficients": [0.] * len(FEATURE_NAMES),
                 "intercept": 0.}
        self.assertAlmostEqual(predict_feature_score({f: 0. for f in FEATURE_NAMES}, model), 50.)


if __name__ == "__main__":
    unittest.main()
