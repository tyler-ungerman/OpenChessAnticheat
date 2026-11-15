"""PGN product screening should prioritize explainable evidence across all games."""
import unittest

import pandas as pd

from backend.src.detectors.move_evidence import build_game_evidence
from backend.src.detectors.history_assessment import summarize_history


def move(game_id, ply, *, matching=True, gap=100):
    return {
        "game_id": game_id, "player": "alice", "ply": ply,
        "move_number": (ply + 1) // 2,
        "move_san": "Nd5", "move_uci": "c3d5",
        "fen_before": "test-fen", "best_move": "c3d5",
        "is_top1": matching, "is_capture": False, "is_check": False,
        "legal_moves_count": 20, "eval_gap": gap, "eval_before": 40,
        "centipawn_loss": 0, "time_spent": 3,
    }


class EvidencePriorityTests(unittest.TestCase):
    def test_evidence_game_beats_statistically_high_empty_game(self):
        scored = pd.DataFrame([
            {"game_id": 1, "player": "alice", "composite_z": 3.9,
             "date": "2026-11-15", "time_control": "blitz"},
            {"game_id": 2, "player": "alice", "composite_z": 0.7,
             "date": "2026-11-15", "time_control": "rapid"},
        ])
        moves = pd.DataFrame([
            move(1, 30, matching=False),
            move(2, 31),
            move(2, 41),
        ])
        result = build_game_evidence(moves, scored, player="alice",
                                     top_games=1, top_moves=3)
        self.assertEqual(result[0]["game_id"], 2)
        self.assertEqual(result[0]["candidate_count"], 2)

    def test_opening_matches_not_evidence(self):
        scored = pd.DataFrame([{"game_id": 1, "player": "alice", "composite_z": 2.0}])
        moves = pd.DataFrame([move(1, ply=9)])
        result = build_game_evidence(moves, scored)
        self.assertEqual(result[0]["candidate_count"], 0)

    def test_account_assessment_counts_repeated_games(self):
        records = [
            {"candidate_count": 2, "time_control": "blitz", "game_status": "exploratory_review"},
            {"candidate_count": 2, "time_control": "rapid", "game_status": "exploratory_review"},
            {"candidate_count": 2, "time_control": "bullet", "game_status": "no_outlier_detected"},
        ] + [{"candidate_count": 0, "time_control": "blitz"}] * 7
        result = summarize_history(records)
        self.assertEqual(result["assessment"], "Review recommended")
        self.assertEqual(result["games_with_repeated_candidates"], 3)
        self.assertEqual(result["candidate_moves"], 6)
        self.assertEqual(result["game_review_alerts"], 2)
        self.assertNotIn("cheat_probability", result)

    def test_many_precise_games_without_game_level_alerts_do_not_imply_cheating(self):
        records = [{"candidate_count": 3, "time_control": "rapid",
                    "game_status": "no_outlier_detected"} for _ in range(100)]
        summary = summarize_history(records)
        self.assertEqual(summary["assessment"], "Moves worth examining")
        self.assertEqual(summary["game_review_alerts"], 0)

    def test_single_good_game_does_not_trigger_account_review(self):
        records = [{"candidate_count": 4, "time_control": "rapid"}]
        records += [{"candidate_count": 0, "time_control": "rapid"}] * 29
        result = summarize_history(records)
        self.assertEqual(result["assessment"], "Moves worth examining")


if __name__ == "__main__":
    unittest.main()
