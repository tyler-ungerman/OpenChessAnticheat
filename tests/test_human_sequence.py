"""Human-choice sequence integration and time-control guards."""
import unittest

import chess
import pandas as pd

from backend.src.detectors.human_sequence import (
    load_human_prior, scan_human_sequences, _move_context,
)


def rows(game_id, tc, n=18):
    board = chess.Board()
    return [{
        "game_id": game_id, "player": "alice",
        "time_control": tc, "ply": 21 + 2*i,
        "move_number": 11 + i, "date": "2026.11.16",
        "fen_before": board.fen(), "move_uci": "e2e4",
        "move_san": "e4", "best_move": "e2e4",
        "shallow_best_move": "d2d4", "is_top1": True,
        "centipawn_loss": 0, "elo": 1500,
        "eval_before": 0, "time_spent": 3.0,
    } for i in range(n)]


class SequenceTests(unittest.TestCase):
    def test_bundled_model_and_human_sequences(self):
        model = load_human_prior()
        self.assertIsNotNone(model)
        frame = pd.DataFrame(rows(1, "180", 20) + rows(2, "600", 20))
        report = scan_human_sequences(frame, model, player="alice")
        self.assertEqual(report["eligible_games"], 1)
        self.assertEqual(report["ineligible_games"], 1)
        self.assertEqual(report["scored_moves"], 20)
        self.assertEqual(len(report["ranked_games"][0]["moves"]), 5)
        self.assertEqual(report["ranked_games"][0]["engine_first_count"], 5)
        self.assertIn("human_top1_probability", report["ranked_games"][0]["moves"][0])
        self.assertNotIn("cheat_probability", str(report))

    def test_forcing_tactics_not_called_quiet_evidence(self):
        board = chess.Board()
        self.assertEqual(_move_context(board.fen(), "e2e4"), "quiet decision")
        board.push_uci("e2e4")
        board.push_uci("d7d5")
        self.assertEqual(_move_context(board.fen(), "e4d5"), "capture")
        forced = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        self.assertTrue(forced.is_check())
        self.assertEqual(_move_context(forced.fen(), next(iter(forced.legal_moves)).uci()),
                         "responding to check")

    def test_same_sequence_score_but_truthful_quiet_breakdown(self):
        model = load_human_prior()
        frame = pd.DataFrame(rows(7, "180", 20))
        frame.loc[0, "move_uci"] = "not_legal"
        report = scan_human_sequences(frame, model, player="alice")
        candidate = report["ranked_games"][0]
        self.assertEqual(candidate["engine_first_count"], 5)
        self.assertIn("quiet_decision_count", candidate)
        self.assertEqual(
            candidate["tactical_or_constrained_count"] + candidate["quiet_decision_count"],
            5
        )
        self.assertGreaterEqual(candidate["windows_searched"], 1)

    def test_no_model_stays_optional(self):
        frame = pd.DataFrame(rows(1, "180"))
        report = scan_human_sequences(frame, None)
        self.assertEqual(report["ranked_games"], [])


if __name__ == "__main__":
    unittest.main()
