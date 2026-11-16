"""Forcing tactical moves must not become human-assistance sequence evidence."""
import unittest

import chess
import numpy as np

from backend.src.detectors.sequence_policy import (
    informative_choice, peak_five_choice_window,
)


class ForcingSequenceTests(unittest.TestCase):
    def test_quiet_and_forcing_legal_positions(self):
        start = chess.Board()
        self.assertTrue(informative_choice(start.fen(), "e2e4"))
        self.assertFalse(informative_choice(start.fen(), "e2e5"))
        self.assertFalse(informative_choice(start.fen(), "f2f3") and start.is_check())
        start.push_uci("e2e4")
        start.push_uci("d7d5")
        self.assertFalse(informative_choice(start.fen(), "e4d5"))  # capture
        self.assertFalse(informative_choice("not a FEN", "e2e4"))

    def test_in_check_responses_discounted(self):
        board = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
        self.assertTrue(board.is_check())
        move = next(iter(board.legal_moves))
        self.assertFalse(informative_choice(board.fen(), move.uci()))

    def test_forcing_moves_never_inflate_window(self):
        surprise = np.array([1., 99., 1., 99., 1.])
        top = np.ones(5, dtype=int)
        probs = np.full(5, 0.2)
        mask = np.array([True, False, True, False, True])
        result = peak_five_choice_window(surprise, top, probs, mask)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["score"], 0.6)
        self.assertEqual(result["informative_count"], 3)
        self.assertEqual(result["top1_informative"], 3)
        self.assertAlmostEqual(result["expected_top1_informative"], 0.6)
        self.assertEqual(result["windows_examined"], 1)

    def test_all_forcing_combinations_are_rejected(self):
        result = peak_five_choice_window(
            np.array([5., 5., 5., 5., 5.]), np.ones(5),
            np.full(5, 0.15), np.zeros(5, dtype=bool)
        )
        self.assertIsNone(result)

    def test_five_consecutive_player_decisions_not_filtered_into_false_streak(self):
        result = peak_five_choice_window(
            np.array([2., 2., 2., 2., 2., 2.]), np.ones(6),
            np.full(6, 0.2),
            np.array([True, False, True, False, True, False])
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["start"], 0)
        self.assertEqual(result["windows_examined"], 1)


if __name__ == "__main__":
    unittest.main()
