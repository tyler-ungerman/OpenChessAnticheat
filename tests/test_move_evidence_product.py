"""Move-evidence product report: quiet difficult moves without invented confidence."""
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

import pandas as pd

from backend.src.detectors.move_evidence import build_game_evidence

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_pgn_product", ROOT / "examples/analyze_pgn.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def move(ply=35, gap=120, quiet=True, top1=True, seconds=3):
    return {
        "game_id": 7, "player": "Alice", "ply": ply, "move_number": (ply+1)//2,
        "move_san": "Nd5", "move_uci": "c3d5", "fen_before": "test-fen",
        "best_move": "c3d5", "is_top1": top1, "is_capture": not quiet,
        "is_check": False, "legal_moves_count": 25, "eval_gap": gap,
        "eval_before": 20, "centipawn_loss": 0, "time_spent": seconds,
    }


class ProductEvidenceTests(unittest.TestCase):
    def test_quiet_move_surfaces_as_concrete_evidence(self):
        scored = pd.DataFrame([{"game_id": 7, "player": "Alice", "opponent": "Bob",
                                "time_control": "blitz", "composite_z": 1.5,
                                "date": "2026-11-15"}])
        result = build_game_evidence(pd.DataFrame([move()]), scored, player="alice")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["candidate_moves"][0]["san"], "Nd5")
        self.assertEqual(result[0]["candidate_moves"][0]["fen_before"], "test-fen")
        self.assertIn("3.0s", result[0]["candidate_moves"][0]["reason"])
        self.assertNotIn("cheat_probability", str(result))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            report._print_move_evidence(result, False)
        self.assertIn("Move 18: Nd5", output.getvalue())

    def test_obvious_moves_not_evidence(self):
        scored = pd.DataFrame([{"game_id": 7, "player": "Alice", "composite_z": 2.0}])
        rows = [move(ply=35, quiet=False), move(ply=37, gap=5),
                move(ply=39, top1=False)]
        result = build_game_evidence(pd.DataFrame(rows), scored, player="Alice")
        self.assertEqual(result[0]["candidate_moves"], [])
        self.assertEqual(result[0]["candidate_count"], 0)

    def test_player_filter_and_rank_order(self):
        scored = pd.DataFrame([
            {"game_id": 7, "player": "Alice", "composite_z": 1.5},
            {"game_id": 8, "player": "Other", "composite_z": 4.0},
        ])
        ev = pd.DataFrame([move()])
        result = build_game_evidence(ev, scored, player="alice")
        self.assertEqual([g["game_id"] for g in result], [7])


if __name__ == "__main__":
    unittest.main()
