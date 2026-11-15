"""Frozen equal-length chess account windows (no engine required)."""
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("score_public_snapshot",
                                               ROOT / "scripts/score_public_snapshot.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class WindowTests(unittest.TestCase):
    def test_exact_game_window_and_time_control_separation(self):
        with tempfile.TemporaryDirectory() as d:
            pgn = Path(d) / "source.pgn"
            with pgn.open("w") as out:
                for i in range(35):
                    game = chess.pgn.Game()
                    game.headers["Event"] = "Rated Game"
                    game.headers["White"] = "sample_user"
                    game.headers["Black"] = "different_user"
                    game.headers["TimeControl"] = "300+0"
                    game.headers["Date"] = f"2026.10.{i % 28 + 1:02d}"
                    game.headers["Site"] = f"https://lichess.org/{i:08d}"
                    game.add_variation(chess.Move.from_uci("e2e4"))
                    print(game, file=out, end="\n\n")
                game = chess.pgn.Game()
                game.headers["Event"] = "Rated Game"
                game.headers["White"] = "sample_user"
                game.headers["Black"] = "different_user"
                game.headers["TimeControl"] = "600+0"
                game.add_variation(chess.Move.from_uci("e2e4"))
                print(game, file=out, end="\n\n")
            windows = module.normalized_windows(pgn, "sample_user", 30)
            self.assertEqual(set(windows), {"blitz"})
            stream = io.StringIO(windows["blitz"].decode())
            n = 0
            while chess.pgn.read_game(stream) is not None:
                n += 1
            self.assertEqual(n, 30)


if __name__ == "__main__":
    unittest.main()
