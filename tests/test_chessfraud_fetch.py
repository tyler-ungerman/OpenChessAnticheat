"""Reproducible controlled-label benchmark acquisition tests (offline)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_chessfraud", ROOT / "scripts/fetch_chessfraud.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ChessFraudFetchTests(unittest.TestCase):
    def test_csv_label_inspection_and_source_hash(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "chess_fraud_tournament.csv"
            file.write_text("player_id,game_id,is_cheating_player_game\n"
                            "1,a,true\n1,a,true\n2,b,false\n")
            stats = module.describe_csv(file)
            self.assertEqual(stats["rows"], 3)
            self.assertEqual(stats["distinct_player_ids"], 2)
            self.assertEqual(stats["move_rows_from_assisted_player_games"], 2)
            self.assertEqual(len(module.sha256(file)), 64)

    def test_offline_verification_detects_corrupted_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entries = []
            for name in module.FILES:
                path = root / name
                path.write_text("player_id,game_id,is_cheating_player_game\n1,a,true\n")
                entries.append({"file": name, "sha256": module.sha256(path),
                                "summary": module.describe_csv(path)})
            (root / "manifest.json").write_text(json.dumps({"schema": module.SCHEMA,
                                                             "entries": entries}))
            module.verify(root)
            (root / module.FILES[0]).write_text("tampered")
            with self.assertRaises(ValueError):
                module.verify(root)


if __name__ == "__main__":
    unittest.main()
