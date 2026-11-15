"""Collector contract tests, no external HTTP needed."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import chess.pgn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collect_public_labels", ROOT / "scripts/collect_public_labels.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CollectorTests(unittest.TestCase):
    def test_fairplay_status_only_as_platform_action(self):
        self.assertEqual(module.classify("chesscom", {"status": "closed:fair_play_violations"}),
                         (1, "platform_fair_play_closure"))
        self.assertEqual(module.classify("chesscom", {"status": "basic"}),
                         (0, "unconfirmed_not_clean"))
        self.assertEqual(module.classify("lichess", {"tosViolation": True}),
                         (1, "platform_tos_violation_unspecified"))
        self.assertEqual(module.classify("lichess", {"disabled": True}),
                         (0, "disabled_reason_unknown"))
        self.assertEqual(module.classify("lichess", {"username": "foo"}),
                         (0, "unconfirmed_not_clean"))

    def test_unknown_status_fails_closed(self):
        with self.assertRaises(ValueError):
            module.classify("chesscom", {"status": "closed:unknown"})

    def test_stable_account_splits_ignore_case(self):
        a = module.account_key("chesscom", "Sample_User")
        b = module.account_key("chesscom", "sample_user")
        self.assertEqual(module.split_for(a), module.split_for(b))
        self.assertIn(module.split_for(a), module.SPLITS)

    def test_pgn_seed_reader_excludes_bots(self):
        pgn = ('[Event "test"]\n[White "Sample_User"]\n[Black "EngineBot"]\n'
               '[BlackTitle "BOT"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n')
        with tempfile.TemporaryDirectory() as d:
            seed = Path(d) / "seed.pgn"
            seed.write_text(pgn)
            self.assertEqual(module.load_accounts(None, [seed], "chesscom", 10),
                             [("chesscom", "Sample_User")])

    def test_snapshot_and_offline_replay(self):
        class MockClient:
            def __init__(self, *args, **kwargs):
                pass
            def get(self, url, accept="application/json"):
                if "lichess.org/api/user/" in url:
                    return json.dumps({"tosViolation": True, "disabled": True}).encode()
                raise ValueError("No real network expected")
        client = module.PublicClient
        module.PublicClient = MockClient
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                accounts = root / "accounts.csv"
                accounts.write_text("platform,player\nlichess,sample_user\n")
                out = root / "snapshot"
                args = SimpleNamespace(accounts=accounts, seed_pgn=[], platform=None,
                                       max_accounts=10, out=out, user_agent="test@example.com",
                                       pause=0, fetch_games=False, max_games=50)
                self.assertEqual(module.run_collect(args), 0)
                self.assertEqual(module.run_verify(SimpleNamespace(out=out)), 0)
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(manifest["accounts"][0]["label"], 1)
                self.assertEqual(manifest["accounts"][0]["split"],
                                 module.split_for("lichess:sample_user"))
                with self.assertRaises(ValueError):
                    module.run_collect(args)  # Never overwrite immutable snapshot
        finally:
            module.PublicClient = client

    def test_pgn_checksum_corruption_detected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "pgn").mkdir()
            (root / "status").mkdir()
            status_bytes = json.dumps({"username": "tester"}).encode()
            (root / "status" / "tester.json").write_bytes(status_bytes)
            (root / "pgn" / "a.pgn").write_bytes(b"changed")
            manifest = {
                "schema": module.SCHEMA,
                "accounts": [{"platform": "lichess", "player": "tester",
                              "account_key": "lichess:tester",
                              "split": module.split_for("lichess:tester"),
                              "label": 0, "label_quality": "unconfirmed_not_clean",
                              "status_path": "status/tester.json",
                              "status_sha256": module.digest(status_bytes),
                              "pgn_path": "pgn/a.pgn",
                              "pgn_sha256": module.digest(b"original")}]
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                module.run_verify(SimpleNamespace(out=root))


if __name__ == "__main__":
    unittest.main()
