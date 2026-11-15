"""Account-disjoint proxy reference: no p-values without sufficient matched accounts."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_empirical_reference",
                                               ROOT / "scripts/build_empirical_reference.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def row(i, split="calibration", label=0, score=1.0, games=30, tc="blitz"):
    return {"account_key": f"chesscom:p{i}", "platform": "chesscom", "player": f"p{i}",
            "split": split, "label": label, "label_quality": "unconfirmed_not_clean",
            "time_control": tc, "games": games, "mean_elo": 1450,
            "max_sprt_llr": score, "max_cusum": score, "max_single_z": score}


class ReferenceTests(unittest.TestCase):
    def test_exactly_ninety_nine_controls_and_holdout(self):
        manifest = {"rows": [row(i) for i in range(99)] +
                    [row(1000, split="test", label=1, score=10.0)], "failures": []}
        result = module.make_reference(manifest, source_sha256="abc")
        self.assertEqual(len(result["cohorts"]), 1)
        self.assertEqual(result["held_out_test"][0]["proxy_tail_p_value"], 0.01)
        self.assertEqual(result["cohorts"]["chesscom:blitz:1400"]["n_accounts"], 99)

    def test_fails_with_ninety_eight_controls(self):
        result = module.make_reference({"rows": [row(i) for i in range(98)],
                                        "failures": []}, source_sha256="abc")
        self.assertEqual(result["cohorts"], {})

    def test_never_mix_game_lengths_or_time_controls(self):
        rows = [row(i) for i in range(98)]
        rows.extend([row(98, games=10), row(99, tc="rapid")])
        result = module.make_reference({"rows": rows, "failures": []}, source_sha256="abc")
        self.assertFalse(result["cohorts"])

    def test_never_use_positive_as_null(self):
        rows = [row(i) for i in range(98)] + [row(98, label=1)]
        result = module.make_reference({"rows": rows, "failures": []}, source_sha256="abc")
        self.assertFalse(result["cohorts"])

    def test_incomplete_scoring_refused(self):
        with self.assertRaises(ValueError):
            module.make_reference({"rows": [], "failures": [{"reason": "bad"}]},
                                  source_sha256="abc")

    def test_cross_split_account_refused(self):
        with self.assertRaises(ValueError):
            module.make_reference({"rows": [row(1), row(1, split="test")]},
                                  source_sha256="abc")


if __name__ == "__main__":
    unittest.main()
