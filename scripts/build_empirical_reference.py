#!/usr/bin/env python3
"""Build a frozen empirical reference from account-scoring manifests.

This measures extremeness among *unconfirmed-status* accounts, not the actual
false-positive rate on proven clean humans or probability of cheating.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratum(row: dict, rating_width: int, exact_games: int) -> str | None:
    """Game-window equality is essential for maxima/CUSUM comparability."""
    if row["games"] != exact_games:
        return None
    elo = row.get("mean_elo")
    if elo is None or not math.isfinite(float(elo)) or float(elo) <= 0:
        return None
    return f"{row['platform']}:{row['time_control']}:{int(float(elo)) // rating_width * rating_width}"


def make_reference(
    score_manifest: dict,
    *,
    source_sha256: str,
    statistic: str = "max_sprt_llr",
    n_games: int = 30,
    rating_width: int = 200,
    minimum_accounts: int = 99,
) -> dict:
    if statistic not in {"max_sprt_llr", "max_cusum", "max_single_z"}:
        raise ValueError("Unsupported statistic")
    if n_games < 1 or rating_width < 1 or minimum_accounts < 99:
        raise ValueError("Invalid stratum settings; at least 99 controls required")
    if score_manifest.get("failures"):
        raise ValueError("Incomplete scoring manifest; resolve failures first")
    samples = defaultdict(dict)
    held_out = defaultdict(list)
    seen = {}
    for row in score_manifest.get("rows", []):
        key = row["account_key"]
        if key in seen and seen[key] != row["split"]:
            raise ValueError("Cross-split account leakage")
        seen[key] = row["split"]
        grp = stratum(row, rating_width, n_games)
        if grp is None:
            continue
        val = float(row[statistic])
        if not math.isfinite(val):
            continue
        if row["split"] == "calibration" and row["label"] == 0:
            if row["label_quality"] not in {"unconfirmed_not_clean", "disabled_reason_unknown"}:
                continue
            if key in samples[grp]:
                raise ValueError("Duplicate account in reference stratum")
            samples[grp][key] = val
        elif row["split"] == "test":
            held_out[grp].append((key, row["label"], val))
    cohorts = {}
    test_rows = []
    for group, values in sorted(samples.items()):
        if len(values) < minimum_accounts:
            continue
        ordered = sorted(float(x) for x in values.values())
        cohorts[group] = {"n_accounts": len(ordered), "sorted_scores": ordered,
                          "account_hashes": sorted(hashlib.sha256(key.encode("utf-8")).hexdigest()
                                                   for key in values),
                          "source": "unconfirmed_status_accounts_NOT_proven_clean"}
        for key, label, value in held_out[group]:
            # Frozen calibration reference; held-out test accounts never enter the ECDF.
            p = (1 + sum(x >= value for x in ordered)) / (len(ordered) + 1)
            test_rows.append({"account_key": key, "platform_status_label": label,
                              "stratum": group, "score": value, "proxy_tail_p_value": p})
    return {
        "schema": "openchess-proxy-null-v1",
        "status": "exploratory_proxy_reference_NOT_validated_clean_null",
        "scored_manifest_sha256": source_sha256,
        "statistic": statistic,
        "n_games_exact": n_games,
        "elo_bucket_width": rating_width,
        "minimum_accounts_per_bucket": minimum_accounts,
        "cohorts": cohorts,
        "held_out_test": sorted(test_rows, key=lambda r: r["account_key"]),
        "limitations": [
            "Not an estimate of cheating probability or true false-positive rate",
            "Platform unclosed/unconfirmed status is not verified clean",
            "Only same-platform, time-control, rating-bucket, exact-game-count cohorts compared",
            "Account-level selection bias and platform-action labels remain unmeasured",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="score_manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="NEW .json file")
    parser.add_argument("--statistic", default="max_sprt_llr",
                        choices=("max_sprt_llr", "max_cusum", "max_single_z"))
    parser.add_argument("--games", type=int, default=30, help="Exact number of player games per TC")
    parser.add_argument("--elo-width", type=int, default=200)
    parser.add_argument("--min-controls", type=int, default=99)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite existing reference")
    result = make_reference(json.loads(args.scores.read_text(encoding="utf-8")),
                            source_sha256=fingerprint(args.scores), statistic=args.statistic,
                            n_games=args.games, rating_width=args.elo_width,
                            minimum_accounts=args.min_controls)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    n = len(result["cohorts"])
    print(f"{n} qualifying cohorts, {len(result['held_out_test'])} held-out account observations.")
    if n == 0:
        print("Insufficient matched reference accounts: no empirical calibration available.")
        return 2
    print("Research-only proxy null. No cheating probability or validated FPR.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
