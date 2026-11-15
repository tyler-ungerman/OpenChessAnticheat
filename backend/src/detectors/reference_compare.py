"""Opt-in comparison to a frozen *proxy* reference of unconfirmed-status accounts.

Never replaces heuristic screening, invents a cheating probability, or silently
assumes an unconfirmed platform account is clean.
"""
from __future__ import annotations

import bisect
import json
import math
from pathlib import Path

import pandas as pd

SCHEMA = "openchess-proxy-null-v1"
STATISTICS = {"max_sprt_llr", "max_cusum", "max_single_z"}


def read_proxy_reference(path: str | Path) -> dict:
    """Validate a snapshot before comparison; malformed files fail explicitly."""
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    if reference.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference schema")
    if reference.get("statistic") not in STATISTICS:
        raise ValueError("Unsupported reference statistic")
    if reference.get("status") != "exploratory_proxy_reference_NOT_validated_clean_null":
        raise ValueError("Unexpected reference provenance; expected unconfirmed-status proxy")
    if int(reference.get("n_games_exact", 0)) < 1:
        raise ValueError("Reference game count must be positive")
    if int(reference.get("elo_bucket_width", 0)) < 1:
        raise ValueError("Reference rating bucket width must be positive")
    for key, group in reference.get("cohorts", {}).items():
        values = group.get("sorted_scores")
        if not isinstance(values, list) or len(values) < 99:
            raise ValueError(f"Reference cohort {key!r} is undersized")
        if group.get("n_accounts") != len(values):
            raise ValueError(f"Reference cohort {key!r} has inconsistent sample size")
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Reference cohort {key!r} contains invalid scores")
        if values != sorted(values):
            raise ValueError(f"Reference cohort {key!r} is not sorted")
    return reference


def compare_accounts(players: pd.DataFrame, reference: dict, *, platform: str) -> pd.DataFrame:
    """Return account summaries with opt-in proxy tail ranks, preserving original verdicts."""
    if platform not in {"chesscom", "lichess"}:
        raise ValueError("Specify the platform explicitly: chesscom or lichess")
    if reference.get("schema") != SCHEMA:
        raise ValueError("Unsupported reference schema")
    result = players.copy()
    if result.empty:
        return result
    statistic = reference["statistic"]
    count = int(reference["n_games_exact"])
    width = int(reference["elo_bucket_width"])
    cohorts = reference["cohorts"]
    additions = []
    for row in result.to_dict(orient="records"):
        metadata = {"proxy_tail_p_value": None, "proxy_reference_accounts": 0,
                    "proxy_reference_stratum": None, "proxy_reference_status": "no_matching_reference"}
        elo = row.get("mean_elo")
        if int(row.get("games") or 0) != count:
            metadata["proxy_reference_status"] = "game_count_mismatch"
        elif elo is None or not math.isfinite(float(elo)) or float(elo) <= 0:
            metadata["proxy_reference_status"] = "missing_rating"
        else:
            stratum = f"{platform}:{row['time_control']}:{int(float(elo)) // width * width}"
            metadata["proxy_reference_stratum"] = stratum
            group = cohorts.get(stratum)
            if group:
                values = group["sorted_scores"]
                val = float(row[statistic])
                if math.isfinite(val):
                    # Conservative inclusion of ties, plus-one finite sample rank.
                    exceedances = len(values) - bisect.bisect_left(values, val)
                    metadata["proxy_tail_p_value"] = (exceedances + 1) / (len(values) + 1)
                    metadata["proxy_reference_accounts"] = len(values)
                    metadata["proxy_reference_status"] = "compared_unconfirmed_proxy"
        additions.append(metadata)
    for column in additions[0]:
        result[column] = [item[column] for item in additions]
    return result
