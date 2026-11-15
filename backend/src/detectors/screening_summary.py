"""Simple, reproducible screening index for the default user report.

This is an explicitly heuristic 0-100 indicator, not P(cheating).
Only observations in this PGN are used; no labelled cases or calibration required.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


SIGNALS = (
    ("z_accuracy", "Move accuracy"),
    ("z_critical", "Difficult positions"),
    ("z_choice", "Move choices"),
    ("z_surprise", "Engine surprise"),
    ("z_timing", "Thinking time"),
    ("z_phase_shift", "Performance shifts"),
    ("z_contextual_toggle", "Situational changes"),
)


def _safe_number(value, default=0.0) -> float:
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def _label(score: int) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Some unusual play"
    if score < 75:
        return "Elevated"
    return "High"


def build_screening_summary(scored: pd.DataFrame, *, player: str | None = None) -> list[dict]:
    """Build a user-facing report with per-time-control indices and evidence shares.

    The score intentionally avoids multiplying by sqrt(games) or accumulated
    unvalidated LLRs, and thus resists raising a warning solely because the
    user supplied more ordinary games.
    """
    if scored.empty:
        return []
    frame = scored.copy()
    if player is not None:
        frame = frame[frame["player"].astype(str).str.casefold() == player.casefold()]
    rows = []
    for (username, tc), games in frame.groupby(["player", "time_control"], sort=False):
        if tc == "unknown":
            continue
        z = pd.to_numeric(games["composite_z"], errors="coerce")
        z = z[np.isfinite(z)]
        if z.empty:
            continue
        n = len(z)
        # A game needs multiple strong signals to cross the existing review tiers.
        flagged = int(games["status"].isin(("exploratory_outlier", "exploratory_review")).sum())
        outliers = int((games["status"] == "exploratory_outlier").sum())
        positive = z.clip(lower=0)
        highest = float(positive.max())
        peak = min(1.0, highest / 3.5)
        repeated = float((z >= 1.5).sum() / n)
        repeats = min(1.0, repeated / 0.20)
        flagged_rate = min(1.0, flagged / max(1, n) / 0.12)
        index = int(round(100 * (0.35 * peak + 0.30 * repeats + 0.35 * flagged_rate)))
        index = max(0, min(100, index))

        # Breakdown is *relative signal emphasis*, not % assisted moves.
        components = []
        for column, label in SIGNALS:
            if column not in games.columns:
                continue
            values = pd.to_numeric(games[column], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                continue
            # Mean positive signal over all evaluated games; avoids ranking a
            # channel by a single noisy spike.
            positive_mean = float(values.clip(lower=0).sum()) / n
            if positive_mean > 0:
                components.append((label, positive_mean))
        total = sum(weight for _, weight in components)
        shares = [
            {"signal": label, "share_percent": round(100 * weight / total)}
            for label, weight in sorted(components, key=lambda item: -item[1])
        ] if total > 0 else []
        rows.append({
            "player": str(username), "time_control": str(tc), "games": n,
            "suspicion_index": index, "suspicion_label": _label(index),
            "review_games": flagged, "outlier_games": outliers,
            "peak_game_score": round(highest, 2),
            "strong_game_fraction": round(repeated, 3),
            "indicator_shares": shares,
            "kind": "heuristic_screening_index",
            "probability_of_cheating": None,
            "explanation": "Relative screening score, not a probability or fair-play finding.",
        })
    return sorted(rows, key=lambda row: (-row["suspicion_index"], row["time_control"]))
