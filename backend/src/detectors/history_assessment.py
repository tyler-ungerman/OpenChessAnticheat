"""A concise chess-history assessment derived from explainable move evidence.

No calibration, new runtime libraries, or pretrained weights. This is a
review-priority summary, not a probability of cheating or a claim of guilt.
"""
from __future__ import annotations

from collections import Counter


def summarize_history(evidence: list[dict]) -> dict:
    games = len(evidence)
    controls = Counter(str(item.get("time_control") or "unknown") for item in evidence)
    with_evidence = [item for item in evidence if item["candidate_count"] > 0]
    repeat_games = [item for item in evidence if item["candidate_count"] >= 2]
    candidates = sum(item["candidate_count"] for item in evidence)
    game_alerts = sum(
        item.get("game_status") in {"exploratory_outlier", "exploratory_review"}
        for item in evidence
    )

    # A large archive of genuine human chess naturally contains many precise
    # engine-first moves. Counts alone are NOT an account-level cheating signal.
    # Require concordance with multiple independent game-level review tiers.
    if games < 8:
        label = "Limited history"
        interpretation = "Not enough games to assess repeated patterns."
    elif game_alerts >= 2 and len(repeat_games) >= 2:
        label = "Review recommended"
        interpretation = "Multiple independently unusual games also contain repeated noteworthy decisions."
    elif len(with_evidence) >= 1:
        label = "Moves worth examining"
        interpretation = "Some decisions merit review, but no supported account-level assisted-play pattern emerged."
    else:
        label = "No strong indicators"
        interpretation = "The analysis did not identify notable move-level evidence."

    return {
        "games": games,
        "time_controls": dict(sorted(controls.items())),
        "games_with_candidates": len(with_evidence),
        "games_with_repeated_candidates": len(repeat_games),
        "candidate_moves": candidates,
        "game_review_alerts": game_alerts,
        "assessment": label,
        "interpretation": interpretation,
        "meaning": "Uncalibrated review priority; cannot establish cheating or innocence.",
    }
