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

    # Many quiet, precise moves are ordinary chess. Only repeated *games*
    # support elevating the review priority, avoiding a single tactical game
    # or one excellent engine-first move becoming an account alert.
    if games < 8:
        label = "Limited history"
        interpretation = "Not enough games to assess repeated patterns."
    elif len(repeat_games) >= 3:
        label = "Review recommended"
        interpretation = "Several games contain multiple difficult engine-matching decisions."
    elif len(with_evidence) >= 2:
        label = "Some moves worth reviewing"
        interpretation = "There are isolated notable moves, but no consistent assisted-play pattern established."
    else:
        label = "No sustained indicators"
        interpretation = "No recurring difficult engine-matching pattern was found."

    return {
        "games": games,
        "time_controls": dict(sorted(controls.items())),
        "games_with_candidates": len(with_evidence),
        "games_with_repeated_candidates": len(repeat_games),
        "candidate_moves": candidates,
        "assessment": label,
        "interpretation": interpretation,
        "meaning": "Uncalibrated review priority; cannot establish cheating or innocence.",
    }
