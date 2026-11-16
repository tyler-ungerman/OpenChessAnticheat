#!/usr/bin/env python3
"""Compare deployed and forcing-aware five-move signals on controlled assistance.

No fitting, tuning, or new model. Uses the *already bundled* human prior,
the same 5-move policy as PGN inference, and disjoint ChessFraud participant
cohorts. Both strategies score precisely the same eligible player-games.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.src.detectors.human_prior import (
    frame_features, predict_human_top1, human_surprise_scores,
)
from backend.src.detectors.human_sequence import load_human_prior
from backend.src.detectors.sequence_policy import informative_choice, peak_five_choice_window
from experiment_human_prior import real_tournament_holdout, REPO, REVISION


def _auc(values, scores):
    if len(set(values)) != 2:
        raise ValueError("Missing class in benchmark cohort")
    return round(float(roc_auc_score(values, scores)), 4)


def score_group(df: pd.DataFrame, model: dict) -> dict:
    df = df[df["is_used"].fillna(False).astype(bool)].copy()
    df = df.dropna(subset=["position_before", "move_player", "move_stockfish_15"])
    df = df[pd.to_numeric(df["half_move"], errors="coerce") >= 21]
    if df.empty:
        raise ValueError("No eligible controlled positions")
    features = frame_features(df)
    p = predict_human_top1(features, model)
    top = (df["move_player"].astype(str).to_numpy() ==
           df["move_stockfish_15"].astype(str).to_numpy()).astype(int)
    surprise = human_surprise_scores(top, p)
    quiet = np.asarray([informative_choice(fen, uci)
                        for fen, uci in zip(df["position_before"], df["move_player"])], dtype=bool)
    df["top"] = top
    df["surprise"] = surprise
    df["p"] = p
    df["quiet"] = quiet

    entries = []
    excluded_no_quiet = 0
    forced_in_original_top = 0
    original_windows = 0
    for _, game in df.groupby(["player_id", "game_id"], sort=False):
        game = game.sort_values("half_move")
        if len(game) < 12:
            continue
        y = int(game["is_cheating_player_game"].iloc[0])
        t = game["top"].to_numpy(dtype=int)
        s = game["surprise"].to_numpy(dtype=float)
        pp = game["p"].to_numpy(dtype=float)
        q = game["quiet"].to_numpy(dtype=bool)
        start_old = int(np.argmax(np.convolve(s, np.ones(5) / 5, mode="valid")))
        old_score = float(s[start_old:start_old + 5].mean())
        old_engine = float(np.convolve(t, np.ones(5) / 5, mode="valid").max())
        forced_in_original_top += int((~q[start_old:start_old+5]).sum())
        original_windows += 5

        selected = peak_five_choice_window(s, t, pp, q)
        if selected is None:
            excluded_no_quiet += 1
            continue
        new_score = selected["score"]
        # Same quiet-only eligible windows and same fixed denominator for baseline.
        quiet_baseline = peak_five_choice_window(t.astype(float), t, pp, q)
        discounted = peak_five_choice_window(
            s, t, pp, q, forcing_weight=0.25
        )
        entries.append({
            "assisted": y, "old_model": old_score, "old_engine": old_engine,
            "quiet_model": new_score, "quiet_engine": quiet_baseline["score"],
            "discounted_model": discounted["score"],
            "old_fully_engine": bool((t[start_old:start_old+5] == 1).all()),
            "new_informative_moves": selected["informative_count"],
        })
    result = pd.DataFrame(entries)
    if result.empty:
        raise ValueError("No eligible games after quiet filtering")
    y = result["assisted"].astype(int)
    metrics = {
        "eligible_player_games": int(len(result)),
        "assisted_games": int(y.sum()),
        "unassisted_games": int((1-y).sum()),
        "excluded_games_without_three_quiet_choices": excluded_no_quiet,
        "forcing_fraction_in_original_top_windows": round(forced_in_original_top / max(1, original_windows), 4),
        "scores": {
            k: {"auc": _auc(y, result[k]), "average_precision":
                round(float(average_precision_score(y, result[k])), 4)}
            for k in ("old_engine", "old_model", "quiet_engine", "quiet_model", "discounted_model")
        },
        "unassisted_original_perfect_five_rate": round(
            float(result.loc[y==0, "old_fully_engine"].mean()), 4),
    }
    return metrics


def main():
    model = load_human_prior()
    if model is None:
        raise RuntimeError("Bundled human-choice model missing")
    tournament = load_dataset(REPO, "chess_fraud", split="full",
                              revision=REVISION).to_pandas()
    results = {}
    for split in ("test", "validation"):
        group, players = real_tournament_holdout(tournament, split=split)
        metrics = score_group(group, model)
        metrics["players"] = len(players)
        results[split] = metrics
    # This is a development comparison, not an untouched external validation.
    report = {
        "experiment": "forcing-window-compare-v1",
        "cohorts": results,
        "note": "Held-out players from original human prior, but scoring policy refined after observing real examples; comparative exploratory evidence.",
        "deployment_decision": (
            "DISCOUNTED_POLICY_SUPPORTED" if all(
                results[k]["scores"]["discounted_model"]["auc"] >=
                results[k]["scores"]["old_model"]["auc"] - 0.02
                for k in results
            ) else "DISCOUNTED_POLICY_LOSES_TOO_MUCH_AUC"
        ),
        "candidate_note": "Only one additional choice tested: 25% weight for forcing decisions, requiring at least 3 quiet moves per window.",
    }
    out = ROOT / "data" / "forcing_window_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
