"""Detector composition root: registers every built-in algorithm with the registry.

Importing this module populates :data:`src.detectors.base.DETECTOR_REGISTRY`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import DetectorConfig, register_detector
from .game_style import GameInput, evaluate_style_deviation, game_inputs_from_eval_df
from .hmm_toggle import fit_game_hmm_toggle
from .statistical_z import aggregate_player_scores, score_games


@register_detector(
    "statistical_z_cusum",
    description="7-Signal Dual-Anchored (Self + Control Elo-Cohort) Modified Z-scores with CUSUM & SPRT aggregation.",
    default_enabled=True,
)
def run_statistical_detector(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    cfg = DetectorConfig(
        flag_z_threshold=float(params.get("flag_z_threshold", 2.5)),
        review_z_threshold=float(params.get("review_z_threshold", 1.8)),
    )
    scored_df = score_games(features_df, eval_df=eval_df, config=cfg)
    player_df = aggregate_player_scores(
        scored_df, features_df=features_df, window=int(params.get("window", 5)), config=cfg
    )
    return {
        "algorithm": "statistical_z_cusum",
        "player_summary": player_df.where(player_df.notna(), None).to_dict(orient="records"),
        "game_scores": scored_df.where(scored_df.notna(), None).to_dict(orient="records"),
    }


@register_detector(
    "hmm_toggle_detector",
    description="2-State Viterbi Hidden Markov Model (Human vs. Assisted) for intra-game engine toggling streaks.",
    default_enabled=True,
)
def run_hmm_toggle_detector(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    min_streak = int(params.get("min_assisted_streak", 8))
    min_llr = float(params.get("min_hmm_llr", 4.0))

    if eval_df is None or eval_df.empty:
        return {"algorithm": "hmm_toggle_detector", "player_summary": [], "game_scores": []}

    rows = []
    valid_plies = eval_df[eval_df["centipawn_loss"].notna() & (eval_df["legal_moves_count"] > 1)]
    for (gid, color), grp in valid_plies.groupby(["game_id", "color"], sort=False):
        first = grp.iloc[0]
        res = fit_game_hmm_toggle(grp)
        status = (
            "flagged_toggle"
            if (res["hmm_max_streak"] >= min_streak and res["hmm_llr"] >= min_llr)
            else ("review_toggle" if res["hmm_max_streak"] >= 6 else "normal")
        )
        rows.append(
            {
                "game_id": gid,
                "date": first.get("date"),
                "time_control": first.get("time_control"),
                "player": first.get("player"),
                "color": color,
                "elo": first.get("elo"),
                **res,
                "status": status,
            }
        )

    game_df = pd.DataFrame(rows).sort_values(["hmm_llr", "hmm_max_streak"], ascending=[False, False])
    if game_df.empty:
        return {"algorithm": "hmm_toggle_detector", "player_summary": [], "game_scores": []}

    player_rows = []
    for player, grp in game_df.groupby("player", sort=False):
        player_rows.append(
            {
                "player": player,
                "games": len(grp),
                "mean_assisted_frac": round(float(grp["hmm_assisted_frac"].mean()), 4),
                "max_assisted_streak": int(grp["hmm_max_streak"].max()),
                "max_hmm_llr": round(float(grp["hmm_llr"].max()), 3),
                "flagged_toggle_games": int((grp["status"] == "flagged_toggle").sum()),
            }
        )
    player_df = pd.DataFrame(player_rows).sort_values("max_hmm_llr", ascending=False)

    return {
        "algorithm": "hmm_toggle_detector",
        "player_summary": player_df.where(player_df.notna(), None).to_dict(orient="records"),
        "game_scores": game_df.where(game_df.notna(), None).to_dict(orient="records"),
    }


@register_detector(
    "bayesian_sprt_ensemble",
    description="Full Bayesian SPRT ensemble fusing multi-depth surprise, HMM toggling, and meta-history checks.",
    default_enabled=True,
)
def run_bayesian_sprt_ensemble(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    cfg = DetectorConfig(
        sprt_flag_llr=float(params.get("sprt_flag_llr", 6.85)),
        sprt_review_llr=float(params.get("sprt_review_llr", 4.20)),
    )
    scored_df = score_games(features_df, eval_df=eval_df, config=cfg)
    player_df = aggregate_player_scores(scored_df, features_df=features_df, config=cfg)
    return {
        "algorithm": "bayesian_sprt_ensemble",
        "player_summary": player_df.where(player_df.notna(), None).to_dict(orient="records"),
        "game_scores": scored_df.where(scored_df.notna(), None).to_dict(orient="records"),
    }


@register_detector(
    "game_style_fingerprint",
    description="Personal style-profile deviation from prior games (accuracy, tactical precision, volatility, blunder and opening profile).",
    default_enabled=True,
)
def run_game_style_detector(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    empty = {"algorithm": "game_style_fingerprint", "player_summary": [], "game_scores": []}
    if eval_df is None or eval_df.empty:
        return empty

    max_games = int(params.get("max_games", 50))
    skip_opening_plies = int(params.get("skip_opening_plies", 8))
    games = game_inputs_from_eval_df(eval_df, skip_opening_plies=skip_opening_plies)

    by_player: dict[str, list[GameInput]] = {}
    for game in games:
        by_player.setdefault(game.player, []).append(game)

    rows: list[dict[str, Any]] = []
    for player_games in by_player.values():
        rows.extend(
            evaluate_style_deviation(
                player_games, max_games=max_games, skip_opening_plies=skip_opening_plies
            )
        )

    game_df = pd.DataFrame(rows)
    if game_df.empty:
        return empty
    game_df = game_df.sort_values(
        ["style_deviation", "game_id"], ascending=[False, True]
    ).reset_index(drop=True)

    player_rows = []
    for player, group in game_df.groupby("player", sort=False):
        ordered = group.assign(_date=group["date"].fillna("").astype(str)).sort_values(
            ["_date", "game_id"]
        )
        player_rows.append(
            {
                "player": player,
                "games": len(group),
                "mean_style_deviation": round(float(group["style_deviation"].mean()), 4),
                "max_style_deviation": round(float(group["style_deviation"].max()), 4),
                "latest_style_deviation": round(float(ordered.iloc[-1]["style_deviation"]), 4),
                "mean_profile_games": round(float(group["profile_games"].mean()), 2),
            }
        )
    player_df = pd.DataFrame(player_rows).sort_values(
        "max_style_deviation", ascending=False
    ).reset_index(drop=True)

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")

    return {
        "algorithm": "game_style_fingerprint",
        "player_summary": records(player_df),
        "game_scores": records(game_df),
    }
