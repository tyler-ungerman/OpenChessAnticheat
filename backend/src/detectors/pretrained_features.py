"""Shared features for controlled ChessFraud training and PGN inference.

Only fields available from BOTH the tournament and the user's PGN pipeline
are used. Keep feature definitions and missing-data semantics identical.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "top1_rate", "mean_capped_cpl", "median_cpl", "blunder_rate",
    "precise_rate", "median_time_sec", "time_cv", "quick_top1_rate",
    "contested_top1_rate", "shallow_deep_disagreement_rate",
    "deep_only_top1_rate", "longest_top1_streak_fraction",
    "late_top1_rate", "top1_think_time_ratio",
)
MIN_MOVES = 12


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def aggregate_moves(
    rows: pd.DataFrame,
    *,
    move_column: str, engine_column: str, cpl_column: str, time_column: str,
    shallow_column: str, eval_before_column: str, ply_column: str,
    evaluated_column: str | None = None,
) -> dict[str, float] | None:
    if rows.empty:
        return None
    frame = rows.sort_values(ply_column).copy()
    if evaluated_column:
        frame = frame[frame[evaluated_column].fillna(False).astype(bool)]
    cpl = _finite(frame[cpl_column])
    played = frame[move_column].fillna("").astype(str)
    best = frame[engine_column].fillna("").astype(str)
    ok = cpl.notna() & (played != "") & (best != "")
    if int(ok.sum()) < MIN_MOVES:
        return None
    cpl = cpl[ok].clip(lower=0)
    top = (played[ok].to_numpy() == best[ok].to_numpy()).astype(float)
    time = _finite(frame.loc[ok, time_column])
    valid_times = time[(time >= 0) & (time <= 7200)].dropna()
    matched = pd.Series(top, index=time.index)
    quick = (time.notna()) & (time >= 0) & (time <= 5)
    quick_top1 = float(matched.loc[quick].mean()) if int(quick.sum()) >= 3 else math.nan

    before = _finite(frame.loc[ok, eval_before_column])
    contested = before.abs() <= 200
    contested_top1 = float(matched.loc[contested].mean()) if int(contested.sum()) >= 5 else math.nan

    shallow = frame.loc[ok, shallow_column].fillna("").astype(str)
    both = shallow != ""
    different = both & (shallow != best[ok])
    disagree = float(different.sum() / both.sum()) if int(both.sum()) >= 5 else math.nan
    deep_only = float(matched.loc[different].mean()) if int(different.sum()) >= 3 else math.nan

    longest = 0
    running = 0
    for item in top:
        running = running + 1 if item else 0
        longest = max(longest, running)

    ply = _finite(frame.loc[ok, ply_column])
    late = ply >= 20
    late_top1 = float(matched.loc[late].mean()) if int(late.sum()) >= 5 else math.nan
    fast_median = time.loc[matched == 1].dropna()
    other_median = time.loc[matched == 0].dropna()
    ratio = (float(fast_median.median() / other_median.median())
             if len(fast_median) >= 3 and len(other_median) >= 3
             and other_median.median() > 0.01 else math.nan)

    return {
        "top1_rate": float(top.mean()),
        "mean_capped_cpl": float(cpl.clip(upper=300).mean()),
        "median_cpl": float(cpl.median()),
        "blunder_rate": float((cpl >= 200).mean()),
        "precise_rate": float((cpl <= 20).mean()),
        "median_time_sec": float(valid_times.median()) if len(valid_times) else math.nan,
        "time_cv": float(valid_times.std(ddof=0) / valid_times.mean())
                   if len(valid_times) > 2 and valid_times.mean() > 0.001 else math.nan,
        "quick_top1_rate": quick_top1,
        "contested_top1_rate": contested_top1,
        "shallow_deep_disagreement_rate": disagree,
        "deep_only_top1_rate": deep_only,
        "longest_top1_streak_fraction": float(longest / len(top)),
        "late_top1_rate": late_top1,
        "top1_think_time_ratio": ratio,
    }


def from_chessfraud(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "player_id", "is_cheating_player_game", "time_control",
                "is_used", "move_player", "move_stockfish_15", "move_stockfish_9",
                "centipawn_loss", "move_thinking_time", "eval_before", "half_move"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing ChessFraud fields: {sorted(missing)}")
    out = []
    for (player, game), group in frame.groupby(["player_id", "game_id"], sort=True):
        if set(group["is_cheating_player_game"].dropna().astype(int).unique()) - {0, 1}:
            raise ValueError("Unknown game labels")
        labels = group["is_cheating_player_game"].dropna().astype(int).unique()
        if len(labels) != 1:
            raise ValueError(f"Conflicting game label for {player}/{game}")
        if not (group["time_control"].astype(str) == "300+0").all():
            continue
        features = aggregate_moves(
            group, move_column="move_player", engine_column="move_stockfish_15",
            cpl_column="centipawn_loss", time_column="move_thinking_time",
            shallow_column="move_stockfish_9", eval_before_column="eval_before",
            ply_column="half_move", evaluated_column="is_used",
        )
        if features is not None:
            out.append({"player_id": str(player), "game_id": str(game),
                        "label": int(labels[0]), **features})
    return pd.DataFrame(out)


def from_local_game(frame: pd.DataFrame) -> dict[str, float] | None:
    return aggregate_moves(
        frame, move_column="move_uci", engine_column="best_move",
        cpl_column="centipawn_loss", time_column="time_spent",
        shallow_column="shallow_best_move", eval_before_column="eval_before",
        ply_column="ply",
    )
