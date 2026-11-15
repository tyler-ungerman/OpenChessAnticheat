"""Shared features for controlled ChessFraud training and live PGN inference.

Only fields available in BOTH public ChessFraud and local Stockfish evaluations
are used. Inference requires 5-minute blitz; cross-engine search-depth mismatch
remains a limitation of the experimental model.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "top1_rate", "mean_capped_cpl", "median_cpl", "blunder_rate",
    "precise_rate", "median_time_sec", "time_cv", "quick_top1_rate",
)
MIN_MOVES = 12


def _finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def aggregate_moves(
    rows: pd.DataFrame,
    *,
    move_column: str, engine_column: str, cpl_column: str, time_column: str,
    evaluated_column: str | None = None,
) -> dict[str, float] | None:
    """Game-level features; unknown/missing time stays NaN for imputation."""
    if rows.empty:
        return None
    frame = rows.copy()
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
    }


def from_chessfraud(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"game_id", "player_id", "is_cheating_player_game", "time_control",
                "is_used", "move_player", "move_stockfish_15",
                "centipawn_loss", "move_thinking_time"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing ChessFraud fields: {sorted(missing)}")
    # Only one explicitly documented time control. No player_id as a feature.
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
            evaluated_column="is_used",
        )
        if features is not None:
            out.append({"player_id": str(player), "game_id": str(game),
                        "label": int(labels[0]), **features})
    return pd.DataFrame(out)


def from_local_game(frame: pd.DataFrame) -> dict[str, float] | None:
    """Same feature definitions as training: eval move, Stockfish best, CPL and time."""
    return aggregate_moves(
        frame, move_column="move_uci", engine_column="best_move",
        cpl_column="centipawn_loss", time_column="time_spent",
    )
