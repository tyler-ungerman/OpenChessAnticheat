"""Small, portable estimate of human engine-first move selection.

Provides identical feature extraction for published ChessFraud-Synth rows and
our local evaluated PGN rows. No chess neural network, network access, or
scikit-learn needed to infer after a JSON model passes held-out evaluation.
"""
from __future__ import annotations

import math

import chess
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "elo_1000", "opp_elo_1000", "ply_100",
    "log_legal_moves", "abs_eval_1000", "clock_base_minutes",
    "deep_shallow_agree", "best_is_capture", "best_gives_check", "piece_fraction",
]


def _num(value, default=math.nan):
    try:
        n = float(value)
        return n if math.isfinite(n) else default
    except (TypeError, ValueError):
        return default


def _clock_base(value) -> float:
    value = str(value or "").strip()
    if "/" in value:
        return math.nan
    try:
        return max(0, min(60, float(value.split("+")[0])/60))
    except ValueError:
        return math.nan


def _best_properties(fen, uci):
    try:
        board = chess.Board(str(fen))
        move = chess.Move.from_uci(str(uci))
        if move not in board.legal_moves:
            return math.nan, math.nan, math.nan, math.nan
        return (float(board.legal_moves.count()),
                float(board.is_capture(move)),
                float(board.gives_check(move)),
                float(len(board.piece_map())) / 32.0)
    except (TypeError, ValueError, AssertionError):
        return math.nan, math.nan, math.nan, math.nan


def frame_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return position-only features, never use the observed move as input."""
    if data.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)
    out = []
    for row in data.to_dict(orient="records"):
        fen = row.get("position_before", row.get("fen_before"))
        deep = row.get("move_stockfish_15", row.get("best_move"))
        shallow = row.get("move_stockfish_9", row.get("shallow_best_move"))
        legal, capture, check, pieces = _best_properties(fen, deep)
        elo = _num(row.get("player_elo", row.get("elo")))
        opponent_elo = _num(row.get("opponent_elo"))
        ply = _num(row.get("half_move", row.get("ply")))
        evaluation = _num(row.get("eval_before"))
        out.append({
            "elo_1000": elo / 1000,
            "opp_elo_1000": opponent_elo / 1000,
            "ply_100": ply / 100,
            "log_legal_moves": math.log1p(legal) if math.isfinite(legal) else math.nan,
            "abs_eval_1000": min(1.0, abs(evaluation)/1000) if math.isfinite(evaluation) else math.nan,
            "clock_base_minutes": _clock_base(row.get("time_control")),
            "deep_shallow_agree": float(str(deep) == str(shallow))
                                  if pd.notna(deep) and pd.notna(shallow) else math.nan,
            "best_is_capture": capture,
            "best_gives_check": check,
            "piece_fraction": pieces,
        })
    return pd.DataFrame(out, index=data.index, columns=FEATURE_NAMES)


def human_surprise_scores(top1: np.ndarray, human_p: np.ndarray) -> np.ndarray:
    """Log likelihood of an engine-biased chooser vs human-choice baseline.

    q=0.85 is a fixed counterfactual assumption, not estimated prevalence.
    Non-best moves can still be assisted (e.g. 2nd-best engine suggestions).
    """
    p = np.clip(np.asarray(human_p, dtype=float), 0.01, 0.99)
    is_top = np.asarray(top1, dtype=int)
    return np.where(is_top == 1, np.log(0.85 / p), np.log(0.15 / (1-p)))


def predict_human_top1(features: pd.DataFrame, model: dict) -> np.ndarray:
    if model.get("schema") != "openchess-human-choice-v1" or model.get("model_features") != FEATURE_NAMES:
        raise ValueError("Incompatible human choice model")
    x = features[FEATURE_NAMES].to_numpy(dtype=float)
    missing = ~np.isfinite(x)
    x[missing] = np.broadcast_to(np.asarray(model["impute"]), x.shape)[missing]
    x = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = x @ np.asarray(model["coef"]) + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
