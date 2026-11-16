"""Shared, cheap definition of informative chess decisions and five-move windows.

The five positions are consecutive *player* decisions. Captures, checks,
responses while in check, and near-forced positions stay in the window but
cannot contribute evidence. This prevents forcing tactical combinations from
appearing as five independently surprising decisions.
"""
from __future__ import annotations

import math

import chess
import numpy as np

WINDOW_MOVES = 5
MIN_QUIET_DECISIONS = 3


def informative_choice(fen: object, move_uci: object) -> bool:
    try:
        board = chess.Board(str(fen))
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            return False
        if board.is_check() or board.legal_moves.count() < 8:
            return False
        if board.is_capture(move) or board.gives_check(move) or move.promotion:
            return False
        return True
    except (ValueError, TypeError, AssertionError):
        return False


def peak_five_choice_window(
    surprises: np.ndarray, top1: np.ndarray, human_probs: np.ndarray,
    informative: np.ndarray, *, min_quiet: int = MIN_QUIET_DECISIONS,
    forcing_weight: float = 0.0
) -> dict | None:
    surprises = np.asarray(surprises, dtype=float)
    top1 = np.asarray(top1, dtype=int)
    human_probs = np.asarray(human_probs, dtype=float)
    informative = np.asarray(informative, dtype=bool)
    if not 0 <= forcing_weight <= 1:
        raise ValueError("forcing_weight must be between 0 and 1")
    n = len(surprises)
    if n < WINDOW_MOVES or any(len(x) != n for x in (top1, human_probs, informative)):
        return None
    good = informative & np.isfinite(surprises) & np.isfinite(human_probs)
    candidate_indices = []
    for start in range(n - WINDOW_MOVES + 1):
        indices = slice(start, start + WINDOW_MOVES)
        count = int(good[indices].sum())
        if count < min_quiet:
            continue
        # Fixed denominator of 5 prevents 3/5 quiet windows from getting a
        # disproportionate score; excluded forcing moves add exactly zero.
        weights = np.where(good[indices], 1.0, forcing_weight)
        finite = np.isfinite(surprises[indices])
        score = float(np.where(finite, weights * surprises[indices], 0.0).sum() / WINDOW_MOVES)
        candidate_indices.append((score, start, count))
    if not candidate_indices:
        return None
    score, start, count = max(candidate_indices, key=lambda x: (x[0], -x[1]))
    window = slice(start, start + WINDOW_MOVES)
    mask = good[window]
    return {
        "start": start,
        "score": score,
        "informative_count": count,
        "top1_informative": int(top1[window][mask].sum()),
        "expected_top1_informative": float(human_probs[window][mask].sum()),
        "informative_mask": mask.tolist(),
        "windows_examined": len(candidate_indices),
    }
