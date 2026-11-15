"""Player style fingerprints: per-game metrics and deviation from game history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SKIP_OPENING_PLIES = 8
DEFAULT_MAX_PROFILE_GAMES = 50
OPENING_MOVE_LIMIT = 10  # mirrors features._assign_phase


@dataclass(frozen=True)
class MoveInput:
    ply: int
    move_number: int
    player_move: str | None
    engine_best: str | None
    cp_loss: float | None
    eval_swing_cp: float | None
    maia_probability: float | None = None
    is_opening_book: bool = False


@dataclass(frozen=True)
class GameInput:
    game_id: int | str
    color: str
    player: str
    date: str | None
    time_control: str | None
    elo: float | None
    moves: list[MoveInput]


@dataclass(frozen=True)
class StyleProfile:
    means: dict[str, float]
    stds: dict[str, float]
    games_count: int


def split_analysis_window(
    moves: list[MoveInput],
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> tuple[list[MoveInput], int]:
    """Return evaluated plies after the opening, plus the analyzed count."""
    window = [
        move
        for move in moves
        if move.ply > skip_opening_plies
        and move.cp_loss is not None
        and math.isfinite(move.cp_loss)
    ]
    return window, len(window)


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _game_style_metrics(
    game: GameInput,
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> dict[str, float] | None:
    window, analyzed = split_analysis_window(game.moves, skip_opening_plies)
    if analyzed == 0:
        return None

    engine_match = _safe_mean([1.0 if m.player_move == m.engine_best else 0.0 for m in window])
    cp_losses = [float(m.cp_loss) for m in window]
    blunder_rate = _safe_mean([1.0 if m.cp_loss >= 120 else 0.0 for m in window])
    critical = [m for m in window if m.eval_swing_cp is not None and m.eval_swing_cp >= 100]
    tactical_accuracy = (
        _safe_mean([1.0 if m.player_move == m.engine_best else 0.0 for m in critical])
        if critical
        else 0.0
    )
    eval_volatility = (
        float(np.std(np.array(cp_losses, dtype=float), ddof=1)) if len(cp_losses) > 1 else 0.0
    )
    maia_probs = [m.maia_probability for m in window if m.maia_probability is not None]
    maia_likelihood = _safe_mean([float(p) for p in maia_probs]) if maia_probs else 0.0
    opening_ratio = _safe_mean([1.0 if m.is_opening_book else 0.0 for m in window])
    opening_diversity = float(max(0.0, min(1.0, 1.0 - opening_ratio)))

    return {
        "avg_centipawn_loss": _safe_mean(cp_losses),
        "engine_match_rate": engine_match,
        "tactical_move_ratio": tactical_accuracy,
        "evaluation_volatility": eval_volatility,
        "maia_likelihood": maia_likelihood,
        "blunder_rate": blunder_rate,
        "opening_diversity": opening_diversity,
    }


def build_style_profile(
    games: list[GameInput],
    max_games: int = DEFAULT_MAX_PROFILE_GAMES,
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> StyleProfile | None:
    if not games:
        return None
    sample = games[-max_games:] if max_games > 0 else games
    metrics: dict[str, list[float]] = {}
    count = 0
    for game in sample:
        values = _game_style_metrics(game, skip_opening_plies)
        if values is None:
            continue
        count += 1
        for key, value in values.items():
            metrics.setdefault(key, []).append(float(value))

    if count == 0:
        return None

    means = {key: float(mean(values)) for key, values in metrics.items()}
    stds = {
        key: float(np.std(np.array(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0
        for key, values in metrics.items()
    }
    return StyleProfile(means=means, stds=stds, games_count=count)


def style_deviation_score(
    games: list[GameInput],
    max_games: int = DEFAULT_MAX_PROFILE_GAMES,
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> tuple[float, int]:
    """RMS Z-score of the latest game against the player's preceding games."""
    if len(games) < 2:
        return 0.0, 0
    profile = build_style_profile(games[:-1], max_games=max_games, skip_opening_plies=skip_opening_plies)
    if profile is None:
        return 0.0, 0
    current_metrics = _game_style_metrics(games[-1], skip_opening_plies)
    if current_metrics is None:
        return 0.0, profile.games_count

    zs: list[float] = []
    for key, mean_value in profile.means.items():
        std_value = profile.stds.get(key, 0.0)
        if std_value <= 0:
            continue
        current_value = current_metrics.get(key, mean_value)
        zs.append((current_value - mean_value) / std_value)
    if not zs:
        return 0.0, profile.games_count
    score = float(np.sqrt(np.mean(np.square(np.array(zs, dtype=float)))))
    return score, profile.games_count


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return str(value)


def game_inputs_from_eval_df(
    eval_df: pd.DataFrame,
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> list[GameInput]:
    """Adapt ply-level pipeline rows into ``GameInput`` objects."""
    if eval_df.empty:
        return []

    has_maia = "maia_probability" in eval_df.columns
    has_book = "is_opening_book" in eval_df.columns
    games: list[GameInput] = []

    for (game_id, color), group in eval_df.groupby(["game_id", "color"], sort=False):
        group = group.sort_values("ply")
        first = group.iloc[0]
        moves: list[MoveInput] = []
        for row in group.itertuples(index=False):
            move_number = getattr(row, "move_number", None)
            if move_number is None or pd.isna(move_number):
                move_number = (int(row.ply) + 1) // 2
            maia_value = getattr(row, "maia_probability", None) if has_maia else None
            book_value = getattr(row, "is_opening_book", None) if has_book else int(move_number) <= OPENING_MOVE_LIMIT
            moves.append(
                MoveInput(
                    ply=int(row.ply),
                    move_number=int(move_number),
                    player_move=_text(getattr(row, "move_uci", None)),
                    engine_best=_text(getattr(row, "best_move", None)),
                    cp_loss=_number(getattr(row, "centipawn_loss", None)),
                    eval_swing_cp=_number(getattr(row, "eval_gap", None)),
                    maia_probability=_number(maia_value),
                    is_opening_book=bool(book_value),
                )
            )
        games.append(
            GameInput(
                game_id=game_id,
                color=str(color),
                player=str(first.get("player") or ""),
                date=_text(first.get("date")),
                time_control=_text(first.get("time_control")),
                elo=_number(first.get("elo")),
                moves=moves,
            )
        )
    return games


def evaluate_style_deviation(
    games: list[GameInput],
    max_games: int = DEFAULT_MAX_PROFILE_GAMES,
    skip_opening_plies: int = DEFAULT_SKIP_OPENING_PLIES,
) -> list[dict[str, Any]]:
    """Score every game against the player's preceding games, in date order."""
    ordered = sorted(games, key=lambda game: (str(game.date or ""), str(game.game_id)))
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(ordered):
        metrics = _game_style_metrics(game, skip_opening_plies)
        if metrics is None:
            continue
        score, profile_games = style_deviation_score(
            ordered[: index + 1], max_games=max_games, skip_opening_plies=skip_opening_plies
        )
        rows.append(
            {
                "game_id": game.game_id,
                "date": game.date,
                "time_control": game.time_control,
                "player": game.player,
                "color": game.color,
                "elo": game.elo,
                "style_deviation": round(score, 4),
                "profile_games": profile_games,
                **{key: round(value, 4) for key, value in metrics.items()},
            }
        )
    return rows
