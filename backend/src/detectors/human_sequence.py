"""Rank sustained engine-like choices against a compact real-human move prior.

Pure Python + numpy + chess. The model is a behaviour reference, not an
estimated prevalence of assisted play. We only score trained-on blitz controls;
other time controls retain the usual PGN screening.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import chess
import numpy as np
import pandas as pd

from .human_prior import FEATURE_NAMES, frame_features, predict_human_top1, human_surprise_scores

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "human_prior_v1.json"
SUPPORTED_CONTROLS = {"180", "180+0", "180+2", "300", "300+0", "300+3"}
WINDOW_MOVES = 5
MIN_GAME_MOVES = 12
MIN_DISPLAY_QUIET_DECISIONS = 3


def load_human_prior(path: Path | str = MODEL_PATH) -> dict | None:
    file = Path(path)
    if not file.exists():
        return None
    model = json.loads(file.read_text(encoding="utf-8"))
    if model.get("schema") != "openchess-human-choice-v1":
        raise ValueError("Incompatible human-choice model")
    if model.get("model_features") != FEATURE_NAMES:
        raise ValueError("Human-choice feature schema mismatch")
    n = len(FEATURE_NAMES)
    for name in ("impute", "mean", "scale", "coef"):
        values = model.get(name)
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"Bad model parameter: {name}")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
            raise ValueError(f"Non-finite model parameter: {name}")
    if any(x <= 0 for x in model["scale"]):
        raise ValueError("Nonpositive model scale")
    if not math.isfinite(float(model.get("intercept", math.nan))):
        raise ValueError("Bad model intercept")
    return model


def _move_context(fen: object, played_uci: object) -> str:
    """Describe the observed decision without changing the trained score.

    A capture or check is tactical, not necessarily forced. Only responses
    while in check or positions with few legal options are called constrained.
    """
    try:
        board = chess.Board(str(fen))
        move = chess.Move.from_uci(str(played_uci))
        if move not in board.legal_moves:
            return "unknown"
        if board.is_check():
            return "responding to check"
        if board.legal_moves.count() < 8:
            return "few legal choices"
        if board.is_castling(move):
            return "castling"
        if board.is_capture(move):
            return "capture"
        if board.gives_check(move):
            return "check"
        if move.promotion:
            return "promotion"
        return "quiet decision"
    except (TypeError, ValueError, AssertionError):
        return "unknown"


def scan_human_sequences(eval_df: pd.DataFrame, model: dict | None,
                         *, player: str | None = None, top_games: int = 5) -> dict:
    output = {
        "model": model.get("schema") if model is not None else None,
        "eligible_games": 0, "ineligible_games": 0,
        "scored_moves": 0, "ranked_games": [],
        "suppressed_low_context_windows": 0,
        "description": ("Observed Stockfish choices compared with a lightweight "
                        "position-adjusted human baseline. Scores are NOT probabilities of cheating."),
    }
    if model is None or eval_df.empty or top_games <= 0:
        return output
    frame = eval_df
    if player:
        frame = frame[frame["player"].astype(str).str.casefold() == player.casefold()]
    elo_lookup = {}
    for gid, grp in frame.groupby("game_id", sort=False):
        elo_lookup[gid] = {
            str(username).casefold(): pd.to_numeric(part["elo"], errors="coerce").median()
            for username, part in grp.groupby("player", sort=False)
        }
    for (game_id, username), game in frame.groupby(["game_id", "player"], sort=False):
        game = game.sort_values("ply")
        tc = str(game["time_control"].iloc[0]).strip()
        if tc not in SUPPORTED_CONTROLS:
            output["ineligible_games"] += 1
            continue
        plies = pd.to_numeric(game["ply"], errors="coerce")
        valid_cpl = pd.to_numeric(game["centipawn_loss"], errors="coerce").notna()
        valid = game.loc[valid_cpl & (plies >= 21) & game["best_move"].notna()].copy()
        if len(valid) < MIN_GAME_MOVES:
            output["ineligible_games"] += 1
            continue

        players = elo_lookup.get(game_id, {})
        other = [elo for name, elo in players.items() if name != str(username).casefold()]
        valid["opponent_elo"] = other[0] if other else math.nan
        features = frame_features(valid)
        human_p = predict_human_top1(features, model)
        is_top = valid["is_top1"].fillna(False).astype(int).to_numpy()
        adjusted = human_surprise_scores(is_top, human_p)
        if len(adjusted) < WINDOW_MOVES:
            continue

        # The maximum average likelihood ratio of contiguous player decisions
        # is the same fixed statistic evaluated against the real ChessFraud holdouts.
        means = np.convolve(adjusted, np.ones(WINDOW_MOVES)/WINDOW_MOVES, mode="valid")
        start = int(np.argmax(means))
        block = valid.iloc[start:start + WINDOW_MOVES]
        p_block = human_p[start:start + WINDOW_MOVES]
        top_block = is_top[start:start + WINDOW_MOVES]
        output["eligible_games"] += 1
        output["scored_moves"] += len(valid)

        displayed_moves = []
        quiet_expected = 0.0
        quiet_engine_matches = 0
        quiet_count = 0
        for i, (_, row) in enumerate(block.iterrows()):
            move_number = int(row.get("move_number") or (int(row["ply"])+1)//2)
            played = str(row.get("move_san") or row.get("move_uci") or "?")
            context = _move_context(row.get("fen_before"), row.get("move_uci"))
            is_quiet = context == "quiet decision"
            if is_quiet:
                quiet_count += 1
                quiet_expected += float(p_block[i])
                quiet_engine_matches += int(top_block[i])
            displayed_moves.append({
                "move_number": move_number, "san": played, "uci": row.get("move_uci"),
                "fen_before": row.get("fen_before"),
                "is_engine_top1": bool(top_block[i]),
                "move_context": context,
                "is_informative_quiet_move": is_quiet,
                "human_top1_probability": round(float(p_block[i]), 3),
                "time_spent_seconds": (float(row["time_spent"])
                                       if pd.notna(row.get("time_spent")) else None),
            })
        if quiet_count < MIN_DISPLAY_QUIET_DECISIONS:
            # Detection score is not changed; this only prevents the CLI
            # from suggesting a sequence lacking interpretable move evidence.
            output["suppressed_low_context_windows"] += 1
            continue
        output["ranked_games"].append({
            "game_id": game_id, "player": username,
            "time_control": tc, "date": game["date"].iloc[0] if "date" in game else None,
            "sequence_score": round(float(means[start]), 3),
            "engine_first_count": int(top_block.sum()),
            "human_expected_engine_first": round(float(p_block.sum()), 2),
            "quiet_decision_count": quiet_count,
            "quiet_engine_first_count": quiet_engine_matches,
            "quiet_expected_engine_first": round(quiet_expected, 2),
            "tactical_or_constrained_count": WINDOW_MOVES - quiet_count,
            "windows_searched": len(adjusted) - WINDOW_MOVES + 1,
            "start_move": displayed_moves[0]["move_number"],
            "end_move": displayed_moves[-1]["move_number"],
            "moves": displayed_moves,
        })
    output["ranked_games"].sort(
        key=lambda game: (-game["sequence_score"], str(game["game_id"]))
    )
    output["ranked_games"] = output["ranked_games"][:top_games]
    return output
