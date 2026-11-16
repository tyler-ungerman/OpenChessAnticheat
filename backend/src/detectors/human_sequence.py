"""Rank sustained engine-like choices against a compact real-human move prior.

Pure Python + numpy + chess. The model is a behaviour reference, not an
estimated prevalence of assisted play. We only score trained-on blitz controls;
other time controls retain the usual PGN screening.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .human_prior import FEATURE_NAMES, frame_features, predict_human_top1, human_surprise_scores
from .sequence_policy import informative_choice, peak_five_choice_window, WINDOW_MOVES

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "human_prior_v1.json"
SUPPORTED_CONTROLS = {"180", "180+0", "180+2", "300", "300+0", "300+3"}
WINDOW_MOVES = 5
MIN_GAME_MOVES = 12


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


def scan_human_sequences(eval_df: pd.DataFrame, model: dict | None,
                         *, player: str | None = None, top_games: int = 5) -> dict:
    output = {
        "model": model.get("schema") if model is not None else None,
        "eligible_games": 0, "ineligible_games": 0,
        "scored_moves": 0, "ranked_games": [],
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

        quiet = np.asarray([
            informative_choice(fen, uci)
            for fen, uci in zip(valid["fen_before"], valid["move_uci"])
        ], dtype=bool)
        selected = peak_five_choice_window(adjusted, is_top, human_p, quiet)
        output["eligible_games"] += 1
        output["scored_moves"] += len(valid)
        if selected is None:
            continue
        start = selected["start"]
        block = valid.iloc[start:start + WINDOW_MOVES]
        p_block = human_p[start:start + WINDOW_MOVES]
        top_block = is_top[start:start + WINDOW_MOVES]

        displayed_moves = []
        for i, (_, row) in enumerate(block.iterrows()):
            move_number = int(row.get("move_number") or (int(row["ply"])+1)//2)
            played = str(row.get("move_san") or row.get("move_uci") or "?")
            displayed_moves.append({
                "move_number": move_number, "san": played, "uci": row.get("move_uci"),
                "fen_before": row.get("fen_before"),
                "is_engine_top1": bool(top_block[i]),
                "is_informative_quiet_move": bool(selected["informative_mask"][i]),
                "human_top1_probability": round(float(p_block[i]), 3),
                "time_spent_seconds": (float(row["time_spent"])
                                       if pd.notna(row.get("time_spent")) else None),
            })
        output["ranked_games"].append({
            "game_id": game_id, "player": username,
            "time_control": tc, "date": game["date"].iloc[0] if "date" in game else None,
            "sequence_score": round(float(selected["score"]), 3),
            "engine_first_count": selected["top1_informative"],
            "informative_moves": selected["informative_count"],
            "excluded_forcing_moves": WINDOW_MOVES - selected["informative_count"],
            "human_expected_engine_first": round(selected["expected_top1_informative"], 2),
            "windows_examined": selected["windows_examined"],
            "start_move": displayed_moves[0]["move_number"],
            "end_move": displayed_moves[-1]["move_number"],
            "moves": displayed_moves,
        })
    output["ranked_games"].sort(
        key=lambda game: (-game["sequence_score"], str(game["game_id"]))
    )
    output["ranked_games"] = output["ranked_games"][:top_games]
    return output
