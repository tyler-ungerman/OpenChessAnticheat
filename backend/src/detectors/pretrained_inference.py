"""Optional, dependency-light inference for pretrained ChessFraud game classifier.

The model only applies to matching 300+0 blitz games and outputs an
experimental assistance score, NOT a real-world cheating probability.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .pretrained_features import FEATURE_NAMES, from_local_game

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "chessfraud_v1.json"
SCHEMA = "openchess-chessfraud-game-logistic-v1"


def load_pretrained_model(path: Path | str = DEFAULT_MODEL_PATH) -> dict | None:
    file = Path(path)
    if not file.exists():
        return None
    model = json.loads(file.read_text(encoding="utf-8"))
    if model.get("schema") != SCHEMA or model.get("model_features") != list(FEATURE_NAMES):
        raise ValueError("Pretrained model has incompatible schema or feature set")
    for name in ("impute_medians", "scale_means", "scale_std", "coefficients"):
        arr = model.get(name)
        if not isinstance(arr, list) or len(arr) != len(FEATURE_NAMES):
            raise ValueError(f"Incompatible pretrained parameter {name}")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in arr):
            raise ValueError(f"Invalid pretrained parameter {name}")
    if any(float(x) <= 0 for x in model["scale_std"]):
        raise ValueError("Invalid feature scales")
    if not math.isfinite(float(model.get("intercept", math.nan))):
        raise ValueError("Invalid model intercept")
    if model.get("time_control") != "300+0":
        raise ValueError("Unsupported model time control")
    return model


def predict_feature_score(features: dict[str, float], model: dict) -> float:
    """Exact JSON replay of sklearn's impute->standardize->logistic pipeline."""
    x = []
    for index, name in enumerate(FEATURE_NAMES):
        val = float(features[name])
        if not math.isfinite(val):
            val = float(model["impute_medians"][index])
        x.append((val - float(model["scale_means"][index])) / float(model["scale_std"][index]))
    logit = float(model["intercept"]) + sum(
        float(weight) * value for weight, value in zip(model["coefficients"], x)
    )
    # Numerical-stable sigmoid; 0–100 relative output, not population-calibrated.
    logit = min(30.0, max(-30.0, logit))
    return float(100.0 / (1.0 + math.exp(-logit)))


def score_pgn_games(eval_df: pd.DataFrame, model: dict | None, *,
                    player: str | None = None) -> list[dict]:
    if model is None or eval_df.empty:
        return []
    frame = eval_df
    if player:
        frame = frame[frame["player"].astype(str).str.casefold() == player.casefold()]
    out = []
    for (game_id, username), game in frame.groupby(["game_id", "player"], sort=True):
        # Inference cannot transfer unvalidated 5-minute training to bullet/rapid.
        tc = game["time_control"].astype(str).iloc[0]
        if tc != "300+0":
            continue
        values = from_local_game(game)
        if values is None:
            continue
        out.append({
            "game_id": game_id,
            "player": username,
            "experimental_assistance_score": round(predict_feature_score(values, model), 1),
            "model": model["schema"],
            "interpretation": "Benchmark-relative experimental score; not probability of cheating",
        })
    return sorted(out, key=lambda x: (-x["experimental_assistance_score"], str(x["game_id"])))
