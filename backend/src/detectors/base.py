"""Detector registry primitives and shared configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class DetectorConfig:
    min_history_games: int = 8
    # Signal weights for composite suspicion score (sum to 1.0)
    weight_accuracy: float = 0.26
    weight_critical: float = 0.22
    weight_choice_model: float = 0.16
    weight_surprise: float = 0.14
    weight_timing: float = 0.10
    weight_phase_shift: float = 0.06
    weight_contextual_toggle: float = 0.06
    # Thresholds for flagging games
    flag_z_threshold: float = 2.5
    review_z_threshold: float = 1.8
    # SPRT decision bounds (Wald thresholds: alpha=0.001, beta=0.05 -> A ≈ 6.85)
    sprt_flag_llr: float = 6.85
    sprt_review_llr: float = 4.20


DetectorFn = Callable[[pd.DataFrame, pd.DataFrame | None, dict[str, Any]], dict[str, Any]]


DETECTOR_REGISTRY: dict[str, dict[str, Any]] = {}


def register_detector(name: str, description: str = "", default_enabled: bool = True):
    """
    Decorator to register modular detection algorithms so the API automatically
    exposes them in `/detectors` and runs selected algorithms in `/analyze/*`.
    """
    def decorator(fn: DetectorFn) -> DetectorFn:
        DETECTOR_REGISTRY[name] = {
            "name": name,
            "description": description or (fn.__doc__ or "").strip(),
            "default_enabled": default_enabled,
            "fn": fn,
        }
        return fn
    return decorator
