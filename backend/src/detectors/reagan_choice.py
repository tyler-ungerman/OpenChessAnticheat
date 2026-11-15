"""Regan-style move-choice sensitivity fitted from centipawn loss."""

from __future__ import annotations

import numpy as np


def fit_game_sensitivity_s(move_cpls: np.ndarray, c_fixed: float = 0.85) -> float | None:
    losses = move_cpls[np.isfinite(move_cpls) & (move_cpls >= 0)]
    if len(losses) < 8:
        return None
    # Smooth by 0.03 pawns (3 cp) so 0-CPL recaptures don't collapse s to 0
    pawns = np.clip((losses + 3.0) / 100.0, 0.03, 5.0)
    # Closed-form MLE for Weibull scale parameter s given shape c_fixed:
    s_mle = float(np.mean(pawns ** c_fixed) ** (1.0 / c_fixed))
    return round(float(np.clip(s_mle, 0.08, 2.50)), 4)
