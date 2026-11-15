"""Two-state Human/Assisted HMM with Viterbi decoding over ply-level signals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def fit_game_hmm_toggle(side_plies: pd.DataFrame) -> dict[str, float]:
    """
    2-State Hidden Markov Model (State 0 = Unassisted Human, State 1 = Engine Assisted)
    using forward-backward & Viterbi decoding over ply-level (CPL, EvalGap, TimeSpent)
    to detect mid-game engine 'toggling' streaks that get diluted in full-game averages.
    """
    valid = side_plies[
        side_plies["centipawn_loss"].notna() & (side_plies["legal_moves_count"] > 1)
    ].sort_values("ply")

    if len(valid) < 8:
        return {"hmm_assisted_frac": 0.0, "hmm_max_streak": 0, "hmm_llr": 0.0}

    cpls = pd.to_numeric(valid["centipawn_loss"], errors="coerce").fillna(0.0).to_numpy()
    gaps = pd.to_numeric(valid.get("eval_gap", 30.0), errors="coerce").fillna(30.0).to_numpy()
    is_top1 = valid["is_top1"].fillna(False).to_numpy(dtype=bool)
    times = pd.to_numeric(valid.get("time_spent"), errors="coerce").to_numpy()

    # Transition matrix in log space: regimes are sticky (p_stay = 0.88)
    log_trans = np.log(np.array([[0.90, 0.10], [0.12, 0.88]]))
    log_init = np.log(np.array([0.85, 0.15]))

    n = len(cpls)
    log_emit = np.zeros((n, 2), dtype=float)

    for t in range(n):
        loss_pawns = min(4.0, max(0.0, cpls[t] / 100.0))
        sharp = gaps[t] >= 75.0

        # Emission likelihood under Human state (s_human = 0.55) vs Engine state (s_eng = 0.09)
        ll_h = -((loss_pawns / 0.55) ** 0.85) - math.log(0.55)
        ll_e = -((loss_pawns / 0.09) ** 0.85) - math.log(0.09)

        if sharp and is_top1[t]:
            ll_e += 0.45
            ll_h -= 0.25
        elif loss_pawns > 0.80:
            ll_e -= 2.50  # Engines virtually never drop >80 cp

        # Robotic time window bonus (3.0s - 7.5s on a sharp position)
        if math.isfinite(times[t]) and sharp:
            if 2.5 <= times[t] <= 7.5 and is_top1[t]:
                ll_e += 0.30
            elif times[t] > 18.0:
                ll_h += 0.25

        log_emit[t, 0] = ll_h
        log_emit[t, 1] = ll_e

    # Viterbi decoding for most likely state sequence
    dp = np.zeros((n, 2), dtype=float)
    back = np.zeros((n, 2), dtype=int)
    dp[0] = log_init + log_emit[0]

    for t in range(1, n):
        for s in (0, 1):
            scores = dp[t - 1] + log_trans[:, s]
            best_prev = int(np.argmax(scores))
            dp[t, s] = scores[best_prev] + log_emit[t, s]
            back[t, s] = best_prev

    states = np.zeros(n, dtype=int)
    states[-1] = int(np.argmax(dp[-1]))
    for t in range(n - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]

    # Max contiguous streak of state 1 (Assisted)
    max_streak = 0
    cur_streak = 0
    for s in states:
        if s == 1:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0

    assisted_frac = float(np.mean(states == 1))
    # Log-likelihood ratio of best HMM path vs pure-human path
    pure_human_ll = float(log_init[0] + np.sum(log_emit[:, 0]) + (n - 1) * log_trans[0, 0])
    best_path_ll = float(np.max(dp[-1]))
    hmm_llr = max(0.0, best_path_ll - pure_human_ll)

    return {
        "hmm_assisted_frac": round(assisted_frac, 4),
        "hmm_max_streak": int(max_streak),
        "hmm_llr": round(hmm_llr, 3),
    }
