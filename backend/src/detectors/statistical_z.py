"""Dual-anchored modified Z-scores, CUSUM/SPRT aggregation, and game scoring."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .base import DetectorConfig, MIN_SIGMA
from .calibration import (
    _expected_acpl_for_elo,
    _expected_top1_for_elo,
    _lookup_cohort_z,
    cohort_time_controls,
    load_calibration,
)
from .hmm_toggle import fit_game_hmm_toggle
from .reagan_choice import fit_game_sensitivity_s


def _robust_z(value: float | None, series: pd.Series, higher_is_suspicious: bool = True) -> float | None:
    """
    Compute a Modified Z-Score using Median and MAD (Median Absolute Deviation),
    falling back to standard mean/std if MAD is zero.
    """
    if value is None or not math.isfinite(float(value)):
        return None
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 4:
        return None

    med = float(clean.median())
    mad = float((clean - med).abs().median())
    # A degenerate or highly concentrated reference is not evidence of infinite surprise.
    # The floors are conservative engineering safeguards, NOT empirical calibration.
    scale_floor = MIN_SIGMA.get(str(series.name), 0.05)
    if mad > 1e-6:
        z = (float(value) - med) / max(1.4826 * mad, scale_floor)
    else:
        z = (float(value) - med) / max(float(clean.std()), scale_floor)
    oriented = z if higher_is_suspicious else -z
    return float(np.clip(oriented, -6.0, 6.0))


def _max_valid(*values: float | None) -> float | None:
    """Return the maximum finite value among candidates, or None if all are None/NaN."""
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return max(valid) if valid else None


def _combine_z_scores(components: list[tuple[float | None, float]]) -> tuple[float, int]:
    """Weighted average of dependent signals; NOT a standard-normal Z statistic."""
    weighted_sum = 0.0
    active_weight = 0.0
    active_count = 0

    for z_val, weight in components:
        if z_val is not None and math.isfinite(z_val):
            clamped = max(-2.0, min(5.0, z_val))
            weighted_sum += clamped * weight
            active_weight += weight
            active_count += 1

    if active_weight <= 0.0:
        return 0.0, 0

    # Signal families overlap; Stouffer's independence denominator is invalid here.
    composite_z = weighted_sum / active_weight
    return round(composite_z, 3), active_count


def _game_sprt_llr(composite_z: float, z_H0: float = 0.0, z_H1: float = 2.2) -> float:
    """
    Convert a standardized game composite_z into a log-likelihood ratio
    log(P(Z | H1_assisted) / P(Z | H0_clean)), clipped to [-1.2, 4.5] so a single
    noisy game cannot dominate indefinitely.
    """
    if not math.isfinite(composite_z):
        return 0.0
    # Under H0: Z ~ N(0, 1); under H1: Z ~ N(z_H1, 1), shifted by 0.65 threshold
    raw_llr = (z_H1 - z_H0) * (composite_z - 0.85)
    return round(float(np.clip(raw_llr, -1.2, 4.5)), 3)


def score_games(
    features_df: pd.DataFrame,
    eval_df: pd.DataFrame | None = None,
    config: DetectorConfig | None = None,
    calibration: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run multi-signal algorithmic anomaly detection across all game-player rows.
    Combines 7 behavioral Z-score channels + ply-level HMM toggle state estimation.
    """
    if features_df.empty:
        return pd.DataFrame()

    cfg = config or DetectorConfig()
    cal = calibration if calibration is not None else load_calibration()
    cohort_tcs = cohort_time_controls(cal)
    c_fixed = float(cal.get("consistency_c", 0.85))
    df = features_df.copy()

    # 1. Fit Regan-style choice sensitivity `s` AND 2-state HMM toggle detector per (game_id, color)
    if eval_df is not None and not eval_df.empty:
        ply_records = []
        valid_plies = eval_df[eval_df["centipawn_loss"].notna() & (eval_df["legal_moves_count"] > 1)]
        for (gid, color), grp in valid_plies.groupby(["game_id", "color"], sort=False):
            s_val = fit_game_sensitivity_s(grp["centipawn_loss"].to_numpy(dtype=float), c_fixed=c_fixed)
            hmm_res = fit_game_hmm_toggle(grp)
            ply_records.append({"game_id": gid, "color": color, "choice_sensitivity_s": s_val, **hmm_res})
        if ply_records:
            df = df.merge(pd.DataFrame(ply_records), on=["game_id", "color"], how="left")

    for col in ("choice_sensitivity_s", "hmm_assisted_frac", "hmm_max_streak", "hmm_llr"):
        if col not in df.columns:
            df[col] = np.nan

    player_groups: dict[tuple[str, str], pd.DataFrame] = {
        (str(p), str(tc)): grp
        for (p, tc), grp in df.groupby(["player", "time_control"], sort=False)
    }
    tc_groups: dict[str, pd.DataFrame] = {
        str(tc): grp for tc, grp in df.groupby("time_control", sort=False)
    }

    scored_rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        player = str(row["player"])
        tc = str(row["time_control"])
        has_cohort_for_tc = tc in cohort_tcs
        elo = row.get("elo")
        p_cohort = player_groups.get((player, tc))
        loo_cohort = p_cohort[p_cohort.index != idx] if p_cohort is not None else pd.DataFrame()
        has_personal_history = len(loo_cohort) >= cfg.min_history_games
        ref_cohort = loo_cohort if has_personal_history else tc_groups.get(tc, df)

        # --- Signal A: Accuracy & Error Suppression (Self + Rating Cohort) ---
        self_acpl = (
            _robust_z(row.get("capped_acpl"), loo_cohort["capped_acpl"], higher_is_suspicious=False)
            if has_personal_history
            else None
        )
        if has_cohort_for_tc:
            cohort_acpl = _lookup_cohort_z(
                row.get("capped_acpl"), "capped_acpl", tc, elo, cal, higher_is_suspicious=False
            )
        else:
            exp_acpl = _expected_acpl_for_elo(elo, tc)
            cohort_acpl = (exp_acpl - float(row["capped_acpl"])) / 18.0 if pd.notna(row.get("capped_acpl")) else None
        z_acpl = _max_valid(self_acpl, cohort_acpl)

        self_top1 = (
            _robust_z(row.get("top1_rate"), loo_cohort["top1_rate"], higher_is_suspicious=True)
            if has_personal_history
            else None
        )
        if has_cohort_for_tc:
            cohort_top1 = _lookup_cohort_z(
                row.get("top1_rate"), "top1_rate", tc, elo, cal, higher_is_suspicious=True
            )
        else:
            exp_top1 = _expected_top1_for_elo(elo, tc)
            cohort_top1 = (float(row["top1_rate"]) - exp_top1) / 0.11 if pd.notna(row.get("top1_rate")) else None
        z_top1 = _max_valid(self_top1, cohort_top1)

        z_blunder = (
            _robust_z(row.get("blunder_rate"), loo_cohort["blunder_rate"], higher_is_suspicious=False)
            if has_personal_history
            else None
        )

        accuracy_parts = [x for x in (z_acpl, z_top1, z_blunder) if x is not None]
        z_accuracy = float(np.mean(accuracy_parts)) if accuracy_parts else None

        # --- Signal B: Critical & Quiet-Critical Precision ---
        self_crit_top1 = (
            _robust_z(row.get("critical_top1_rate"), loo_cohort["critical_top1_rate"], higher_is_suspicious=True)
            if has_personal_history
            else None
        )
        cohort_crit_top1 = (
            _lookup_cohort_z(
                row.get("critical_top1_rate"), "critical_top1_rate", tc, elo, cal, higher_is_suspicious=True
            )
            if has_cohort_for_tc
            else _robust_z(row.get("critical_top1_rate"), ref_cohort["critical_top1_rate"], higher_is_suspicious=True)
        )
        z_crit_top1 = _max_valid(self_crit_top1, cohort_crit_top1)

        z_quiet_crit = (
            _robust_z(
                row.get("quiet_critical_top1_rate"),
                ref_cohort["quiet_critical_top1_rate"],
                higher_is_suspicious=True,
            )
            if "quiet_critical_top1_rate" in ref_cohort.columns
            else None
        )
        z_fast_crit = _robust_z(
            row.get("fast_critical_top1_rate"), ref_cohort["fast_critical_top1_rate"], higher_is_suspicious=True
        )
        crit_parts = [x for x in (z_crit_top1, z_quiet_crit, z_fast_crit) if x is not None]
        z_critical = float(np.mean(crit_parts)) if crit_parts else None

        # --- Signal C: Move-Choice Sensitivity `s` ---
        self_choice = (
            _robust_z(row.get("choice_sensitivity_s"), loo_cohort["choice_sensitivity_s"], higher_is_suspicious=False)
            if has_personal_history
            else None
        )
        cohort_choice = (
            _lookup_cohort_z(
                row.get("choice_sensitivity_s"), "choice_sensitivity_s", tc, elo, cal, higher_is_suspicious=False
            )
            if has_cohort_for_tc
            else _robust_z(
                row.get("choice_sensitivity_s"), ref_cohort["choice_sensitivity_s"], higher_is_suspicious=False
            )
        )
        z_choice = _max_valid(self_choice, cohort_choice)

        # --- Signal D: Machine Surprise & Multi-Depth Horizon Gain ---
        z_mach_surp = (
            _robust_z(
                row.get("mean_machine_surprise"), ref_cohort["mean_machine_surprise"], higher_is_suspicious=True
            )
            if "mean_machine_surprise" in ref_cohort.columns
            else None
        )
        z_deep_only = (
            _robust_z(
                row.get("deep_only_top1_rate"), ref_cohort["deep_only_top1_rate"], higher_is_suspicious=True
            )
            if "deep_only_top1_rate" in ref_cohort.columns
            else None
        )
        surp_parts = [x for x in (z_mach_surp, z_deep_only) if x is not None]
        z_surprise = float(np.mean(surp_parts)) if surp_parts else None

        # --- Signal E: Clock Pacing, Complexity Decoupling & Post-Blunder Timing ---
        if not has_personal_history and has_cohort_for_tc:
            z_cv_time = _lookup_cohort_z(
                row.get("cv_move_time"), "cv_move_time", tc, elo, cal, higher_is_suspicious=False
            )
        else:
            z_cv_time = _robust_z(row.get("cv_move_time"), ref_cohort["cv_move_time"], higher_is_suspicious=False)
        z_gap_corr = _robust_z(
            row.get("time_vs_eval_gap_corr"), ref_cohort["time_vs_eval_gap_corr"], higher_is_suspicious=False
        )
        if z_accuracy is not None and z_accuracy > 0.5:
            timing_parts = [x for x in (z_cv_time, z_gap_corr) if x is not None]
            z_timing = float(np.mean(timing_parts)) if timing_parts else None
        else:
            z_timing = None

        # --- Signal F: Opening-to-Middlegame Phase Surge ---
        z_phase = _robust_z(
            row.get("opening_to_mid_acpl_drop"), ref_cohort["opening_to_mid_acpl_drop"], higher_is_suspicious=True
        )

        # --- Signal G: Contextual Toggling (Comeback Deficit Surge & Time-Scramble Immunity) ---
        z_deficit = (
            _robust_z(row.get("deficit_acpl_drop"), ref_cohort["deficit_acpl_drop"], higher_is_suspicious=True)
            if "deficit_acpl_drop" in ref_cohort.columns
            else None
        )
        z_scramble = (
            _robust_z(row.get("scramble_acpl_drop"), ref_cohort["scramble_acpl_drop"], higher_is_suspicious=True)
            if "scramble_acpl_drop" in ref_cohort.columns
            else None
        )
        toggle_parts = [x for x in (z_deficit, z_scramble) if x is not None]
        z_contextual_toggle = float(np.mean(toggle_parts)) if toggle_parts else None

        composite_z, active_signals = _combine_z_scores(
            [
                (z_accuracy, cfg.weight_accuracy),
                (z_critical, cfg.weight_critical),
                (z_choice, cfg.weight_choice_model),
                (z_surprise, cfg.weight_surprise),
                (z_timing, cfg.weight_timing),
                (z_phase, cfg.weight_phase_shift),
                (z_contextual_toggle, cfg.weight_contextual_toggle),
            ]
        )

        # No normal-tail probability: this score is not calibrated to a null distribution.
        sprt_llr_step = _game_sprt_llr(composite_z)

        all_z_channels = (
            z_accuracy,
            z_critical,
            z_choice,
            z_surprise,
            z_timing,
            z_phase,
            z_contextual_toggle,
        )
        strong_signals = sum(1 for z in all_z_channels if z is not None and z >= 1.5)

        if tc == "unknown":
            status = "untimed_excluded"
        elif composite_z >= cfg.flag_z_threshold and strong_signals >= 2:
            status = "exploratory_outlier"
        elif composite_z >= cfg.review_z_threshold and strong_signals >= 2:
            status = "exploratory_review"
        else:
            status = "no_outlier_detected"

        triggers: list[str] = []
        if z_accuracy is not None and z_accuracy >= 2.0:
            triggers.append(f"accuracy_spike(z={z_accuracy:.2f})")
        if z_critical is not None and z_critical >= 2.0:
            triggers.append(f"critical_precision(z={z_critical:.2f})")
        if z_choice is not None and z_choice >= 2.0:
            triggers.append(f"sharp_choice_s(z={z_choice:.2f})")
        if z_surprise is not None and z_surprise >= 2.0:
            triggers.append(f"machine_surprise(z={z_surprise:.2f})")
        if z_timing is not None and z_timing >= 2.0:
            triggers.append(f"flat_clock_pacing(z={z_timing:.2f})")
        if z_phase is not None and z_phase >= 2.0:
            triggers.append(f"midgame_surge(z={z_phase:.2f})")
        if z_contextual_toggle is not None and z_contextual_toggle >= 2.0:
            triggers.append(f"situational_toggle(z={z_contextual_toggle:.2f})")
        if pd.notna(row.get("hmm_max_streak")) and int(row["hmm_max_streak"]) >= 8:
            triggers.append(f"hmm_streak({int(row['hmm_max_streak'])}plies)")

        baseline_mode = (
            "self+cohort"
            if (has_personal_history and has_cohort_for_tc)
            else ("self_history" if has_personal_history else ("control_cohort" if has_cohort_for_tc else "elo_fallback"))
        )
        cohort_calibrated = baseline_mode in {"self+cohort", "control_cohort"}

        scored_rows.append(
            {
                "game_id": row["game_id"],
                "date": row["date"],
                "time_control": tc,
                "player": player,
                "opponent": row.get("opponent"),
                "color": row["color"],
                "elo": elo,
                "opponent_elo": row.get("opponent_elo"),
                "baseline_mode": baseline_mode,
                "acpl": row.get("acpl"),
                "relative_acpl": row.get("relative_acpl"),
                "top1_rate": row.get("top1_rate"),
                "critical_top1_rate": row.get("critical_top1_rate"),
                "quiet_critical_top1_rate": row.get("quiet_critical_top1_rate"),
                "choice_sensitivity_s": row.get("choice_sensitivity_s"),
                "mean_machine_surprise": row.get("mean_machine_surprise"),
                "hmm_assisted_frac": row.get("hmm_assisted_frac"),
                "hmm_max_streak": row.get("hmm_max_streak"),
                "z_accuracy": round(z_accuracy, 3) if z_accuracy is not None else None,
                "z_critical": round(z_critical, 3) if z_critical is not None else None,
                "z_choice": round(z_choice, 3) if z_choice is not None else None,
                "z_surprise": round(z_surprise, 3) if z_surprise is not None else None,
                "z_timing": round(z_timing, 3) if z_timing is not None else None,
                "z_phase_shift": round(z_phase, 3) if z_phase is not None else None,
                "z_contextual_toggle": (
                    round(z_contextual_toggle, 3) if z_contextual_toggle is not None else None
                ),
                "composite_z": composite_z,
                "sprt_llr_step": sprt_llr_step,
                "null_p_value": None,
                "calibration_status": "cohort_baseline" if cohort_calibrated else "uncalibrated",
                "status": status,
                "triggers": ", ".join(triggers) if triggers else "none",
            }
        )

    result = pd.DataFrame(scored_rows)
    return result.sort_values(["composite_z", "game_id"], ascending=[False, True]).reset_index(drop=True)


def aggregate_player_scores(
    scored_df: pd.DataFrame,
    features_df: pd.DataFrame | None = None,
    window: int = 5,
    config: DetectorConfig | None = None,
) -> pd.DataFrame:
    """
    Roll up game-level anomaly scores into account/player-level verdicts using:
      - Rolling window Z-score peaks (`max_rolling5_z`)
      - One-sided CUSUM streak detection (`max_cusum`)
      - Heuristic SPRT-like accumulation (`max_sprt_llr`); NOT a calibrated LLR
      - Longitudinal meta-history checks (post-loss rebound & opening-to-middlegame contradiction)
    """
    if scored_df.empty:
        return pd.DataFrame()

    cfg = config or DetectorConfig()
    df = scored_df[scored_df["time_control"] != "unknown"].copy()
    if df.empty:
        return pd.DataFrame()

    if features_df is not None and not features_df.empty:
        meta_cols = [
            c
            for c in ("game_id", "color", "prev_game_bad_result", "opening_exit_eval", "middlegame_acpl")
            if c in features_df.columns
        ]
        if len(meta_cols) > 2:
            df = df.merge(features_df[meta_cols], on=["game_id", "color"], how="left")

    df = df.sort_values(["player", "time_control", "date", "game_id"]).reset_index(drop=True)

    summaries: list[dict[str, Any]] = []
    for (player, tc), grp in df.groupby(["player", "time_control"], sort=False):
        z_vals = grp["composite_z"].to_numpy(dtype=float)
        n_games = len(z_vals)

        if n_games >= window:
            kernel = np.ones(window) / math.sqrt(window)
            rolling_z = np.convolve(z_vals, kernel, mode="valid")
            max_rolling_z = float(np.max(rolling_z))
        else:
            max_rolling_z = float(np.sum(z_vals) / math.sqrt(max(1, n_games)))

        cusum = 0.0
        max_cusum = 0.0
        sprt_llr = 0.0
        max_sprt_llr = 0.0

        for z in z_vals:
            cusum = max(0.0, cusum + (z - 0.5))
            max_cusum = max(max_cusum, cusum)
            step_llr = _game_sprt_llr(float(z))
            sprt_llr = max(0.0, sprt_llr + step_llr)
            max_sprt_llr = max(max_sprt_llr, sprt_llr)

        # Meta-history Check 1: Post-Loss Rebound Toggling (composite_z surge right after a bad loss)
        rebound_delta = 0.0
        if "prev_game_bad_result" in grp.columns and n_games >= 4:
            after_loss = grp[grp["prev_game_bad_result"] == 1]["composite_z"]
            after_norm = grp[grp["prev_game_bad_result"] == 0]["composite_z"]
            if len(after_loss) >= 1 and len(after_norm) >= 2:
                rebound_delta = round(float(after_loss.mean() - after_norm.mean()), 3)

        # Meta-history Check 2: Weak Opening Theory vs. Elite Middlegame Execution
        opening_contradiction = False
        if "opening_exit_eval" in grp.columns and "middlegame_acpl" in grp.columns:
            op_eval = pd.to_numeric(grp["opening_exit_eval"], errors="coerce").dropna()
            mid_acpl = pd.to_numeric(grp["middlegame_acpl"], errors="coerce").dropna()
            if len(op_eval) >= 3 and len(mid_acpl) >= 3:
                opening_contradiction = bool(op_eval.mean() <= -75.0 and mid_acpl.mean() <= 18.0)

        # Heuristic priorities only. These are not cheat labels or guarantees of innocence.
        flagged_count = int((grp["status"] == "exploratory_outlier").sum())
        if max_sprt_llr >= cfg.sprt_flag_llr or (flagged_count >= 2 and max_rolling_z >= 3.2):
            account_verdict = "manual_review_recommended"
        elif max_sprt_llr >= cfg.sprt_review_llr or flagged_count >= 1:
            account_verdict = "statistical_outlier"
        else:
            account_verdict = "no_outlier_detected"

        modes = set()
        if "baseline_mode" in grp.columns:
            modes = {str(value) for value in grp["baseline_mode"].dropna()}
        cohort_calibrated = bool(modes & {"self+cohort", "control_cohort"})
        calibration_status = "cohort_baseline" if cohort_calibrated else "uncalibrated"
        calibration_note = (
            "Cohort Z-score calibration loaded; account-level empirical null not configured."
            if cohort_calibrated
            else "Heuristic scores only; no independent labelled account-level null distribution."
        )

        summaries.append(
            {
                "player": player,
                "time_control": tc,
                "games": n_games,
                "mean_elo": round(float(grp["elo"].dropna().mean()), 0) if grp["elo"].notna().any() else None,
                "mean_acpl": round(float(grp["acpl"].dropna().mean()), 2),
                "mean_z": round(float(np.mean(z_vals)), 3),
                "max_single_z": round(float(np.max(z_vals)), 3),
                f"max_rolling{window}_z": round(max_rolling_z, 3),
                "max_cusum": round(max_cusum, 3),
                "max_sprt_llr": round(max_sprt_llr, 3),
                "rebound_z_delta": rebound_delta,
                "opening_contradiction": opening_contradiction,
                "null_p_value": None,
                "calibration_status": calibration_status,
                "calibration_note": calibration_note,
                "account_verdict": account_verdict,
                "flagged_games": flagged_count,
                "flagged_pct": round(100.0 * flagged_count / n_games, 1),
            }
        )

    out = pd.DataFrame(summaries)
    return out.sort_values(["max_sprt_llr", "max_cusum", "max_single_z"], ascending=[False, False, False]).reset_index(
        drop=True
    )
