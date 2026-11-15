from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _classify_time_control(tc: str | None) -> str:
    """Bucket raw TimeControl header (e.g., '180+2') or normalized class into bullet/blitz/rapid/daily."""
    if not tc or not isinstance(tc, str):
        return "unknown"
    lowered = tc.strip().lower()
    if lowered in ("bullet", "blitz", "rapid", "daily", "classical"):
        return "rapid" if lowered == "classical" else lowered
    if "/" in lowered:
        return "daily"
    try:
        parts = lowered.split("+")
        base = float(parts[0])
        inc = float(parts[1]) if len(parts) > 1 else 0.0
    except ValueError:
        return "unknown"
    estimated_duration = base + 40.0 * inc
    if estimated_duration < 180:
        return "bullet"
    if estimated_duration < 600:
        return "blitz"
    return "rapid"


def _assign_phase(df: pd.DataFrame) -> pd.Series:
    """Classify each ply into opening, middlegame, or endgame using ply and piece count."""
    phases = pd.Series("middlegame", index=df.index, dtype="object")
    phases = phases.mask(df["move_number"] <= 10, "opening")
    if "piece_count" in df.columns:
        phases = phases.mask(df["piece_count"].notna() & (df["piece_count"] <= 12), "endgame")
    return phases


def _safe_pearson(x: pd.Series, y: pd.Series) -> float | None:
    """Compute Pearson correlation safely, returning None if variance is zero or sample is too small."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 6:
        return None
    if valid["x"].std() < 1e-9 or valid["y"].std() < 1e-9:
        return None
    corr = float(valid["x"].corr(valid["y"]))
    return round(corr, 4) if math.isfinite(corr) else None


def _ensure_move_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure optional move-level columns exist even when loading external JSONL
    benchmark files that only have core Stockfish columns.
    """
    out = df.copy()
    if "is_quiet" not in out.columns:
        if "is_capture" in out.columns and "is_check" in out.columns:
            out["is_quiet"] = (~out["is_capture"].fillna(False)) & (~out["is_check"].fillna(False))
        elif "san" in out.columns or "move_san" in out.columns:
            san_col = out["move_san"] if "move_san" in out.columns else out["san"]
            san_str = san_col.astype(str)
            out["is_quiet"] = (~san_str.str.contains("x")) & (~san_str.str.contains(r"[\+#]"))
        else:
            out["is_quiet"] = True

    if "human_prior_prob" not in out.columns or out["human_prior_prob"].isna().all():
        cpl = pd.to_numeric(out.get("centipawn_loss", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
        gap = pd.to_numeric(out.get("eval_gap", 30), errors="coerce").fillna(30.0).clip(lower=0.0)
        legal = pd.to_numeric(out.get("legal_moves_count", 20), errors="coerce").fillna(20.0).clip(lower=1.0)
        quiet_mult = np.where(out["is_quiet"], 0.72, 1.20)
        complexity_pen = np.exp(-gap / 220.0)
        unnorm = np.exp(-(cpl / 100.0) / 0.65) * quiet_mult * complexity_pen
        out["human_prior_prob"] = np.clip(unnorm / (unnorm + 0.18 * legal), 0.005, 0.99).round(4)

    if "machine_surprise" not in out.columns:
        is_top1 = out["is_top1"].fillna(False).astype(bool)
        gap = pd.to_numeric(out.get("eval_gap", 0), errors="coerce").fillna(0.0)
        hp = pd.to_numeric(out["human_prior_prob"], errors="coerce").fillna(0.25).clip(lower=1e-3)
        out["machine_surprise"] = np.where(is_top1 & (gap >= 40.0), -np.log(hp), 0.0).round(4)

    if "is_deep_only_top1" not in out.columns:
        is_top1 = out["is_top1"].fillna(False).astype(bool)
        gap = pd.to_numeric(out.get("eval_gap", 0), errors="coerce").fillna(0.0)
        out["is_deep_only_top1"] = is_top1 & out["is_quiet"] & (gap >= 90.0)

    if "rank_instability" not in out.columns:
        out["rank_instability"] = 0.0

    return out


def _compute_side_features(moves: pd.DataFrame) -> dict[str, Any]:
    """Aggregate move-level rows for one player in one game into behavioral features."""
    total_plies = len(moves)
    evaluated = moves[moves["centipawn_loss"].notna() & (moves["legal_moves_count"] > 1)].copy()
    n_eval = len(evaluated)

    # 1. Core Accuracy & Centipawn Loss Metrics
    if n_eval > 0:
        cpl = pd.to_numeric(evaluated["centipawn_loss"], errors="coerce")
        wpl = pd.to_numeric(evaluated["win_prob_loss"], errors="coerce").fillna(0.0)
        eval_before = pd.to_numeric(evaluated["eval_before"], errors="coerce")
        eval_after = pd.to_numeric(evaluated["eval_after"], errors="coerce")
        non_won = evaluated[eval_before.abs() <= 250]

        acpl = float(cpl.mean())
        capped_acpl = float(cpl.clip(upper=300).mean())
        median_cpl = float(cpl.median())
        non_won_acpl = float(non_won["centipawn_loss"].mean()) if len(non_won) > 0 else None
        mean_wp_loss = float(wpl.mean())
        top1_rate = float(evaluated["is_top1"].mean())
        top3_rate = float(evaluated["is_top3"].mean())
        blunder_rate = float((cpl > 200).mean())
        mistake_rate = float((cpl > 100).mean())
        inaccuracy_rate = float((cpl > 50).mean())
        final_eval_cp = float(eval_after.dropna().iloc[-1]) if eval_after.notna().any() else 0.0
    else:
        acpl = capped_acpl = median_cpl = non_won_acpl = mean_wp_loss = None
        top1_rate = top3_rate = blunder_rate = mistake_rate = inaccuracy_rate = None
        final_eval_cp = 0.0

    # 2. Critical & Quiet-Critical Positions
    critical = evaluated[
        evaluated["eval_gap"].notna()
        & (evaluated["eval_gap"] >= 80)
        & (evaluated["eval_before"].abs() <= 250)
    ]
    n_critical = len(critical)
    critical_top1_rate = float(critical["is_top1"].mean()) if n_critical > 0 else None
    critical_acpl = float(critical["centipawn_loss"].mean()) if n_critical > 0 else None

    quiet_critical = critical[critical["is_quiet"].fillna(True)]
    quiet_critical_top1_rate = (
        float(quiet_critical["is_top1"].mean()) if len(quiet_critical) > 0 else critical_top1_rate
    )

    fast_critical = critical[critical["time_spent"].notna() & (critical["time_spent"] <= 3.5)]
    fast_critical_top1_rate = (
        float(fast_critical["is_top1"].mean()) if len(fast_critical) > 0 else None
    )

    # 3. Horizon Gain & Machine Surprise (Human-Prior Residual)
    if n_eval > 0:
        deep_only_top1_rate = float(evaluated["is_deep_only_top1"].fillna(False).mean())
        surprise_vals = pd.to_numeric(evaluated["machine_surprise"], errors="coerce").dropna()
        mean_machine_surprise = float(surprise_vals.mean()) if len(surprise_vals) > 0 else 0.0
        max_machine_surprise = float(surprise_vals.max()) if len(surprise_vals) > 0 else 0.0
    else:
        deep_only_top1_rate = mean_machine_surprise = max_machine_surprise = None

    # 4. Game Phase Splits & Opening Theory vs. Middlegame Skill Contradiction
    phase_metrics: dict[str, Any] = {}
    for phase_name in ("opening", "middlegame", "endgame"):
        sub = evaluated[evaluated["phase"] == phase_name]
        phase_metrics[f"{phase_name}_moves"] = len(sub)
        phase_metrics[f"{phase_name}_acpl"] = (
            round(float(sub["centipawn_loss"].mean()), 2) if len(sub) > 0 else None
        )
        phase_metrics[f"{phase_name}_top1_rate"] = (
            round(float(sub["is_top1"].mean()), 4) if len(sub) > 0 else None
        )

    opening_to_mid_acpl_drop = (
        round(phase_metrics["opening_acpl"] - phase_metrics["middlegame_acpl"], 2)
        if phase_metrics["opening_acpl"] is not None
        and phase_metrics["middlegame_acpl"] is not None
        else None
    )

    opening_moves = moves[moves["move_number"] <= 10]
    opening_exit_eval = (
        float(pd.to_numeric(opening_moves["eval_after"], errors="coerce").dropna().iloc[-1])
        if not opening_moves.empty and opening_moves["eval_after"].notna().any()
        else None
    )

    # 5. Conditional Deficit / Comeback Toggling (Accuracy when trailing vs. equal/ahead)
    if n_eval > 0:
        eb = pd.to_numeric(evaluated["eval_before"], errors="coerce")
        trailing = evaluated[(eb < -80) & (eb >= -400)]
        comfortable = evaluated[(eb >= -50) & (eb <= 250)]
        deficit_acpl = float(trailing["centipawn_loss"].mean()) if len(trailing) >= 3 else None
        deficit_top1_rate = float(trailing["is_top1"].mean()) if len(trailing) >= 3 else None
        comfort_acpl = float(comfortable["centipawn_loss"].mean()) if len(comfortable) >= 3 else None
        # Positive deficit_acpl_drop = player becomes much MORE accurate when losing (comeback toggle)
        deficit_acpl_drop = (
            round(comfort_acpl - deficit_acpl, 2)
            if (comfort_acpl is not None and deficit_acpl is not None)
            else None
        )
    else:
        deficit_acpl = deficit_top1_rate = deficit_acpl_drop = None

    # 6. Timing, Pacing, Time-Scramble Immunity & Post-Blunder Response
    timed = moves[moves["time_spent"].notna()].copy()
    clock_coverage = len(timed) / total_plies if total_plies > 0 else 0.0

    if len(timed) > 0:
        t = pd.to_numeric(timed["time_spent"], errors="coerce").dropna()
        mean_move_time = float(t.mean()) if len(t) > 0 else None
        sd_move_time = float(t.std()) if len(t) > 1 else 0.0
        cv_move_time = (
            (sd_move_time / mean_move_time)
            if (mean_move_time is not None and mean_move_time > 1e-6)
            else None
        )
        median_move_time = float(t.median()) if len(t) > 0 else None
        max_move_time = float(t.max()) if len(t) > 0 else None
        near_instant_rate = float((t <= 1.0).mean()) if len(t) > 0 else None
        long_think_rate = float((t >= 15.0).mean()) if len(t) > 0 else None

        eval_timed = evaluated[evaluated["time_spent"].notna()]
        log_time = np.log1p(pd.to_numeric(eval_timed["time_spent"], errors="coerce"))
        time_gap_corr = _safe_pearson(log_time, eval_timed["eval_gap"])
        time_legal_corr = _safe_pearson(log_time, eval_timed["legal_moves_count"])

        # Time-scramble accuracy (< 20s remaining on clock vs > 30s remaining)
        if "clock_remaining" in evaluated.columns and evaluated["clock_remaining"].notna().any():
            clk = pd.to_numeric(evaluated["clock_remaining"], errors="coerce")
            scramble_moves = evaluated[clk <= 20.0]
            normal_clk_moves = evaluated[clk > 30.0]
            if len(scramble_moves) >= 3 and len(normal_clk_moves) >= 5:
                s_acpl = float(scramble_moves["centipawn_loss"].clip(upper=300).mean())
                n_acpl = float(normal_clk_moves["centipawn_loss"].clip(upper=300).mean())
                # Positive scramble_acpl_drop = MORE accurate under <20s clock pressure than normal time
                scramble_acpl_drop = round(n_acpl - s_acpl, 2)
            else:
                scramble_acpl_drop = None
        else:
            scramble_acpl_drop = None

        # Post-opponent-blunder response time ratio (humans pause to double-check; overlays don't)
        if "prev_opp_cpl" in timed.columns and median_move_time and median_move_time > 0.5:
            opp_blunders = timed[pd.to_numeric(timed["prev_opp_cpl"], errors="coerce") >= 150]
            if len(opp_blunders) >= 2:
                post_blunder_time_ratio = round(
                    float(opp_blunders["time_spent"].median()) / median_move_time, 3
                )
            else:
                post_blunder_time_ratio = None
        else:
            post_blunder_time_ratio = None
    else:
        mean_move_time = sd_move_time = cv_move_time = median_move_time = max_move_time = None
        near_instant_rate = long_think_rate = time_gap_corr = time_legal_corr = None
        scramble_acpl_drop = post_blunder_time_ratio = None

    return {
        "total_moves": total_plies,
        "evaluated_moves": n_eval,
        "acpl": round(acpl, 2) if acpl is not None else None,
        "capped_acpl": round(capped_acpl, 2) if capped_acpl is not None else None,
        "median_cpl": round(median_cpl, 2) if median_cpl is not None else None,
        "non_won_acpl": round(non_won_acpl, 2) if non_won_acpl is not None else None,
        "mean_win_prob_loss": round(mean_wp_loss, 4) if mean_wp_loss is not None else None,
        "top1_rate": round(top1_rate, 4) if top1_rate is not None else None,
        "top3_rate": round(top3_rate, 4) if top3_rate is not None else None,
        "blunder_rate": round(blunder_rate, 4) if blunder_rate is not None else None,
        "mistake_rate": round(mistake_rate, 4) if mistake_rate is not None else None,
        "inaccuracy_rate": round(inaccuracy_rate, 4) if inaccuracy_rate is not None else None,
        "critical_moves": n_critical,
        "critical_top1_rate": round(critical_top1_rate, 4) if critical_top1_rate is not None else None,
        "quiet_critical_top1_rate": (
            round(quiet_critical_top1_rate, 4) if quiet_critical_top1_rate is not None else None
        ),
        "critical_acpl": round(critical_acpl, 2) if critical_acpl is not None else None,
        "fast_critical_moves": len(fast_critical),
        "fast_critical_top1_rate": (
            round(fast_critical_top1_rate, 4) if fast_critical_top1_rate is not None else None
        ),
        "deep_only_top1_rate": (
            round(deep_only_top1_rate, 4) if deep_only_top1_rate is not None else None
        ),
        "mean_machine_surprise": (
            round(mean_machine_surprise, 4) if mean_machine_surprise is not None else None
        ),
        "max_machine_surprise": (
            round(max_machine_surprise, 4) if max_machine_surprise is not None else None
        ),
        **phase_metrics,
        "opening_to_mid_acpl_drop": opening_to_mid_acpl_drop,
        "opening_exit_eval": round(opening_exit_eval, 1) if opening_exit_eval is not None else None,
        "deficit_acpl": round(deficit_acpl, 2) if deficit_acpl is not None else None,
        "deficit_top1_rate": round(deficit_top1_rate, 4) if deficit_top1_rate is not None else None,
        "deficit_acpl_drop": deficit_acpl_drop,
        "scramble_acpl_drop": scramble_acpl_drop,
        "post_blunder_time_ratio": post_blunder_time_ratio,
        "final_eval_cp": round(final_eval_cp, 1),
        "clock_coverage": round(clock_coverage, 4),
        "mean_move_time_sec": round(mean_move_time, 3) if mean_move_time is not None else None,
        "sd_move_time_sec": round(sd_move_time, 3) if sd_move_time is not None else None,
        "cv_move_time": round(cv_move_time, 4) if cv_move_time is not None else None,
        "median_move_time_sec": round(median_move_time, 3) if median_move_time is not None else None,
        "max_move_time_sec": round(max_move_time, 3) if max_move_time is not None else None,
        "near_instant_move_rate": round(near_instant_rate, 4) if near_instant_rate is not None else None,
        "long_think_rate": round(long_think_rate, 4) if long_think_rate is not None else None,
        "time_vs_eval_gap_corr": time_gap_corr,
        "time_vs_legal_moves_corr": time_legal_corr,
    }


def build_features(eval_df: pd.DataFrame, min_evaluated_moves: int = 5) -> pd.DataFrame:
    """
    Transform ply-level `eval_df` into game-player-level behavioral features
    (one row per player per game), enriched with opponent-relative comparisons,
    surprise residuals, and rolling player baselines.
    """
    if eval_df.empty:
        return pd.DataFrame()

    df = _ensure_move_columns(eval_df)
    df = df.sort_values(["game_id", "ply"]).reset_index(drop=True)
    df["phase"] = _assign_phase(df)
    df["elo_num"] = pd.to_numeric(df["elo"], errors="coerce")

    # Attach opponent's immediately preceding CPL to measure post-blunder response timing
    df["prev_opp_cpl"] = df.groupby("game_id")["centipawn_loss"].shift(1)

    game_side_rows: list[dict[str, Any]] = []

    for (game_id, color), side_df in df.groupby(["game_id", "color"], sort=False):
        side_df = side_df.sort_values("ply")
        first = side_df.iloc[0]
        feats = _compute_side_features(side_df)

        if feats["evaluated_moves"] < min_evaluated_moves:
            continue

        tc_raw = first.get("time_control")
        elo_val = first.get("elo_num")
        game_side_rows.append(
            {
                "game_id": game_id,
                "date": first.get("date"),
                "time_control_raw": tc_raw,
                "time_control": _classify_time_control(tc_raw),
                "color": color,
                "player": first.get("player"),
                "elo": int(elo_val) if pd.notna(elo_val) else None,
                **feats,
            }
        )

    features_df = pd.DataFrame(game_side_rows)
    if features_df.empty:
        return features_df

    # Join opponent metrics from the same game_id to compute relative performance deltas
    opp_cols = features_df[
        ["game_id", "color", "player", "elo", "acpl", "capped_acpl", "top1_rate", "mean_win_prob_loss"]
    ].rename(
        columns={
            "color": "opp_color",
            "player": "opponent",
            "elo": "opponent_elo",
            "acpl": "opponent_acpl",
            "capped_acpl": "opponent_capped_acpl",
            "top1_rate": "opponent_top1_rate",
            "mean_win_prob_loss": "opponent_win_prob_loss",
        }
    )

    merged = features_df.merge(opp_cols, on="game_id", how="left")
    merged = merged[merged["color"] != merged["opp_color"]].drop(columns=["opp_color"]).copy()

    merged["elo_diff"] = merged["elo"] - merged["opponent_elo"]
    merged["relative_acpl"] = (merged["acpl"] - merged["opponent_acpl"]).round(2)
    merged["relative_capped_acpl"] = (merged["capped_acpl"] - merged["opponent_capped_acpl"]).round(2)
    merged["relative_top1_rate"] = (merged["top1_rate"] - merged["opponent_top1_rate"]).round(4)

    # Player-level historical baseline (leave-one-out / expanding within time_control)
    merged = merged.sort_values(["player", "time_control", "date", "game_id"]).reset_index(drop=True)
    grp = merged.groupby(["player", "time_control"], sort=False)

    player_tc_games = grp["acpl"].transform("count")
    player_tc_sum_acpl = grp["acpl"].transform("sum")
    player_tc_sum_top1 = grp["top1_rate"].transform("sum")

    loo_count = (player_tc_games - 1).replace(0, np.nan)
    merged["player_baseline_acpl"] = ((player_tc_sum_acpl - merged["acpl"]) / loo_count).round(2)
    merged["player_baseline_top1_rate"] = ((player_tc_sum_top1 - merged["top1_rate"]) / loo_count).round(4)
    merged["acpl_vs_baseline"] = (merged["acpl"] - merged["player_baseline_acpl"]).round(2)
    merged["top1_vs_baseline"] = (merged["top1_rate"] - merged["player_baseline_top1_rate"]).round(4)

    # Track whether the player lost or blundered their immediately preceding game (for post-loss rebound)
    merged["prev_game_bad_result"] = (
        grp["final_eval_cp"].shift(1).fillna(0.0) < -150.0
    ).astype(int)

    return merged.sort_values(["game_id", "color"]).reset_index(drop=True)