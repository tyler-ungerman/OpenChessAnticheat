"""Account-level empirical null screening, deliberately separate from cheat probabilities.

Use independently sourced accounts as calibration references. Never use games from the
queried account, evaluation set, or model-development set as null controls.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def empirical_account_tail(
    score: float,
    controls: pd.DataFrame,
    *,
    account_id: str,
    time_control: str,
    statistic: str = "max_sprt_llr",
    min_controls: int = 99,
) -> dict[str, Any]:
    """Conservative right-tail conformal rank, per *account* and time control.

    This is an empirical outlier p-value under exchangeability of null accounts,
    not P(cheating | data). No decision is emitted without enough independent
    controls. max_sprt_llr is a legacy name for a heuristic accumulation, not LLR.
    """
    if statistic not in {"max_sprt_llr", "max_cusum", "max_single_z"}:
        raise ValueError("Unsupported account-level statistic")
    required = {"player", "time_control", statistic}
    if not required.issubset(controls.columns):
        raise ValueError(f"Missing calibration columns: {sorted(required - set(controls.columns))}")
    if not math.isfinite(float(score)):
        raise ValueError("Score must be finite")
    if not str(account_id).strip():
        raise ValueError("Account ID is required")
    sample = controls.loc[
        (controls["time_control"].astype(str) == str(time_control))
        & (controls["player"].astype(str).str.casefold() != str(account_id).casefold()),
        ["player", statistic],
    ].copy()
    sample[statistic] = pd.to_numeric(sample[statistic], errors="coerce")
    sample = sample[np.isfinite(sample[statistic])]
    if sample["player"].astype(str).str.casefold().duplicated().any():
        raise ValueError("Null calibration requires one score per independent account/time-control")
    n = len(sample)
    response: dict[str, Any] = {
        "null_p_value": None,
        "n_independent_controls": n,
        "statistic": statistic,
        "time_control": time_control,
        "calibration_status": "insufficient_controls",
        "account_verdict": "not_calibrated",
    }
    if n < min_controls:
        return response
    # Inclusive ties are conservative. Add-one correction prevents a zero p-value.
    exceedances = int(np.count_nonzero(sample[statistic].to_numpy(dtype=float) >= score))
    p_value = (exceedances + 1) / (n + 1)
    response.update(
        null_p_value=p_value,
        empirical_tail_percentile=100.0 * (1.0 - p_value),
        calibration_status="empirical_null",
        account_verdict="manual_review_recommended" if p_value <= 0.01 else "not_an_empirical_outlier",
        resolution_floor=1.0 / (n + 1),
    )
    return response


def validate_label_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Reject missing/contradictory labels and cross-split account leakage.

    Expected one or more games per account with columns: platform, player, label,
    label_source, label_checked_at, split. Labels: 1 confirmed *platform action*,
    0 presumed clean (not proven non-cheating). Neither label proves an individual game.
    """
    required = {"platform", "player", "label", "label_source", "label_checked_at", "split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(manifest.columns))}")
    frame = manifest.copy()
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Manifest is empty or has missing fields")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    if not frame["split"].isin(["train", "calibration", "test"]).all():
        raise ValueError("Unknown split")
    if (frame["label_source"].astype(str).str.strip() == "").any():
        raise ValueError("Label provenance is required")
    frame["account_key"] = (
        frame["platform"].astype(str).str.casefold().str.strip() + ":"
        + frame["player"].astype(str).str.casefold().str.strip()
    )
    if frame["account_key"].str.endswith(":").any():
        raise ValueError("Account identity is missing")
    if frame.groupby("account_key")["split"].nunique().gt(1).any():
        raise ValueError("Account appears in more than one split")
    if frame.groupby("account_key")["label"].nunique().gt(1).any():
        raise ValueError("Contradictory labels for an account")
    return frame
