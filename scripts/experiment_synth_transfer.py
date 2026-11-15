#!/usr/bin/env python3
"""Cross-time-control synthetic-to-real *move-level* transfer experiment.

Uses ChessFraud-Synth's observed human moves vs counterfactual engine moves.
Evaluation ONLY on real logged-assistance ChessFraud tournament moves from
held-out human participants. Does not export a product classifier.

Run: python scripts/experiment_synth_transfer.py --out data/synth_transfer.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = "artemlepin/chess-fraud"
REVISION = "bd2804f268bf07c306217929db9b8dda5803392b"
FEATURES = (
    "match_sf15", "match_sf9", "match_maia", "match_allie",
    "sf_agreement", "human_agreement", "abs_eval", "ply_fraction",
)
SYNTH_SOURCES = ("move_stockfish_15", "move_maia2_2050", "move_allie_2500")


def feature_rows(rows: pd.DataFrame, chosen: pd.Series) -> pd.DataFrame:
    """Same feature transformation for observed and alternative moves."""
    sf15 = rows["move_stockfish_15"].fillna("").astype(str)
    sf9 = rows["move_stockfish_9"].fillna("").astype(str)
    maia = rows["move_maia2_2050"].fillna("").astype(str)
    allie = rows["move_allie_2500"].fillna("").astype(str)
    move = chosen.fillna("").astype(str)
    ply = pd.to_numeric(rows["half_move"], errors="coerce")
    eval_cp = pd.to_numeric(rows["eval_before"], errors="coerce")
    return pd.DataFrame({
        "match_sf15": (move == sf15).astype(float),
        "match_sf9": (move == sf9).astype(float),
        "match_maia": (move == maia).astype(float),
        "match_allie": (move == allie).astype(float),
        "sf_agreement": (sf15 == sf9).astype(float),
        "human_agreement": (maia == allie).astype(float),
        "abs_eval": eval_cp.abs().clip(upper=1000),
        "ply_fraction": (ply / 120).clip(upper=1),
    }, index=rows.index)


def train_synthetic(synth: pd.DataFrame, max_positions: int = 60000):
    required = {"is_used", "move_player", "move_stockfish_15", "move_stockfish_9",
                "move_maia2_2050", "move_allie_2500", "half_move",
                "eval_before", "time_control"}
    if not required.issubset(synth):
        raise ValueError(f"Missing synthetic columns: {sorted(required - set(synth))}")
    used = synth[synth["is_used"].fillna(False).astype(bool)].copy()
    used = used.dropna(subset=["move_player", "move_stockfish_15"])
    if len(used) > max_positions:
        used = used.sample(n=max_positions, random_state=20261115)
    # Fixed choice: one synthetic assistance source per source position.
    rng = np.random.default_rng(20261115)
    source_names = rng.choice(SYNTH_SOURCES, size=len(used))
    chosen = pd.Series([row.get(source) for (_, row), source in
                        zip(used.iterrows(), source_names)], index=used.index)
    valid = chosen.notna() & (chosen.astype(str) != used["move_player"].astype(str))
    used = used.loc[valid]
    chosen = chosen.loc[valid]
    negatives = feature_rows(used, used["move_player"])
    positives = feature_rows(used, chosen)
    X = pd.concat([negatives, positives], ignore_index=True)
    y = np.r_[np.zeros(len(negatives)), np.ones(len(positives))]
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(C=0.1, max_iter=1000, random_state=20261115)
    )
    model.fit(X, y)
    tc = used["time_control"].astype(str).value_counts().head(12).to_dict()
    return model, len(used), {str(k): int(v) for k, v in tc.items()}


def evaluate_tournament(model, tournament: pd.DataFrame, *,
                        holdout_players: list[str]) -> dict:
    required = {"player_id", "is_used", "is_cheating_move", "move_player",
                "move_stockfish_15", "move_stockfish_9", "move_maia2_2050",
                "move_allie_2500", "half_move", "eval_before"}
    if not required.issubset(tournament):
        raise ValueError(f"Missing tournament columns: {sorted(required - set(tournament))}")
    selected = tournament[
        tournament["player_id"].astype(str).isin(holdout_players)
        & tournament["is_used"].fillna(False).astype(bool)
    ].copy()
    selected = selected.dropna(subset=["is_cheating_move"])
    if selected.empty:
        raise ValueError("No held-out labelled moves")
    y = selected["is_cheating_move"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("Holdout missing a class")
    features = feature_rows(selected, selected["move_player"])
    scores = model.predict_proba(features)[:, 1]
    baseline = features["match_sf15"].to_numpy(dtype=float)
    return {
        "moves": len(selected), "positive": int(y.sum()),
        "holdout_players": len(set(selected["player_id"])),
        "synthetic_transfer_roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "sf15_agreement_roc_auc": round(float(roc_auc_score(y, baseline)), 4),
        "synthetic_transfer_average_precision": round(float(average_precision_score(y, scores)), 4),
        "sf15_agreement_average_precision": round(float(average_precision_score(y, baseline)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-positions", type=int, default=60000)
    args = parser.parse_args()
    if args.max_positions < 100:
        parser.error("Need at least 100 positions")

    # The same 10 players as the earlier game-level holdout. This is a single
    # preregistered transfer test, not a tuning set.
    from train_chessfraud_model import from_chessfraud, split_accounts
    tournament = load_dataset(REPO, "chess_fraud", split="full", revision=REVISION).to_pandas()
    game_rows = from_chessfraud(tournament)
    _, _, test_idx = split_accounts(game_rows)
    heldout = sorted(game_rows.iloc[test_idx]["player_id"].astype(str).unique())
    synth = load_dataset(REPO, "chess_fraud_synth", split="train",
                         revision=REVISION).to_pandas()
    model, positions, tc_counts = train_synthetic(synth, max_positions=args.max_positions)
    metrics = evaluate_tournament(model, tournament, holdout_players=heldout)
    result = {
        "experiment": "synth-to-controlled-real-move-transfer-v1",
        "dataset_revision": REVISION,
        "synthetic_training_positions": positions,
        "synthetic_time_control_counts": tc_counts,
        "heldout_tournament_move_metrics": metrics,
        "verdict": ("CONTINUE" if metrics["synthetic_transfer_roc_auc"] >=
                    metrics["sf15_agreement_roc_auc"] + 0.02 else "KILL"),
        "limits": ["Synthetic positives are candidate move substitutions, not observed cheating",
                   "Real holdout remains only 5+0", "Move-level detector cannot give account cheating probabilities",
                   "No test labels, player IDs or future game results used as features"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
