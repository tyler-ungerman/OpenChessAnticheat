#!/usr/bin/env python3
"""Train a compact, exportable ChessFraud game classifier on real assistance labels.

Freeze accounts before fitting ANY preprocessing. Publish only if the untouched
player-held-out test beats the no-training engine-match baseline. No probability
calibration on test data or use of cheat labels as input features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.src.detectors.pretrained_features import FEATURE_NAMES, from_chessfraud

SOURCE = ("https://github.com/artem-lepin-ml/chess-fraud/releases/download/"
          "v1.0-data/chess_fraud_tournament.csv")
RANDOM_STATE = 20261115


def split_accounts(frame: pd.DataFrame):
    """Non-overlapping player partitions (60/20/20), stable on same source."""
    groups = frame["player_id"].astype(str).to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_val, test = next(outer.split(frame, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE + 1)
    train_sub, valid_sub = next(inner.split(frame.iloc[train_val], groups=groups[train_val]))
    train, valid = train_val[train_sub], train_val[valid_sub]
    parts = (train, valid, test)
    accounts = [set(groups[idx]) for idx in parts]
    assert all(not accounts[i] & accounts[j] for i in range(3) for j in range(i + 1, 3))
    return parts


def auc(y, scores):
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "n_games": int(len(y)),
        "positive_games": int(y.sum()),
        "n_negative_games": int((1 - y).sum()),
        "roc_auc": round(auc(y, scores), 4),
        "average_precision": round(float(average_precision_score(y, scores)), 4),
        "brier_score": round(float(brier_score_loss(y, scores)), 4),
    }


def train(frame: pd.DataFrame, source_sha: str, *, min_auc: float = 0.60,
          min_gain: float = 0.02) -> tuple[dict, dict]:
    data = from_chessfraud(frame)
    if data.empty or data["player_id"].nunique() < 25:
        raise ValueError("Too few eligible independent players")
    train_idx, valid_idx, test_idx = split_accounts(data)
    X = data.loc[:, list(FEATURE_NAMES)]
    y = data["label"].to_numpy(dtype=int)
    for name, idx in zip(("train", "validation", "test"), (train_idx, valid_idx, test_idx)):
        if len(np.unique(y[idx])) != 2 or min(int(y[idx].sum()), int((1-y[idx]).sum())) < 5:
            raise ValueError(f"{name} needs >=5 positive and >=5 negative player-games")
    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.3, max_iter=2000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X.iloc[train_idx], y[train_idx])
    model_proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    baseline = X.iloc[test_idx]["top1_rate"].to_numpy()
    baseline_auc = auc(y[test_idx], baseline)
    test = evaluate(y[test_idx], model_proba)
    val = evaluate(y[valid_idx], pipeline.predict_proba(X.iloc[valid_idx])[:, 1])
    passed = bool(test["roc_auc"] >= min_auc and test["roc_auc"] >= baseline_auc + min_gain)
    if not np.isfinite(baseline_auc):
        passed = False
    imputer = pipeline.named_steps["impute"]
    scaler = pipeline.named_steps["scale"]
    classifier = pipeline.named_steps["model"]
    # sklearn model is converted to language-neutral, deterministic JSON.
    output = {
        "schema": "openchess-chessfraud-game-logistic-v1",
        "status": "experimental_not_deployment_calibrated",
        "source": SOURCE,
        "source_sha256": source_sha,
        "model_features": list(FEATURE_NAMES),
        "impute_medians": imputer.statistics_.tolist(),
        "scale_means": scaler.mean_.tolist(),
        "scale_std": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "time_control": "300+0",
        "minimum_moves": 12,
        "heldout_roc_auc": test["roc_auc"],
        "baseline_roc_auc": round(float(baseline_auc), 4),
        "not_a_real_world_probability": True,
    }
    metrics = {
        "dataset_rows": len(frame),
        "eligible_player_games": len(data),
        "unique_players": int(data["player_id"].nunique()),
        "source_sha256": source_sha,
        "split_counts": {
            name: {"players": int(data.iloc[idx]["player_id"].nunique()), "games": len(idx)}
            for name, idx in zip(("train", "validation", "test"), (train_idx, valid_idx, test_idx))
        },
        "validation": val, "test": test,
        "baseline_test_auc": round(float(baseline_auc), 4),
        "passed_release_gate": passed,
        "gate": f"test_auc >= {min_auc} and gain >= {min_gain} against top1-only baseline",
        "warnings": [
            "Positive label means controlled tournament assistance, not public platform guilt",
            "Player-disjoint test remains small and only covers 300+0 blitz",
            "Probability is conditional on benchmark's artificial assisted-play prevalence",
            "Inference Stockfish version/depth may differ from tournament benchmark",
        ],
    }
    return output, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=SOURCE, help="Official release CSV or local snapshot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--min-gain", type=float, default=0.02)
    args = parser.parse_args()
    if str(args.csv).startswith("http"):
        import urllib.request
        with urllib.request.urlopen(args.csv, timeout=120) as stream:
            raw = stream.read(150_000_001)
    else:
        raw = Path(args.csv).read_bytes()
    if len(raw) > 150_000_000:
        raise ValueError("Dataset exceeds 150MB")
    source_sha = hashlib.sha256(raw).hexdigest()
    from io import BytesIO
    frame = pd.read_csv(BytesIO(raw), low_memory=False)
    model, metrics = train(frame, source_sha, min_auc=args.min_auc, min_gain=args.min_gain)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)
    if metrics["passed_release_gate"]:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
        print(f"PASS: exportable model written to {args.out}")
    else:
        print("NO SHIP: model did not beat baseline under the predetermined gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
