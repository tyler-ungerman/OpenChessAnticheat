#!/usr/bin/env python3
"""Score an immutable public-account snapshot with the existing Stockfish pipeline.

Outputs one account-level row per (platform, player, time_control). Snapshots and
engine parameters are recorded; scores are research features, not cheat verdicts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform as system_platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stockfish", type=str, help="Path to Stockfish binary")
    args = parser.parse_args()
    if args.depth < 1 or args.depth > 30 or args.workers < 1:
        parser.error("Invalid depth/workers")
    root = Path(__file__).resolve().parents[1]
    snapshot = args.snapshot.resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "openchess-public-labels-v1":
        raise ValueError("Unsupported or missing snapshot manifest")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Output must be a new, empty directory")
    args.out.mkdir(parents=True, exist_ok=True)
    reports_dir = args.out / "reports"
    reports_dir.mkdir()
    engine_binary = args.stockfish or os.environ.get("STOCKFISH_PATH") or "stockfish"
    try:
        engine_result = subprocess.run([engine_binary, "--help"], capture_output=True,
                                       text=True, timeout=15, check=False)
        engine_fingerprint = (engine_result.stdout + engine_result.stderr)[:1000]
    except (OSError, subprocess.TimeoutExpired):
        engine_fingerprint = "unavailable"
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    lock = {
        "snapshot_sha256": sha256_file(manifest_path),
        "source_manifest": str(manifest_path),
        "source_git_commit": git.stdout.strip() if git.returncode == 0 else "unknown",
        "python_version": sys.version,
        "platform": system_platform.platform(),
        "engine_binary": engine_binary,
        "engine_version_output": engine_fingerprint,
        "engine_binary_sha256": (
            sha256_file(Path(engine_binary)) if Path(engine_binary).is_file() else None
        ),
        "depth": args.depth, "workers": args.workers,
        "evaluation_code_sha256": sha256_file(root / "backend/src/engine/evaluation.py"),
        "scoring_code_sha256": sha256_file(root / "backend/src/detectors/statistical_z.py"),
        "feature_code_sha256": sha256_file(root / "backend/src/features/engineering.py"),
        "rows": [],
        "failures": [],
    }
    env = os.environ.copy()
    if args.stockfish:
        env["STOCKFISH_PATH"] = args.stockfish
    for row in manifest["accounts"]:
        relative = row.get("pgn_path")
        if not relative:
            continue
        pgn = (snapshot / relative).resolve()
        if not pgn.is_relative_to(snapshot) or not pgn.is_file():
            lock["failures"].append({"account": row["account_key"], "reason": "missing_pgn"})
            continue
        if sha256_file(pgn) != row["pgn_sha256"]:
            lock["failures"].append({"account": row["account_key"], "reason": "pgn_hash_mismatch"})
            continue
        destination = reports_dir / (hashlib.sha256(row["account_key"].encode()).hexdigest()[:20] + ".json")
        cmd = [sys.executable, str(root / "examples/analyze_pgn.py"), str(pgn),
               "--player", row["player"], "--depth", str(args.depth),
               "--workers", str(args.workers), "--json-out", str(destination)]
        run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        if run.returncode != 0 or not destination.is_file():
            lock["failures"].append({"account": row["account_key"], "reason": "scoring_failed",
                                     "stderr_tail": run.stderr[-500:]})
            continue
        report = json.loads(destination.read_text(encoding="utf-8"))
        for record in report.get("player_summary", []):
            lock["rows"].append({
                "platform": row["platform"], "player": row["player"],
                "account_key": row["account_key"], "split": row["split"],
                "label": row["label"], "label_quality": row["label_quality"],
                "time_control": record["time_control"], "games": record["games"],
                "mean_elo": record.get("mean_elo"),
                "max_sprt_llr": record["max_sprt_llr"],
                "max_single_z": record["max_single_z"],
                "max_cusum": record["max_cusum"],
                "report_sha256": sha256_file(destination),
            })
        print(f"Scored {row['account_key']}", file=sys.stderr)
    lock["rows"].sort(key=lambda r: (r["account_key"], r["time_control"]))
    (args.out / "score_manifest.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not lock["rows"] or lock["failures"]:
        print(f"Scored {len(lock['rows'])} player/time-control rows; "
              f"{len(lock['failures'])} failures. Output incomplete.", file=sys.stderr)
        return 2
    print(f"Scored {len(lock['rows'])} rows to {args.out / 'score_manifest.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
