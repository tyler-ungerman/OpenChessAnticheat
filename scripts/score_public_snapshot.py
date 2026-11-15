#!/usr/bin/env python3
"""Score an immutable public-account snapshot with the existing Stockfish pipeline.

Outputs one account-level row per (platform, player, time_control). Snapshots and
engine parameters are recorded; scores are research features, not cheat verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform as system_platform
import subprocess
import shutil
import sys
from pathlib import Path

import chess.pgn


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_windows(pgn: Path, account: str, n_games: int) -> dict[str, bytes]:
    """Freeze exactly n latest eligible player games per time-control category."""
    from backend.src.features.engineering import _classify_time_control
    from backend.src.ingestion.pgn import is_bot_game
    groups: dict[str, list[tuple[str, chess.pgn.Game]]] = {}
    with pgn.open(encoding="utf-8", errors="replace") as source:
        while True:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            if account.casefold() not in {
                game.headers.get("White", "").casefold(),
                game.headers.get("Black", "").casefold(),
            } or is_bot_game(game.headers):
                continue
            tc = _classify_time_control(game.headers.get("TimeControl"))
            if tc not in {"blitz", "rapid", "bullet"}:
                continue
            date = game.headers.get("UTCDate", game.headers.get("Date", ""))
            clock = game.headers.get("UTCTime", "")
            key = f"{date}|{clock}|{game.headers.get('Site', '')}"
            groups.setdefault(tc, []).append((key, game))
    result = {}
    for tc, games in groups.items():
        if len(games) < n_games:
            continue
        output = io.StringIO()
        for _, game in sorted(games, key=lambda x: x[0])[-n_games:]:
            print(game, file=output, end="\\n\\n")
        result[tc] = output.getvalue().encode("utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--games-per-time-control", type=int, default=30,
                        help="Freeze exactly this many most-recent games per account and time control")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stockfish", type=str, help="Path to Stockfish binary")
    args = parser.parse_args()
    if args.depth < 1 or args.depth > 30 or args.workers < 1 or args.games_per_time_control < 5:
        parser.error("Invalid depth/workers")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
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
    engine_binary = args.stockfish or os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "stockfish"
    engine_resolved = shutil.which(engine_binary)
    if engine_resolved is None:
        raise ValueError("Stockfish is required; install it or specify --stockfish /path/to/stockfish")
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
        "engine_binary": str(Path(engine_resolved).resolve()),
        "python_executable": sys.executable,
        "dependency_versions": subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True,
            text=True, check=False, timeout=30
        ).stdout.splitlines(),
        "engine_version_output": engine_fingerprint,
        "engine_binary_sha256": (
            sha256_file(Path(engine_resolved))
        ),
        "depth": args.depth, "workers": args.workers,
        "multipv": 3, "shallow_depth": 6, "use_cache": True,
        "games_per_time_control": args.games_per_time_control,
        "evaluation_code_sha256": sha256_file(root / "backend/src/engine/evaluation.py"),
        "scoring_code_sha256": sha256_file(root / "backend/src/detectors/statistical_z.py"),
        "feature_code_sha256": sha256_file(root / "backend/src/features/engineering.py"),
        "rows": [],
        "failures": [],
    }
    env = os.environ.copy()
    env["STOCKFISH_PATH"] = str(Path(engine_resolved).resolve())
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
        windows = normalized_windows(pgn, row["player"], args.games_per_time_control)
        if not windows:
            lock.setdefault("skipped_short_history", []).append(row["account_key"])
            continue
        for tc, window_pgn in sorted(windows.items()):
            digest = hashlib.sha256((row["account_key"] + ":" + tc).encode()).hexdigest()[:20]
            subset_pgn = reports_dir / (digest + ".pgn")
            subset_pgn.write_bytes(window_pgn)
            destination = reports_dir / (digest + ".json")
            cmd = [sys.executable, str(root / "examples/analyze_pgn.py"), str(subset_pgn),
                   "--player", row["player"], "--depth", str(args.depth),
                   "--workers", str(args.workers), "--json-out", str(destination)]
            run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
            if run.returncode != 0 or not destination.is_file():
                lock["failures"].append({"account": row["account_key"], "time_control": tc,
                                         "reason": "scoring_failed", "stderr_tail": run.stderr[-500:]})
                continue
            report = json.loads(destination.read_text(encoding="utf-8"))
            summaries = [record for record in report.get("player_summary", [])
                         if record["time_control"] == tc
                         and int(record["games"]) == args.games_per_time_control]
            if len(summaries) != 1:
                lock["failures"].append({"account": row["account_key"], "time_control": tc,
                                         "reason": "unexpected_window_summary"})
                continue
            record = summaries[0]
            lock["rows"].append({
                "platform": row["platform"], "player": row["player"],
                "account_key": row["account_key"], "split": row["split"],
                "label": row["label"], "label_quality": row["label_quality"],
                "time_control": tc, "games": record["games"],
                "mean_elo": record.get("mean_elo"),
                "max_sprt_llr": record["max_sprt_llr"],
                "max_single_z": record["max_single_z"],
                "max_cusum": record["max_cusum"],
                "report_sha256": sha256_file(destination),
                "window_pgn_sha256": sha256_file(subset_pgn),
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
