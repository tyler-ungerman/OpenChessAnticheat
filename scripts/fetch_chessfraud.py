#!/usr/bin/env python3
"""Download and verify the independently annotated ChessFraud tournament benchmark.

Controlled assistance labels are distinct from public fair-play closure labels.
The data are third-party assets; never commit them or merge labels across cohorts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://github.com/artem-lepin-ml/chess-fraud/releases/download/v1.0-data/"
FILES = ("chess_fraud_tournament.csv", "tournament.csv")
SCHEMA = "openchess-chessfraud-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, max_bytes: int = 250_000_000) -> None:
    """Download atomically with an explicit byte cap and timeouts."""
    req = urllib.request.Request(url, headers={"User-Agent": "OpenChessAnticheat-research/1.0"})
    tmp_name = None
    try:
        with urllib.request.urlopen(req, timeout=45) as src:
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                tmp_name = tmp.name
                length = 0
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    length += len(chunk)
                    if length > max_bytes:
                        raise ValueError("Benchmark file exceeds 250MB safety limit")
                    tmp.write(chunk)
        os.replace(tmp_name, dest)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def describe_csv(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
        reader = csv.DictReader(stream)
        headers = list(reader.fieldnames or [])
        if not headers:
            raise ValueError(f"Invalid empty CSV: {path}")
        n = 0
        players = set()
        games = set()
        assist_positive = 0
        for row in reader:
            n += 1
            if row.get("player_id"):
                players.add(row["player_id"])
            if row.get("game_id"):
                games.add(row["game_id"])
            if str(row.get("is_cheating_player_game", "")).strip().lower() in {"true", "1"}:
                assist_positive += 1
    return {"rows": n, "columns": headers, "distinct_player_ids": len(players),
            "distinct_game_ids": len(games),
            "move_rows_from_assisted_player_games": assist_positive}


def create(root: Path) -> dict:
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory must be empty; do not overwrite frozen data")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for filename in FILES:
        url = BASE + filename
        destination = root / filename
        download(url, destination)
        entries.append({"file": filename, "source": url,
                        "sha256": sha256(destination),
                        "bytes": destination.stat().st_size,
                        "summary": describe_csv(destination)})
    manifest = {
        "schema": SCHEMA,
        "benchmark": "ChessFraud (controlled cheating-assistance tournament)",
        "release": "v1.0-data",
        "source_repository": "https://github.com/artem-lepin-ml/chess-fraud",
        "dataset_card": "https://huggingface.co/datasets/artemlepin/chess-fraud",
        "license_note": "Verify upstream per-file license and attribution requirements before redistribution",
        "entries": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unexpected benchmark manifest schema")
    seen = set()
    for entry in manifest["entries"]:
        if entry["file"] not in FILES or entry["file"] in seen:
            raise ValueError("Unexpected/duplicate benchmark file")
        seen.add(entry["file"])
        path = root / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}")
        if describe_csv(path) != entry["summary"]:
            raise ValueError(f"CSV content/row count mismatch: {path}")
    if seen != set(FILES):
        raise ValueError("Missing benchmark file")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify"))
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/chessfraud-v1"))
    args = parser.parse_args()
    result = create(args.out) if args.command == "fetch" else verify(args.out)
    for entry in result["entries"]:
        print(f"{entry['file']}: {entry['summary']['rows']} rows, SHA-256 {entry['sha256']}")
    print("Source snapshot validated; performance of OpenChessAnticheat NOT measured.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
