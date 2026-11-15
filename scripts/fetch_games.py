#!/usr/bin/env python3
"""Fetch public Chess.com games into PGN files.

Standard chess only; bot/variant games are excluded or filtered later by the pipeline.

Examples:
    python scripts/fetch_games.py --user alice --out datasets/games.pgn
    python scripts/fetch_games.py --users-file controls.txt --out-dir datasets/controls \
        --months 3 --max-games 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "https://api.chess.com/pub"
DEFAULT_GAMES_PATH = REPO_ROOT / "datasets" / "games.pgn"
DEFAULT_CONTROLS_DIR = REPO_ROOT / "datasets" / "controls"
USER_AGENT = "chess-detection-data-fetch/1.0"


def fetch_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def fetch_games(
    api_base: str,
    username: str,
    months: int,
    time_class: str | None,
    max_games: int,
    delay: float,
    timeout: float,
) -> list[str]:
    """Return up to `max_games` recent standard-chess PGNs, oldest first."""
    endpoint = f"{api_base.rstrip('/')}/player/{urllib.parse.quote(username)}/games/archives"
    archives = fetch_json(endpoint, timeout).get("archives") or []
    if months > 0:
        archives = archives[-months:]

    selected: list[str] = []
    for archive_url in reversed(archives):
        payload = fetch_json(archive_url, timeout)
        games = sorted(payload.get("games") or [], key=lambda game: game.get("end_time", 0), reverse=True)
        for game in games:
            if game.get("rules") != "chess":
                continue
            if time_class and game.get("time_class") != time_class:
                continue
            pgn = game.get("pgn")
            if not pgn:
                continue
            selected.append(pgn)
            if max_games and len(selected) >= max_games:
                selected.reverse()
                return selected
        time.sleep(delay)

    selected.reverse()
    return selected


def write_pgn(path: Path, games: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(game.strip() for game in games) + "\n", encoding="utf-8")


def safe_name(username: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", username)


def read_users(path: Path) -> list[str]:
    users = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            users.append(value)
    return list(dict.fromkeys(users))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch public Chess.com games into PGN files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user", help="Single Chess.com username")
    target.add_argument("--users-file", type=Path, help="File with one username per line")
    parser.add_argument("--out", type=Path, help="Output PGN for --user (default: datasets/games.pgn)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory for --users-file; one <username>.pgn per account (default: datasets/controls)",
    )
    parser.add_argument("--months", type=int, default=3, help="Recent monthly archives to scan; 0 = all")
    parser.add_argument("--max-games", type=int, default=100, help="Games per user; 0 = all")
    parser.add_argument(
        "--time-class",
        choices=("bullet", "blitz", "rapid", "daily"),
        help="Only fetch this time class",
    )
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between archive requests")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Chess.com API base (for testing)")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    users = [args.user] if args.user else read_users(args.users_file)
    if not users:
        print("No usernames provided.", file=sys.stderr)
        return 2

    total_games = 0
    failures = 0
    for username in users:
        destination = (
            (args.out or DEFAULT_GAMES_PATH)
            if args.user
            else (args.out_dir or DEFAULT_CONTROLS_DIR) / f"{safe_name(username)}.pgn"
        )
        try:
            games = fetch_games(
                api_base=args.api_base,
                username=username,
                months=args.months,
                time_class=args.time_class,
                max_games=args.max_games,
                delay=args.delay,
                timeout=args.timeout,
            )
        except urllib.error.HTTPError as exc:
            message = "not found" if exc.code == 404 else f"HTTP {exc.code}"
            print(f"[skip] {username}: {message}", file=sys.stderr)
            failures += 1
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            print(f"[skip] {username}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if not games:
            print(f"[skip] {username}: no standard-chess games in range", file=sys.stderr)
            failures += 1
            continue

        write_pgn(destination, games)
        total_games += len(games)
        print(f"[ok] {username}: {len(games)} games -> {destination}")

    print(f"[done] {total_games} games from {len(users) - failures}/{len(users)} users")
    return 0 if total_games else 1


if __name__ == "__main__":
    raise SystemExit(main())
