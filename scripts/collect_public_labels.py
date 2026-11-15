#!/usr/bin/env python3
"""Reproducible public account-status snapshot and game corpus collector.

No login/API key. Input is user-supplied PGNs or a UTF-8 CSV of public accounts.
This does not certify that a game involved cheating or that an open account is clean.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn

SCHEMA = "openchess-public-labels-v1"
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_-]{2,40}$")
SPLITS = ("train", "calibration", "test")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def account_key(platform: str, user: str) -> str:
    return f"{platform}:{user.casefold()}"


def split_for(key: str) -> str:
    """Persistent account split: does not depend on discovery order or dataset size."""
    h = int(hashlib.sha256(("openchess-v1:" + key).encode()).hexdigest()[:12], 16) / 16**12
    return "train" if h < 0.6 else ("calibration" if h < 0.8 else "test")


def validate_account(platform: str, username: str) -> tuple[str, str]:
    platform = platform.strip().lower()
    username = username.strip()
    if platform not in {"chesscom", "lichess"}:
        raise ValueError("platform must be chesscom or lichess")
    if not ACCOUNT_RE.fullmatch(username):
        raise ValueError("Invalid public username")
    return platform, username


def load_accounts(csv_path: Path | None, pg_paths: list[Path], platform: str | None,
                  limit: int) -> list[tuple[str, str]]:
    accounts: dict[str, tuple[str, str]] = {}
    if csv_path:
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                p, u = validate_account(row["platform"], row["player"])
                accounts[account_key(p, u)] = (p, u)
    for pgn_path in pg_paths:
        if not platform:
            raise ValueError("--platform is required with --seed-pgn")
        with pgn_path.open(encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                for color in ("White", "Black"):
                    user = game.headers.get(color, "")
                    try:
                        p, u = validate_account(platform, user)
                    except ValueError:
                        continue
                    if game.headers.get(color + "Title") == "BOT":
                        continue
                    accounts[account_key(p, u)] = (p, u)
    return [accounts[k] for k in sorted(accounts)[:limit]]


class PublicClient:
    def __init__(self, user_agent: str, pause: float = 1.0, retries: int = 3):
        if not user_agent.strip() or user_agent == "Python-urllib":
            raise ValueError("Provide a descriptive --user-agent, including contact info")
        self.agent = user_agent
        self.pause = max(0.0, pause)
        self.retries = retries

    def get(self, url: str, accept: str = "application/json") -> bytes:
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": self.agent, "Accept": accept
                })
                with urllib.request.urlopen(req, timeout=30) as response:
                    body = response.read(30_000_001)
                    if len(body) > 30_000_000:
                        raise ValueError("API response exceeds 30MB safety limit")
                time.sleep(self.pause)
                return body
            except urllib.error.HTTPError as e:
                if e.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise
                time.sleep(min(30, 2 ** attempt + self.pause))
            except (TimeoutError, urllib.error.URLError):
                if attempt == self.retries:
                    raise
                time.sleep(min(30, 2 ** attempt + self.pause))
        raise RuntimeError("unreachable")


def status_url(platform: str, username: str) -> str:
    safe = urllib.parse.quote(username, safe="")
    if platform == "chesscom":
        return f"https://api.chess.com/pub/player/{safe}"
    return f"https://lichess.org/api/user/{safe}"


def classify(platform: str, payload: dict) -> tuple[int, str]:
    """1 means public platform fair-play closure; 0 means unconfirmed (NOT clean)."""
    if platform == "chesscom":
        status = str(payload.get("status", "")).lower()
        if status == "closed:fair_play_violations":
            return 1, "platform_fair_play_closure"
        if status in {"premium", "basic", "staff", "closed"}:
            return 0, "unconfirmed_not_clean"
        raise ValueError(f"Unrecognized Chess.com status: {status!r}")
    if payload.get("tosViolation") is True:
        return 1, "platform_tos_violation_unspecified"
    if payload.get("tosViolation") is False or payload.get("disabled") is False:
        return 0, "unconfirmed_not_clean"
    if payload.get("disabled") is True:
        return 0, "disabled_reason_unknown"
    # A missing indicator must not be treated as evidence of a clean account.
    return 0, "unconfirmed_not_clean"


def export_games(client: PublicClient, platform: str, user: str, max_games: int) -> bytes:
    safe = urllib.parse.quote(user, safe="")
    if platform == "lichess":
        url = (f"https://lichess.org/api/games/user/{safe}?"
               + urllib.parse.urlencode({"max": max_games, "rated": "true",
                                         "perfType": "blitz,rapid",
                                         "clocks": "true", "opening": "false"}))
        return client.get(url, accept="application/x-chess-pgn")
    # Chess.com provides month-scoped archives; pick newest months and preserve PGNs.
    archives_data = json.loads(client.get(f"https://api.chess.com/pub/player/{safe}/games/archives"))
    archives = archives_data.get("archives", [])
    if not isinstance(archives, list):
        raise ValueError("Unexpected Chess.com archives response")
    pgn_parts = []
    for url in reversed(archives):
        if len(pgn_parts) >= 3:
            break
        # Verify host and path to avoid treating API-provided URLs as arbitrary fetch targets.
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != "api.chess.com" or not parsed.path.startswith(
                f"/pub/player/{safe}/games/"):
            raise ValueError("Unexpected archive URL")
        pgn_parts.append(client.get(url.rstrip("/") + "/pgn", accept="application/x-chess-pgn"))
    # Retain latest max_games per user rather than assuming monthly archive is small.
    assembled = b"\n".join(reversed(pgn_parts)).decode("utf-8", errors="replace")
    stream = io.StringIO(assembled)
    games = []
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        if g.headers.get("Variant", "Standard") != "Standard":
            continue
        if user.casefold() not in {g.headers.get("White", "").casefold(),
                                   g.headers.get("Black", "").casefold()}:
            continue
        if g.headers.get("TimeControl", "-") == "-":
            continue
        games.append(g)
    output = io.StringIO()
    for g in games[-max_games:]:
        print(g, file=output, end="\n\n")
    return output.getvalue().encode("utf-8")


def run_collect(args: argparse.Namespace) -> int:
    client = PublicClient(args.user_agent, args.pause)
    seed_bytes = None
    seed_paths = list(args.seed_pgn)
    if args.seed_user:
        if not args.platform:
            raise ValueError("--platform required with --seed-user")
        platform, user = validate_account(args.platform, args.seed_user)
        seed_bytes = export_games(client, platform, user, min(500, args.max_games * 4))
        if not seed_bytes.strip():
            raise ValueError("Seed user has no downloadable public games")
        with tempfile.NamedTemporaryFile(suffix=".pgn", delete=False) as tmp:
            tmp.write(seed_bytes)
            seed_paths.append(Path(tmp.name))
        try:
            accounts = load_accounts(args.accounts, seed_paths, args.platform, args.max_accounts)
        finally:
            seed_paths[-1].unlink(missing_ok=True)
    else:
        accounts = load_accounts(args.accounts, seed_paths, args.platform, args.max_accounts)
    if not accounts:
        raise ValueError("No valid public accounts found")
    root = args.out
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty; use a new snapshot directory")
    root.mkdir(parents=True, exist_ok=True)
    if seed_bytes is not None:
        (root / "seed.pgn").write_bytes(seed_bytes)
    games_dir = root / "pgn"
    if args.fetch_games:
        games_dir.mkdir()
    rows = []
    failures = []
    for platform, user in accounts:
        url = status_url(platform, user)
        key = account_key(platform, user)
        try:
            raw = client.get(url)
            response = json.loads(raw)
            label, quality = classify(platform, response)
            checked_at = datetime.now(timezone.utc).isoformat()
            row = {
                "platform": platform, "player": user, "account_key": key,
                "label": label, "label_quality": quality, "label_source": url,
                "label_checked_at": checked_at, "status_sha256": digest(raw),
                "split": split_for(key), "pgn_path": "", "pgn_sha256": "",
                "games_requested": args.max_games if args.fetch_games else 0,
            }
            if args.fetch_games:
                try:
                    raw_pgn = export_games(client, platform, user, args.max_games)
                    if not raw_pgn.strip():
                        raise ValueError("Empty game export")
                    filename = digest(key.encode())[:20] + ".pgn"
                    (games_dir / filename).write_bytes(raw_pgn)
                    row["pgn_path"] = "pgn/" + filename
                    row["pgn_sha256"] = digest(raw_pgn)
                except (ValueError, OSError, urllib.error.URLError) as e:
                    failures.append({"account_key": key, "stage": "games", "error": str(e)})
            rows.append(row)
        except (ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as e:
            failures.append({"account_key": key, "stage": "status", "error": str(e)})
        print(f"{len(rows) + len([f for f in failures if f['stage']=='status'])}/{len(accounts)} {key}", file=sys.stderr)
    manifest = {"schema": SCHEMA, "source": "platform_public_api",
                "label_definition": "1=public closure/status indicator; 0=unconfirmed, not clean",
                "snapshot_created_at": datetime.now(timezone.utc).isoformat(),
                "seed_pgn_sha256": digest(seed_bytes) if seed_bytes is not None else None,
                "accounts": rows, "failures": failures}
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} accounts; positive platform indicators: {sum(x['label'] for x in rows)}; "
          f"failures: {len(failures)} to {root}")
    return 0 if not failures else 2


def run_verify(args: argparse.Namespace) -> int:
    root = args.out
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown manifest schema")
    if manifest.get("seed_pgn_sha256"):
        if digest((root / "seed.pgn").read_bytes()) != manifest["seed_pgn_sha256"]:
            raise ValueError("Seed PGN checksum mismatch")
    seen = set()
    for row in manifest["accounts"]:
        key = account_key(*validate_account(row["platform"], row["player"]))
        if key != row["account_key"] or key in seen or row["split"] != split_for(key):
            raise ValueError(f"Duplicate/tampered account key or split: {key}")
        seen.add(key)
        if row["label"] not in (0, 1):
            raise ValueError("Invalid label")
        if row["pgn_path"]:
            file = (root / row["pgn_path"]).resolve()
            if not file.is_relative_to(root.resolve()) or digest(file.read_bytes()) != row["pgn_sha256"]:
                raise ValueError(f"PGN checksum mismatch: {key}")
    print(f"Verified {len(seen)} distinct accounts and local PGN SHA-256 checksums (not remote status).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="Snapshot live public account statuses and optionally games")
    c.add_argument("--accounts", type=Path, help="CSV columns: platform,player")
    c.add_argument("--seed-pgn", type=Path, action="append", default=[], help="Discover public accounts from PGN")
    c.add_argument("--platform", choices=("chesscom", "lichess"), help="Platform for seed accounts")
    c.add_argument("--seed-user", help="Fetch this user's public games as the seed (no seed PGN needed)")
    c.add_argument("--out", type=Path, required=True, help="NEW snapshot directory")
    c.add_argument("--user-agent", required=True, help="Descriptive agent name and contact details")
    c.add_argument("--max-accounts", type=int, default=200)
    c.add_argument("--fetch-games", action="store_true")
    c.add_argument("--max-games", type=int, default=50)
    c.add_argument("--pause", type=float, default=1.0)
    v = sub.add_parser("verify", help="Offline verify frozen dataset manifest and PGN checksums")
    v.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        if args.max_accounts < 1 or args.max_games < 1 or args.max_games > 500:
            raise ValueError("max-accounts and max-games must be positive; max-games <= 500")
        return run_collect(args)
    return run_verify(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
