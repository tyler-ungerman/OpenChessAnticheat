#!/usr/bin/env python3
"""Run the chess fair-play pipeline on a PGN file, without the API server.

Examples:
    python examples/analyze_pgn.py examples/sample_game.pgn
    python examples/analyze_pgn.py my_games.pgn --player alice --depth 12 --json-out results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from src.detectors import aggregate_player_scores, score_games  # noqa: E402
from src.engine import EngineConfig, evaluate_games_df  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import parse_pgn_for_detection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pgn", type=Path, help="PGN file to analyze")
    parser.add_argument(
        "--player",
        help="Only report this player (case-insensitive); scoring still uses the full file.",
    )
    parser.add_argument("--depth", type=int, default=14, help="Stockfish depth (default: 14)")
    parser.add_argument("--multipv", type=int, default=3, help="MultiPV lines (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker processes (default: 4)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the on-disk evaluation cache")
    parser.add_argument("--json-out", type=Path, help="Write the reported rows to a JSON file")
    parser.add_argument("--top", type=int, default=10, help="Rows to print (default: 10)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pgn.is_file():
        print(f"PGN not found: {args.pgn}", file=sys.stderr)
        return 2

    raw_df = parse_pgn_for_detection(str(args.pgn))
    if raw_df.empty:
        print("No games found in the PGN.", file=sys.stderr)
        return 1

    eval_df = evaluate_games_df(
        raw_df,
        config=EngineConfig(
            depth=args.depth,
            multipv=args.multipv,
            workers=args.workers,
            use_cache=not args.no_cache,
        ),
    )
    features_df = build_features(eval_df)
    if features_df.empty:
        print("No games had enough evaluated moves (minimum 5 per player).", file=sys.stderr)
        return 1

    scored_df = score_games(features_df, eval_df=eval_df)
    players_df = aggregate_player_scores(scored_df, features_df=features_df)

    if args.player:
        target = args.player.casefold()
        players_view = players_df[players_df["player"].str.casefold() == target]
        scored_view = scored_df[scored_df["player"].str.casefold() == target]
        if players_view.empty:
            print(f"Player {args.player!r} not found in this PGN.", file=sys.stderr)
            return 1
    else:
        players_view = players_df
        scored_view = scored_df

    print(f"\nGames: {raw_df['game_id'].nunique()} | player-game rows: {len(features_df)}")

    print("\nPlayer summary:")
    print(players_view.head(args.top).to_string(index=False))

    flagged = scored_view[scored_view["status"].isin(["exploratory_outlier", "exploratory_review"])]
    print("\nExploratory outlier / review games (NOT cheat verdicts):")
    print(flagged.head(args.top).to_string(index=False) if not flagged.empty else "none")

    if args.json_out:
        def records(frame):
            return frame.where(frame.notna(), None).to_dict(orient="records")

        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "player_summary": records(players_view),
                    "game_scores": records(scored_view),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
