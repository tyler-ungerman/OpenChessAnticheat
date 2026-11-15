#!/usr/bin/env python3
"""Build a control-cohort calibration file from PGNs of presumed-clean accounts.

The output is consumed by `backend/src/detectors/calibration.py` at
`data/baseline.json`. It is optional: without it the detectors use Elo fallbacks.

Control PGNs are expected as one account per file (`data/controls/<username>.pgn`).
Each file's stem is treated as the focal control account.

Examples:
    python scripts/build_baseline.py --controls data/controls --out data/baseline.json
    python scripts/build_baseline.py --controls data/controls --max-games-per-file 20 \
        --export-moves data/controls_moves.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from src.detectors import build_empirical_baselines, fit_game_sensitivity_s  # noqa: E402
from src.engine import EngineConfig, evaluate_games_df  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import parse_pgn_for_detection  # noqa: E402

DEFAULT_CONTROLS_DIR = REPO_ROOT / "data" / "controls"
DEFAULT_BASELINE_PATH = REPO_ROOT / "data" / "baseline.json"


def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def load_controls(
    controls_dir: Path,
    max_games_per_file: int,
    all_players: bool,
) -> tuple[pd.DataFrame, dict[int, str], list[Path]]:
    frames: list[pd.DataFrame] = []
    focal_by_game: dict[int, str] = {}
    files: list[Path] = []
    offset = 0

    for path in sorted(controls_dir.rglob("*.pgn")):
        raw = parse_pgn_for_detection(str(path))
        if raw.empty:
            print(f"[skip] {path}: no games")
            continue
        if max_games_per_file > 0:
            keep = sorted(raw["game_id"].unique())[:max_games_per_file]
            raw = raw[raw["game_id"].isin(keep)]
        raw = raw.copy()
        raw["game_id"] = raw["game_id"].astype(int) + offset
        offset = int(raw["game_id"].max())
        if not all_players:
            for game_id in raw["game_id"].unique():
                focal_by_game[int(game_id)] = path.stem
        frames.append(raw)
        files.append(path)
        print(f"[load] {path}: {raw['game_id'].nunique()} games")

    if not frames:
        return pd.DataFrame(), focal_by_game, files
    return pd.concat(frames, ignore_index=True), focal_by_game, files


def sensitivity_table(eval_df: pd.DataFrame, consistency_c: float) -> pd.DataFrame:
    valid = eval_df[eval_df["centipawn_loss"].notna() & (eval_df["legal_moves_count"] > 1)]
    rows = [
        {
            "game_id": game_id,
            "color": color,
            "choice_sensitivity_s": fit_game_sensitivity_s(
                group["centipawn_loss"].to_numpy(dtype=float), c_fixed=consistency_c
            ),
        }
        for (game_id, color), group in valid.groupby(["game_id", "color"], sort=False)
    ]
    return pd.DataFrame(rows)


def baseline_rows(features: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in features.iterrows():
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "date": _clean(row.get("date")),
                "color": row.get("color"),
                "username": row.get("player"),
                "time_control": row.get("time_control"),
                "rating": _clean(row.get("elo")),
                "capped_acpl": _clean(row.get("capped_acpl")),
                "top1_rate": _clean(row.get("top1_rate")),
                "critical_top1_rate": _clean(row.get("critical_top1_rate")),
                "choice_sensitivity_s": _clean(row.get("choice_sensitivity_s")),
                "cv_move_time": _clean(row.get("cv_move_time")),
                "label": 0,
                "label_source": "control_account",
            }
        )
    return rows


def export_moves(
    path: Path,
    eval_df: pd.DataFrame,
    focal_by_game: dict[int, str],
    all_players: bool,
) -> int:
    if all_players:
        focal_flags = pd.Series(True, index=eval_df.index)
    else:
        expected = eval_df["game_id"].map(focal_by_game)
        focal_flags = eval_df["player"].astype(str).str.casefold() == expected.astype(str).str.casefold()

    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row, is_focal in zip(eval_df.itertuples(index=False), focal_flags):
            if _clean(row.eval_before) is None or _clean(row.centipawn_loss) is None:
                continue
            record = {
                "ply": int(row.ply),
                "side": "w" if row.color == "White" else "b",
                "fen": row.fen_before,
                "uci": row.move_uci,
                "san": row.move_san,
                "best_uci": row.best_move,
                "n_legal": _clean(row.legal_moves_count),
                "best_cp": _clean(row.eval_before),
                "played_cp": _clean(row.eval_after),
                "cp_loss": _clean(row.centipawn_loss),
                "win_before": round(float(row.win_prob_before) * 100, 2)
                if pd.notna(row.win_prob_before)
                else None,
                "win_after": round(float(row.win_prob_after) * 100, 2)
                if pd.notna(row.win_prob_after)
                else None,
                "rank": _clean(row.move_rank),
                "gap12_cp": _clean(row.eval_gap),
                "time_spent": _clean(row.time_spent),
                "game_id": f"ctrl_{int(row.game_id)}",
                "time_control": row.time_control,
                "account": row.player,
                "is_focal": bool(is_focal),
                "mover_rating": _clean(row.elo),
                "account_label": 0,
                "date": row.date,
            }
            handle.write(json.dumps(record) + "\n")
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build data/baseline.json from control-account PGNs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS_DIR, help="Directory of control PGNs")
    parser.add_argument("--out", type=Path, default=DEFAULT_BASELINE_PATH, help="Baseline JSON output path")
    parser.add_argument("--depth", type=int, default=14, help="Stockfish depth (default: 14)")
    parser.add_argument("--multipv", type=int, default=3, help="MultiPV lines (default: 3)")
    parser.add_argument("--workers", type=int, default=4, help="Stockfish worker processes (default: 4)")
    parser.add_argument("--max-games-per-file", type=int, default=0, help="Limit games per control file")
    parser.add_argument("--all-players", action="store_true", help="Use both sides of each game, not just the file's account")
    parser.add_argument("--consistency-c", type=float, default=0.85, help="Weibull shape used for sensitivity")
    parser.add_argument("--no-cache", action="store_true", help="Ignore the on-disk evaluation cache")
    parser.add_argument("--export-moves", type=Path, help="Also export move-level JSONL.GZ for backend/benchmark.py")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.controls.is_dir():
        print(f"Controls directory not found: {args.controls}", file=sys.stderr)
        return 2

    raw_df, focal_by_game, files = load_controls(
        args.controls, args.max_games_per_file, args.all_players
    )
    if raw_df.empty:
        print("No control games found. Fetch some with scripts/fetch_games.py.", file=sys.stderr)
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
    features = build_features(eval_df)
    if features.empty:
        print("No player-game rows had enough evaluated moves.", file=sys.stderr)
        return 1

    if "choice_sensitivity_s" in features.columns:
        features = features.drop(columns=["choice_sensitivity_s"])
    sensitivity = sensitivity_table(eval_df, args.consistency_c)
    if not sensitivity.empty:
        features = features.merge(sensitivity, on=["game_id", "color"], how="left")

    if not args.all_players:
        expected = features["game_id"].map(focal_by_game)
        found = features["player"].astype(str).str.casefold()
        features = features[found == expected.astype(str).str.casefold()].copy()
        if features.empty:
            print(
                "No focal control rows matched their file names. "
                "Use --all-players or ensure files are named <username>.pgn.",
                file=sys.stderr,
            )
            return 1
        missing = sorted(
            {name for name in focal_by_game.values() if name.casefold() not in set(found)}
        )
        if missing:
            print(f"[warn] no rows for file account(s): {', '.join(missing)}", file=sys.stderr)

    rows = baseline_rows(features)
    buckets = build_empirical_baselines(features)
    accounts = (
        sorted(features["player"].dropna().astype(str).unique())
        if args.all_players
        else sorted(set(focal_by_game.values()))
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_control_accounts": len(accounts),
        "n_clean_games": len(rows),
        "consistency_c": args.consistency_c,
        "pooled_fit": {"consistency_c": args.consistency_c},
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(args.out)

    print(
        f"[baseline] {len(files)} files | {len(accounts)} accounts | {len(rows)} player-game rows | "
        f"{len(buckets)} (time control, Elo) buckets -> {args.out}"
    )
    if not buckets:
        print(
            "[baseline] WARNING: cohort buckets need ~15 player-games per (time control, 100-Elo bucket). "
            "The pipeline will use Elo fallbacks until then.",
            file=sys.stderr,
        )

    if args.export_moves:
        written = export_moves(args.export_moves, eval_df, focal_by_game, args.all_players)
        print(f"[moves] {written} plies -> {args.export_moves}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
