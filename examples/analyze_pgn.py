#!/usr/bin/env python3
"""Run the chess fair-play pipeline on a PGN file, without the API server.

Examples:
    python examples/analyze_pgn.py examples/sample_game.pgn
    python examples/analyze_pgn.py data/personal/games.pgn --player alice
    python examples/analyze_pgn.py data/personal/games.pgn --json-out results.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from src.detectors import (  # noqa: E402
    aggregate_player_scores,
    cohort_time_controls,
    load_calibration,
    score_games,
)
from src.detectors.reference_compare import compare_accounts, read_proxy_reference  # noqa: E402
from src.engine import EngineConfig, evaluate_games_df  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingestion import parse_pgn_for_detection  # noqa: E402

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"

_VERDICT_LABELS = {
    "manual_review_recommended": "Manual review recommended",
    "statistical_outlier": "Statistical outlier",
    "no_outlier_detected": "No outlier detected",
    "not_calibrated": "Not calibrated",
}
_VERDICT_COLORS = {
    "manual_review_recommended": _RED,
    "statistical_outlier": _YELLOW,
    "no_outlier_detected": _GREEN,
    "not_calibrated": _DIM,
}
_STATUS_LABELS = {
    "exploratory_outlier": "Outlier",
    "exploratory_review": "Review",
}
_STATUS_COLORS = {
    "exploratory_outlier": _YELLOW,
    "exploratory_review": _CYAN,
}
_PRIORITY = {
    "manual_review_recommended": 0,
    "statistical_outlier": 1,
    "no_outlier_detected": 2,
}


def _paint(text: str, code: str | None, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled and code else text


def _width() -> int:
    return max(80, min(108, shutil.get_terminal_size((100, 24)).columns))


def _short(value: object, limit: int) -> str:
    text = "—" if value is None or value == "" else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _count(value: int, singular: str, plural: str | None = None) -> str:
    return f"{value} {singular if value == 1 else (plural or singular + 's')}"


def _num(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.{digits}f}"


def _verdict_label(value: object) -> str:
    raw = str(value or "unknown")
    return _VERDICT_LABELS.get(raw, raw.replace("_", " ").capitalize())


def _verdict_color(value: object) -> str:
    return _VERDICT_COLORS.get(str(value or ""), _DIM)


def _status_label(value: object) -> str:
    raw = str(value or "unknown")
    return _STATUS_LABELS.get(raw, raw.replace("_", " ").capitalize())


def _status_color(value: object) -> str:
    return _STATUS_COLORS.get(str(value or ""), _DIM)


def _table(
    headers: list[str],
    rows: list[list[str]],
    *,
    right: set[int] | None = None,
    status_col: int | None = None,
    row_colors: list[str] | None = None,
    use_color: bool = False,
) -> None:
    right = right or set()
    widths = [len(header) for header in headers]
    for cells in rows:
        for index, cell in enumerate(cells):
            widths[index] = max(widths[index], len(cell))

    header = [
        cell.rjust(width) if index in right else cell.ljust(width)
        for index, (cell, width) in enumerate(zip(headers, widths))
    ]
    print("  " + "  ".join(header))
    print("  " + "  ".join("─" * width for width in widths))

    for row_index, cells in enumerate(rows):
        parts = []
        for index, (cell, width) in enumerate(zip(cells, widths)):
            text = cell.rjust(width) if index in right else cell.ljust(width)
            if index == status_col and row_colors:
                text = _paint(text, row_colors[row_index], use_color)
            parts.append(text)
        print("  " + "  ".join(parts).rstrip())


def _print_header(args, raw_df, features_df, players_df, calibration, use_color: bool) -> None:
    width = _width()
    display_path = args.pgn
    if display_path.is_absolute():
        try:
            display_path = display_path.resolve().relative_to(BACKEND_DIR.parent)
        except ValueError:
            pass
    controls = int(calibration.get("n_clean_games") or 0)
    covered = sorted(cohort_time_controls(calibration))
    analyzed = set()
    if "time_control" in features_df.columns:
        analyzed = {str(value) for value in features_df["time_control"].dropna()}
    uncovered = sorted(analyzed - set(covered))
    if controls and covered:
        status = f"local cohort baseline ({controls} control player-games; {', '.join(covered)})"
        if uncovered:
            status += f"; Elo fallback for {', '.join(uncovered)}"
    else:
        status = "no local baseline"

    print()
    print(_paint("Chess fair-play screening", _BOLD, use_color))
    print(_paint("─" * width, _DIM, use_color))
    print(f"  {'File':<12}{_short(display_path, width - 14)}")
    print(
        f"  {'Games':<12}{raw_df['game_id'].nunique()}"
        f"   {'Player-game rows':<18}{len(features_df)}"
        f"   {'Players':<8}{players_df['player'].nunique()}"
    )
    print(f"  {'Calibration':<12}{status}")
    print()


def _print_summary(players_df, use_color: bool) -> None:
    values = players_df["account_verdict"].value_counts()
    manual = int(values.get("manual_review_recommended", 0))
    outliers = int(values.get("statistical_outlier", 0))
    none = int(values.get("no_outlier_detected", 0))
    print(
        _paint("Screening (player/TC rows):", _BOLD, use_color)
        + f" {_count(manual, 'manual-review recommendation')}"
        + f" · {_count(outliers, 'statistical outlier')}"
        + f" · {none} no outlier detected"
    )
    print()


def _print_player_cards(players_view, use_color: bool) -> None:
    width = _width()
    for row in players_view.to_dict(orient="records"):
        verdict = row.get("account_verdict")
        print(
            f"{_paint(str(row['player']), _BOLD, use_color)}"
            f"  {_paint('·', _DIM, use_color)}  {row['time_control']}"
            f"  {_paint('·', _DIM, use_color)}  "
            f"{_paint(_verdict_label(verdict), _verdict_color(verdict), use_color)}"
        )
        print(_paint("  " + "─" * width, _DIM, use_color))
        print(f"  {'Screening':<16}{_paint(_verdict_label(verdict), _verdict_color(verdict), use_color)}")
        print(
            f"  {'Games':<16}{int(row['games'])}"
            f"   {'Capped ACPL':<12}{_num(row.get('mean_capped_acpl'), 1)}"
            f"   {'Mean Elo':<8} {_num(row.get('mean_elo'), 0)}"
        )
        print(f"  {'Raw ACPL':<16}{_num(row.get('mean_acpl'), 1)}"
              f"   {'Median game':<14}{_num(row.get('median_game_acpl'), 1)}"
              f"   {'Top move rate':<14}{_num(100 * row['top1_rate'], 1) + '%' if row.get('top1_rate') is not None else '—'}")
        if row.get("mean_acpl") is not None and row.get("mean_capped_acpl") is not None and row["mean_acpl"] > row["mean_capped_acpl"] * 1.5:
            print(_paint("  Data quality    Raw ACPL is inflated by very large move losses;"
                         " check mate-score swings and use capped ACPL for comparisons.", _YELLOW, use_color))
        print(
            f"  {'Max signal':<16}SPRT {_num(row.get('max_sprt_llr'))}"
            f"   CUSUM {_num(row.get('max_cusum'))}"
            f"   Z {_num(row.get('max_single_z'))}"
        )
        print(
            f"  {'Outlier games':<16}{int(row.get('flagged_games') or 0)}"
            f" / {int(row['games'])} ({_num(row.get('flagged_pct'), 1)}%)"
        )
        null_value = row.get("null_p_value")
        null_text = "not configured" if null_value is None else f"p = {_num(null_value, 4)}"
        print(f"  {'Empirical null':<16}{null_text}")
        if row.get("proxy_reference_status") == "compared_unconfirmed_proxy":
            print(f"  {'Proxy percentile':<16}{100 * (1 - row['proxy_tail_p_value']):.1f} / 100"
                  f" (n={int(row['proxy_reference_accounts'])} unconfirmed accounts)")
        elif row.get("proxy_reference_status"):
            print(f"  {'Proxy reference':<16}{row['proxy_reference_status']}")
        if int(row.get("flagged_games") or 0) == 0 and verdict in {
            "manual_review_recommended",
            "statistical_outlier",
        }:
            print(
                _paint(
                    "  Interpretation  The account tier comes ONLY from accumulated"
                    " heuristic scores; no individual game crossed a review tier.",
                    _DIM,
                    use_color,
                )
            )
        print()


def _ranked(players_df):
    ranked = players_df.copy()
    ranked["_priority"] = ranked["account_verdict"].map(_PRIORITY).fillna(3)
    ranked = ranked.sort_values(
        ["_priority", "games", "max_sprt_llr"],
        ascending=[True, False, False],
    )
    return ranked.drop(columns=["_priority"])


def _print_players_table(ranked, top: int, use_color: bool) -> None:
    shown = ranked.head(top)
    player_width = max(12, min(18, _width() - 66))
    rows = []
    colors = []
    for row in shown.to_dict(orient="records"):
        rows.append(
            [
                _short(row["player"], player_width),
                _short(row["time_control"], 7),
                str(int(row["games"])),
                _num(row.get("mean_acpl"), 1),
                _num(row.get("max_sprt_llr")),
                f"{int(row.get('flagged_games') or 0)} ({_num(row.get('flagged_pct'), 0)}%)",
                _short(_verdict_label(row.get("account_verdict")), 24),
            ]
        )
        colors.append(_verdict_color(row.get("account_verdict")))

    print(_paint("Players ranked by heuristic priority", _BOLD, use_color))
    _table(
        ["Player", "TC", "Games", "ACPL", "Max SPRT", "Outliers", "Screening"],
        rows,
        right={2, 3, 4},
        status_col=6,
        row_colors=colors,
        use_color=use_color,
    )
    remaining = len(ranked) - len(shown)
    if remaining > 0:
        print(
            _paint(
                f"  {remaining} more player/time-control rows not shown. "
                f"Use --top {len(ranked)} or --verbose for all.",
                _DIM,
                use_color,
            )
        )
    if (
        (ranked["account_verdict"] != "no_outlier_detected").any()
        and (ranked["flagged_games"].fillna(0) == 0).any()
    ):
        print(
            _paint(
                "  Account tiers can be cumulative: a statistical outlier may have "
                "zero individually flagged games.",
                _DIM,
                use_color,
            )
        )
    print()


def _print_games_table(scored_view, top: int, use_color: bool) -> None:
    games = scored_view[scored_view["status"].isin(["exploratory_outlier", "exploratory_review"])]
    games = games.sort_values("composite_z", ascending=False)
    print(_paint("Games crossing the exploratory tiers", _BOLD, use_color))
    if games.empty:
        print(_paint("  No individual games crossed the exploratory tiers.", _DIM, use_color))
        print()
        return

    shown = games.head(top)
    rows = []
    colors = []
    for row in shown.to_dict(orient="records"):
        rows.append(
            [
                _short(row.get("date"), 10),
                _short(row.get("game_id"), 5),
                _short(row.get("player"), 16),
                _short(row.get("opponent"), 16),
                _short(row.get("time_control"), 6),
                _short(row.get("color"), 5),
                _status_label(row.get("status")),
                _num(row.get("composite_z")),
                _num(row.get("sprt_llr_step")),
            ]
        )
        colors.append(_status_color(row.get("status")))

    _table(
        ["Date", "Game", "Player", "Opponent", "TC", "Color", "Tier", "Z", "SPRT"],
        rows,
        right={1, 7, 8},
        status_col=6,
        row_colors=colors,
        use_color=use_color,
    )
    remaining = len(games) - len(shown)
    if remaining > 0:
        print(_paint(f"  {remaining} more games not shown.", _DIM, use_color))
    print()


def _print_diagnostic_games(scored_view, top: int, use_color: bool) -> None:
    """Always identify games worth inspection, even if none cross a heuristic tier."""
    if scored_view.empty:
        return
    print(_paint("Games to inspect (ranked, NOT accusations)", _BOLD, use_color))
    scored = scored_view.sort_values(["composite_z", "date"], ascending=[False, False])
    rows = []
    for row in scored.head(top).to_dict(orient="records"):
        signals = []
        for key, label in (
            ("z_accuracy", "accuracy"), ("z_critical", "critical moves"),
            ("z_choice", "choice"), ("z_surprise", "surprise"),
            ("z_timing", "timing"), ("z_phase_shift", "phase"),
            ("z_contextual_toggle", "context"),
        ):
            value = row.get(key)
            if value is not None and float(value) == float(value) and float(value) >= 1.0:
                signals.append((float(value), label))
        signals.sort(reverse=True)
        reasons = ", ".join(f"{label} {value:.1f}" for value, label in signals[:2]) or "no strong signal"
        rows.append([
            _short(row.get("date"), 10), _short(row.get("game_id"), 6),
            _short(row.get("opponent"), 17), _short(row.get("time_control"), 6),
            _num(row.get("composite_z")), _num(row.get("capped_acpl"), 1),
            _num(100 * row["top1_rate"], 0) + "%" if row.get("top1_rate") is not None else "—",
            _short(reasons, 38),
        ])
    _table(["Date", "Game", "Opponent", "TC", "Score", "CPL ≤300", "Top1", "Largest signals"],
           rows, right={4, 5, 6})
    print(_paint("  Ranked relative to this PGN's scores; even the top game may be ordinary.",
                 _DIM, use_color))
    print()


def _print_move_quality(eval_df, player: str | None, use_color: bool) -> None:
    if eval_df is None or eval_df.empty or not player:
        return
    moves = eval_df[eval_df["player"].astype(str).str.casefold() == player.casefold()].copy()
    if moves.empty or "centipawn_loss" not in moves.columns:
        return
    import pandas as pd
    cpl = pd.to_numeric(moves["centipawn_loss"], errors="coerce").dropna()
    if cpl.empty:
        return
    extreme = int((cpl > 1000).sum())
    median = float(cpl.median())
    coverage = "unknown"
    if "move_time" in moves.columns:
        coverage = f"{100 * moves['move_time'].notna().mean():.0f}%"
    print(_paint("Move-level data quality", _BOLD, use_color))
    print(f"  Evaluated moves: {len(cpl)} | Median CPL: {median:.1f}"
          f" | CPL >1000: {extreme} | Recorded move times: {coverage}")
    if extreme:
        print(_paint("  Very large CPL values can come from mate-score changes,"
                     " already-lost positions, or genuine blunders; raw ACPL is not stable.",
                     _DIM, use_color))
    print()


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
    parser.add_argument("--reference", type=Path, help="Optional frozen proxy reference JSON; never required")
    parser.add_argument("--platform", choices=("chesscom", "lichess"),
                        help="Platform of the PGN account; required with --reference")
    parser.add_argument("--json-out", type=Path, help="Write the reported rows to a JSON file")
    parser.add_argument("--top", type=int, default=8, help="Players to show (default: 8)")
    parser.add_argument("--top-games", type=int, default=8, help="Games to show (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Append the full player table")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.pgn.is_absolute() and not args.pgn.is_file():
        candidate = BACKEND_DIR.parent / args.pgn
        if candidate.is_file():
            args.pgn = candidate
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

    calibration = load_calibration()
    scored_df = score_games(features_df, eval_df=eval_df, calibration=calibration)
    players_df = aggregate_player_scores(scored_df, features_df=features_df)
    if args.reference:
        if not args.platform:
            print("--platform chesscom|lichess required with --reference", file=sys.stderr)
            return 2
        try:
            players_df = compare_accounts(
                players_df, read_proxy_reference(args.reference), platform=args.platform
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Invalid reference: {exc}", file=sys.stderr)
            return 2
    use_color = sys.stdout.isatty() and not args.no_color

    _print_header(args, raw_df, features_df, players_df, calibration, use_color)

    if args.player:
        target = args.player.casefold()
        players_view = players_df[players_df["player"].str.casefold() == target].copy()
        scored_view = scored_df[scored_df["player"].str.casefold() == target].copy()
        if players_view.empty:
            available = sorted(players_df["player"].dropna().unique())
            preview = ", ".join(available[:8])
            print(
                f"Player {args.player!r} not found. Available: {preview}"
                + (" ..." if len(available) > 8 else ""),
                file=sys.stderr,
            )
            return 1
        players_view = _ranked(players_view)
        _print_player_cards(players_view, use_color)
    else:
        players_view = players_df
        scored_view = scored_df
        _print_summary(players_df, use_color)
        _print_players_table(_ranked(players_df), args.top, use_color)

    _print_diagnostic_games(scored_view, args.top_games, use_color)
    _print_games_table(scored_view, args.top_games, use_color)
    _print_move_quality(eval_df, args.player, use_color)
    print(_paint("Heuristic tiers only; not a cheating determination.", _DIM, use_color))
    if args.reference:
        print(_paint("Proxy percentiles compare against publicly unconfirmed accounts, "
                     "NOT proven-clean players; not a cheating probability or validated FPR.",
                     _DIM, use_color))

    if args.verbose:
        print(_paint("Full player table", _BOLD, use_color))
        print(players_view.to_string(index=False))
        print()

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
        print(_paint(f"Wrote {args.json_out}", _DIM, use_color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
