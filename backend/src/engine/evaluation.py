from __future__ import annotations

import atexit
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import pandas as pd

# Filesystem anchors preserved from the pre-refactor layout (backend/engine_eval.py):
# backend/ is now two levels above this module, so derive it explicitly.
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT = BACKEND_DIR.parent
CACHE_DIR = BACKEND_DIR / ".cache"
MATE_SCORE_CP = 10000

# Worker-local state so each process spawns Stockfish once and reuses it across games
_WORKER_ENGINE: chess.engine.SimpleEngine | None = None
_WORKER_LIMIT: chess.engine.Limit | None = None
_WORKER_SHALLOW_DEPTH: int = 6
_WORKER_MULTIPV: int = 3


@dataclass(frozen=True)
class EngineConfig:
    stockfish_path: Path | None = None
    depth: int | None = 14
    shallow_depth: int = 6
    time_per_move: float | None = None
    multipv: int = 3
    workers: int | None = None  # Defaults to cpu_count() - 1
    threads_per_worker: int = 1
    hash_mb_per_worker: int = 128
    use_cache: bool = True
    cache_dir: Path = CACHE_DIR


def find_stockfish(explicit: str | Path | None = None) -> Path:
    candidates = [
        str(explicit) if explicit else None,
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        str(
            PROJECT.parent
            / "chess-decision-engine"
            / "engines"
            / "stockfish-19"
            / "stockfish"
            / "stockfish-macos-universal"
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return Path(candidate)
    raise RuntimeError(
        "Stockfish binary not found. Install Stockfish (e.g. `brew install stockfish` or "
        "`apt install stockfish`) or set the STOCKFISH_PATH environment variable."
    )


def score_to_cp(score: chess.engine.PovScore, turn: chess.Color) -> int:
    """Convert PovScore to centipawns from `turn`'s perspective, clamping mates."""
    pov = score.pov(turn)
    if pov.is_mate():
        mate_in = pov.mate()
        if mate_in is not None and mate_in > 0:
            return MATE_SCORE_CP - mate_in
        return -MATE_SCORE_CP - (mate_in or 0)
    return pov.score() or 0


def cp_to_win_prob(cp: int) -> float:
    """
    Convert centipawns (mover's perspective) to win probability in [0.0, 1.0]
    using the standard Lichess sigmoid scaling.
    """
    clamped = max(-1000, min(1000, cp))
    return 0.5 + 0.5 * (2.0 / (1.0 + math.exp(-0.00368208 * clamped)) - 1.0)


def _rank_instability(shallow_moves: list[str], deep_moves: list[str]) -> float:
    """
    Normalized rank churn in [0.0, 1.0] between shallow MultiPV and deep MultiPV.
    High values indicate positions where deeper search overturns shallow intuition.
    """
    if not shallow_moves or not deep_moves:
        return 0.0
    k = min(len(shallow_moves), len(deep_moves))
    if k == 1:
        return 0.0 if shallow_moves[0] == deep_moves[0] else 1.0

    disagreement = 0.0
    if shallow_moves[0] != deep_moves[0]:
        disagreement += 0.6
    overlap = len(set(shallow_moves[:k]) & set(deep_moves[:k]))
    disagreement += 0.4 * (1.0 - overlap / k)
    return round(min(1.0, disagreement), 4)


def _estimate_human_prior(
    board: chess.Board,
    move: chess.Move,
    shallow_moves: list[str],
    shallow_evals: list[int],
    deep_eval_before: int,
    eval_after: int,
) -> float:
    """
    Estimate human move-selection prior probability P_human(move | board) without external weights.
    Combines shallow-search (depth 6) Boltzmann utility with human perceptual biases
    (captures, checks, forward moves vs. counter-intuitive backward quiet retreats).
    """
    move_uci = move.uci()
    legal_count = max(1, board.legal_moves.count())
    if legal_count == 1:
        return 1.0

    # 1. Perceptual salience bias (humans spot captures/checks easily, miss quiet retreats)
    is_cap = board.is_capture(move)
    is_chk = board.gives_check(move)
    from_rank = chess.square_rank(move.from_square)
    to_rank = chess.square_rank(move.to_square)
    rank_delta = (to_rank - from_rank) if board.turn == chess.WHITE else (from_rank - to_rank)
    is_backward_quiet = (rank_delta < 0) and not is_cap and not is_chk

    salience_mult = 1.0
    if is_cap:
        salience_mult *= 1.35
    if is_chk:
        salience_mult *= 1.30
    if is_backward_quiet:
        salience_mult *= 0.55

    # 2. Shallow search (human lookahead horizon) softmax probability
    if shallow_moves and shallow_evals:
        best_shallow_cp = shallow_evals[0]
        if move_uci in shallow_moves:
            idx = shallow_moves.index(move_uci)
            move_shallow_cp = shallow_evals[idx]
        else:
            # Not even in shallow MultiPV top candidates: approximate from deep eval_after
            move_shallow_cp = min(eval_after, best_shallow_cp - 65)

        shallow_loss_pawns = max(0.0, (best_shallow_cp - move_shallow_cp) / 100.0)
    else:
        shallow_loss_pawns = max(0.0, (deep_eval_before - eval_after) / 100.0)

    # Softmax over legal branching factor with temperature tau = 0.65 pawns
    unnormalized = math.exp(-shallow_loss_pawns / 0.65) * salience_mult
    baseline_mass = 1.0 + 0.22 * max(0, legal_count - 1)
    prob = unnormalized / (unnormalized + baseline_mass)
    return round(float(np.clip(prob, 0.005, 0.99)), 4)


def _shutdown_worker() -> None:
    global _WORKER_ENGINE
    if _WORKER_ENGINE is not None:
        engine, _WORKER_ENGINE = _WORKER_ENGINE, None
        try:
            engine.quit()
        except Exception:
            try:
                engine.close()
            except Exception:
                pass


def _init_worker(
    stockfish_path: str,
    depth: int | None,
    shallow_depth: int,
    time_per_move: float | None,
    multipv: int,
    threads: int,
    hash_mb: int,
) -> None:
    """Initializer run once per worker process to start a persistent Stockfish instance."""
    global _WORKER_ENGINE, _WORKER_LIMIT, _WORKER_SHALLOW_DEPTH, _WORKER_MULTIPV
    _WORKER_LIMIT = chess.engine.Limit(depth=depth, time=time_per_move)
    _WORKER_SHALLOW_DEPTH = shallow_depth
    _WORKER_MULTIPV = multipv
    _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _WORKER_ENGINE.configure({"Threads": threads, "Hash": hash_mb})
    atexit.register(_shutdown_worker)


def _enrich_cached_metrics(fen: str | None, move_uci: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Backfill new multi-depth & human-prior fields when loading older cache entries."""
    out = dict(metrics)
    board = chess.Board(fen) if fen else None
    move = chess.Move.from_uci(move_uci) if (board and move_uci) else None

    if board is not None and move is not None and move in board.legal_moves:
        is_cap = board.is_capture(move)
        is_chk = board.gives_check(move)
    else:
        is_cap = bool(out.get("is_capture", False))
        is_chk = bool(out.get("is_check", False))

    out.setdefault("is_capture", is_cap)
    out.setdefault("is_check", is_chk)
    out.setdefault("is_quiet", not is_cap and not is_chk)
    out.setdefault("shallow_best_move", out.get("best_move"))
    out.setdefault("shallow_eval_cp", out.get("eval_before"))
    out.setdefault("horizon_gain_cp", 0)
    out.setdefault("is_deep_only_top1", False)
    out.setdefault("rank_instability", 0.0)

    if out.get("human_prior_prob") is None:
        eval_before = out.get("eval_before") or 0
        eval_after = out.get("eval_after") or 0
        if board is not None and move is not None and move in board.legal_moves:
            hp = _estimate_human_prior(board, move, [], [], int(eval_before), int(eval_after))
        else:
            cpl = max(0, int(out.get("centipawn_loss") or 0))
            hp = round(float(np.clip(math.exp(-(cpl / 100.0) / 0.7) * 0.45, 0.01, 0.95)), 4)
        out["human_prior_prob"] = hp

    hp_val = float(out.get("human_prior_prob") or 0.25)
    is_top1 = bool(out.get("is_top1", False))
    gap = float(out.get("eval_gap") or 0.0)
    if out.get("machine_surprise") is None:
        surprise = -math.log(max(hp_val, 1e-3)) if (is_top1 and gap >= 40.0) else 0.0
        out["machine_surprise"] = round(surprise, 4)

    return out


def _load_legacy_game_cache(
    cache_dir: Path,
) -> tuple[dict[tuple[str, ...], dict[int, dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    """
    Index `.cache/games/*.json` and `.cache/position_evals.json`.
    Automatically backfills new horizon/surprise fields on legacy cache hits.
    """
    game_lookup: dict[tuple[str, ...], dict[int, dict[str, Any]]] = {}
    pos_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    native_cache = cache_dir / "position_evals.json"
    if native_cache.is_file():
        try:
            raw = json.loads(native_cache.read_text(encoding="utf-8"))
            for key_str, metrics in raw.items():
                fen, move_uci = key_str.split("||", 1)
                pos_lookup[(fen, move_uci)] = _enrich_cached_metrics(fen, move_uci, metrics)
        except (OSError, ValueError):
            pass

    games_dir = cache_dir / "games"
    if not games_dir.is_dir():
        return game_lookup, pos_lookup

    for json_file in games_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue

        all_cached_rows = data.get("_rows_player", []) + data.get("_rows_opponent", [])
        if not all_cached_rows:
            continue

        all_cached_rows.sort(key=lambda r: r.get("ply", 0))
        ply_map: dict[int, dict[str, Any]] = {}

        for r in all_cached_rows:
            ply = r.get("ply")
            move_uci = r.get("move_uci")
            fen = r.get("fen")
            if ply is None or not move_uci:
                continue

            eval_before = int(round(r.get("best_score_cp", 0)))
            eval_after = int(round(r.get("played_score_cp", 0)))
            cp_loss = int(round(r.get("cp_loss", max(0, eval_before - eval_after))))

            wp_before = cp_to_win_prob(eval_before)
            wp_after = cp_to_win_prob(eval_after)
            wp_loss = max(0.0, wp_before - wp_after)

            rank = r.get("rank_among_evaluated")
            if rank is None and r.get("best_move"):
                rank = 1

            base_metrics = {
                "best_move": r.get("best_uci"),
                "move_rank": rank,
                "is_top1": bool(r.get("best_move", rank == 1)),
                "is_top3": rank is not None and rank <= 3,
                "eval_before": eval_before,
                "eval_after": eval_after,
                "centipawn_loss": cp_loss,
                "win_prob_before": round(wp_before, 4),
                "win_prob_after": round(wp_after, 4),
                "win_prob_loss": round(wp_loss, 4),
                "eval_gap": int(round(r["candidate_gap_cp"])) if r.get("candidate_gap_cp") is not None else None,
                "legal_moves_count": r.get("legal_count"),
                "piece_count": len(chess.Board(fen).piece_map()) if fen else None,
            }
            metrics = _enrich_cached_metrics(fen, move_uci, base_metrics)
            ply_map[int(ply)] = metrics
            if fen:
                pos_lookup[(fen, move_uci)] = metrics

        move_sig = tuple(f"{r['ply']}:{r['move_uci']}" for r in all_cached_rows if "ply" in r and "move_uci" in r)
        if move_sig:
            game_lookup[move_sig] = ply_map

    return game_lookup, pos_lookup


def _save_native_position_cache(
    cache_dir: Path,
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Persist newly evaluated `(fen_before, move_uci)` entries to `.cache/position_evals.json`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "position_evals.json"
    tmp = target.with_suffix(f".{os.getpid()}.tmp")
    serialized = {f"{fen}||{uci}": metrics for (fen, uci), metrics in lookup.items()}
    try:
        tmp.write_text(json.dumps(serialized), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def evaluate_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    multipv: int = 3,
    shallow_depth: int = 6,
) -> dict[str, Any]:
    """
    Analyze a single position with MultiPV from `board.turn`'s perspective.
    Streams intermediate `info` packets during a single search pass to capture
    both shallow (`shallow_depth`) and final (`limit.depth`) MultiPV rankings at zero extra cost.
    """
    legal_count = board.legal_moves.count()

    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            terminal_cp = 0
        elif outcome.winner == board.turn:
            terminal_cp = MATE_SCORE_CP
        else:
            terminal_cp = -MATE_SCORE_CP
        return {
            "top_moves": [],
            "top_evals": [],
            "shallow_moves": [],
            "shallow_evals": [],
            "best_move": None,
            "shallow_best_move": None,
            "eval_cp": terminal_cp,
            "shallow_eval_cp": terminal_cp,
            "eval_gap": None,
            "rank_instability": 0.0,
            "legal_moves_count": legal_count,
        }

    by_depth: dict[int, dict[int, tuple[str, int]]] = {}
    with engine.analysis(board, limit, multipv=multipv) as analysis:
        for info in analysis:
            pv = info.get("pv")
            score = info.get("score")
            depth = info.get("depth")
            mpv = info.get("multipv", 1)
            if pv and score is not None and depth is not None:
                by_depth.setdefault(int(depth), {})[int(mpv)] = (
                    pv[0].uci(),
                    score_to_cp(score, board.turn),
                )

    if not by_depth:
        return {
            "top_moves": [],
            "top_evals": [],
            "shallow_moves": [],
            "shallow_evals": [],
            "best_move": None,
            "shallow_best_move": None,
            "eval_cp": 0,
            "shallow_eval_cp": 0,
            "eval_gap": None,
            "rank_instability": 0.0,
            "legal_moves_count": legal_count,
        }

    max_d = max(by_depth.keys())
    final_map = by_depth[max_d]
    ordered_final = [final_map[k] for k in sorted(final_map.keys())]
    top_moves = [m for m, _ in ordered_final]
    top_evals = [e for _, e in ordered_final]

    # Pick closest depth snapshot <= shallow_depth
    shallow_candidates = [d for d in by_depth.keys() if d <= shallow_depth]
    s_depth = max(shallow_candidates) if shallow_candidates else min(by_depth.keys())
    shallow_map = by_depth[s_depth]
    ordered_shallow = [shallow_map[k] for k in sorted(shallow_map.keys())]
    shallow_moves = [m for m, _ in ordered_shallow]
    shallow_evals = [e for _, e in ordered_shallow]

    best_move = top_moves[0] if top_moves else None
    shallow_best = shallow_moves[0] if shallow_moves else best_move
    eval_cp = top_evals[0] if top_evals else 0
    shallow_eval_cp = shallow_evals[0] if shallow_evals else eval_cp
    eval_gap = (top_evals[0] - top_evals[1]) if len(top_evals) > 1 else None
    instability = _rank_instability(shallow_moves, top_moves)

    return {
        "top_moves": top_moves,
        "top_evals": top_evals,
        "shallow_moves": shallow_moves,
        "shallow_evals": shallow_evals,
        "best_move": best_move,
        "shallow_best_move": shallow_best,
        "eval_cp": eval_cp,
        "shallow_eval_cp": shallow_eval_cp,
        "eval_gap": eval_gap,
        "rank_instability": instability,
        "legal_moves_count": legal_count,
    }


def evaluate_played_move(
    engine: chess.engine.SimpleEngine,
    fen_before: str,
    move_uci: str,
    limit: chess.engine.Limit,
    multipv: int = 3,
    shallow_depth: int = 6,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate a played move and return multi-depth anti-cheat position and accuracy metrics."""
    board = chess.Board(fen_before)
    piece_count = len(board.piece_map())

    if cache is not None and fen_before in cache:
        before_res = cache[fen_before]
    else:
        before_res = evaluate_position(
            engine, board, limit, multipv=multipv, shallow_depth=shallow_depth
        )
        if cache is not None:
            cache[fen_before] = before_res

    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {move_uci} for FEN: {fen_before}")

    is_capture = board.is_capture(move)
    is_check = board.gives_check(move)
    is_quiet = not is_capture and not is_check

    board_after = board.copy(stack=False)
    board_after.push(move)
    fen_after = board_after.fen()

    if cache is not None and fen_after in cache:
        after_res = cache[fen_after]
    else:
        after_res = evaluate_position(
            engine, board_after, limit, multipv=multipv, shallow_depth=shallow_depth
        )
        if cache is not None:
            cache[fen_after] = after_res

    eval_before = before_res["eval_cp"]
    eval_after = -after_res["eval_cp"]  # Negate opponent's perspective back to mover
    centipawn_loss = max(0, eval_before - eval_after)

    win_prob_before = cp_to_win_prob(eval_before)
    win_prob_after = cp_to_win_prob(eval_after)
    win_prob_loss = max(0.0, win_prob_before - win_prob_after)

    top_moves: list[str] = before_res["top_moves"]
    top_evals: list[int] = before_res["top_evals"]
    shallow_moves: list[str] = before_res.get("shallow_moves", [])
    shallow_evals: list[int] = before_res.get("shallow_evals", [])

    move_rank = (top_moves.index(move_uci) + 1) if move_uci in top_moves else None
    is_top1 = move_rank == 1

    # Horizon gain: how much better the deep #1 move is compared to what shallow search preferred
    shallow_best = before_res.get("shallow_best_move")
    if shallow_best and shallow_best in top_moves:
        shallow_best_deep_eval = top_evals[top_moves.index(shallow_best)]
        horizon_gain_cp = max(0, eval_before - shallow_best_deep_eval)
    elif shallow_best and shallow_best != before_res["best_move"]:
        horizon_gain_cp = max(0, int(before_res.get("eval_gap") or 40))
    else:
        horizon_gain_cp = 0

    is_deep_only_top1 = bool(
        is_top1
        and shallow_best is not None
        and move_uci != shallow_best
        and (move_uci not in shallow_moves[:2] or horizon_gain_cp >= 45)
    )

    human_prior = _estimate_human_prior(
        board=board,
        move=move,
        shallow_moves=shallow_moves,
        shallow_evals=shallow_evals,
        deep_eval_before=eval_before,
        eval_after=eval_after,
    )
    eval_gap = before_res["eval_gap"]
    machine_surprise = (
        round(-math.log(max(human_prior, 1e-3)), 4)
        if (is_top1 and eval_gap is not None and eval_gap >= 40)
        else 0.0
    )

    return {
        "best_move": before_res["best_move"],
        "shallow_best_move": shallow_best,
        "move_rank": move_rank,
        "is_top1": is_top1,
        "is_top3": move_rank is not None and move_rank <= 3,
        "is_capture": is_capture,
        "is_check": is_check,
        "is_quiet": is_quiet,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "shallow_eval_cp": before_res.get("shallow_eval_cp", eval_before),
        "centipawn_loss": centipawn_loss,
        "win_prob_before": round(win_prob_before, 4),
        "win_prob_after": round(wp_after := win_prob_after, 4),
        "win_prob_loss": round(win_prob_loss, 4),
        "eval_gap": eval_gap,
        "horizon_gain_cp": horizon_gain_cp,
        "is_deep_only_top1": is_deep_only_top1,
        "rank_instability": before_res.get("rank_instability", 0.0),
        "human_prior_prob": human_prior,
        "machine_surprise": machine_surprise,
        "legal_moves_count": before_res["legal_moves_count"],
        "piece_count": piece_count,
    }


def _evaluate_single_game(game_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Worker entrypoint: evaluates all plies of one game sequentially using FEN cache."""
    if _WORKER_ENGINE is None or _WORKER_LIMIT is None:
        raise RuntimeError("Worker Stockfish engine was not initialized.")

    if "Clear Hash" in _WORKER_ENGINE.options:
        _WORKER_ENGINE.configure({"Clear Hash": True})

    fen_cache: dict[str, dict[str, Any]] = {}
    enriched_rows: list[dict[str, Any]] = []

    for row in game_records:
        if "_cached_metrics" in row:
            metrics = row.pop("_cached_metrics")
        else:
            metrics = evaluate_played_move(
                engine=_WORKER_ENGINE,
                fen_before=row["fen_before"],
                move_uci=row["move_uci"],
                limit=_WORKER_LIMIT,
                multipv=_WORKER_MULTIPV,
                shallow_depth=_WORKER_SHALLOW_DEPTH,
                cache=fen_cache,
            )
        enriched_rows.append({**row, **metrics})

    return enriched_rows


def evaluate_games_df(
    df: pd.DataFrame,
    config: EngineConfig | None = None,
    skip_cached_opening_plies: bool = True,
) -> pd.DataFrame:
    """
    Main pipeline step: checks `.cache` first so already-evaluated games skip Stockfish,
    evaluates any remaining uncached games in parallel with progress tracking,
    cleanly terminates worker engines, and returns the enriched DataFrame.
    """
    if df.empty:
        return df.copy()

    import time
    from concurrent.futures import as_completed

    cfg = config or EngineConfig()
    game_cache, pos_cache = _load_legacy_game_cache(cfg.cache_dir) if cfg.use_cache else ({}, {})

    grouped_games: list[list[dict[str, Any]]] = [
        group.sort_values("ply").to_dict(orient="records")
        for _, group in df.groupby("game_id", sort=False)
    ]

    cached_games: list[list[dict[str, Any]]] = []
    uncached_games: list[list[dict[str, Any]]] = []

    empty_metrics = {
        "best_move": None,
        "shallow_best_move": None,
        "move_rank": None,
        "is_top1": False,
        "is_top3": False,
        "is_Quiet": False,
        "eval_before": None,
        "eval_after": None,
        "shallow_eval_cp": None,
        "centipawn_loss": None,
        "win_prob_before": None,
        "win_prob_after": None,
        "win_prob_loss": None,
        "eval_gap": None,
        "horizon_gain_cp": 0,
        "is_deep_only_top1": False,
        "rank_instability": 0.0,
        "human_prior_prob": None,
        "machine_surprise": 0.0,
        "piece_count": None,
    }

    for game_records in grouped_games:
        evaluable_sig = tuple(
            f"{r['ply']}:{r['move_uci']}"
            for r in game_records
            if r["ply"] > 8 and r.get("legal_moves_count", 2) > 1
        )
        ply_map = game_cache.get(evaluable_sig)

        if ply_map is not None and skip_cached_opening_plies:
            enriched = []
            for r in game_records:
                board = chess.Board(r["fen_before"])
                hit = ply_map.get(r["ply"])
                if hit is not None:
                    m = _enrich_cached_metrics(r["fen_before"], r["move_uci"], hit)
                    m["piece_count"] = len(board.piece_map())
                    if m.get("legal_moves_count") is None:
                        m["legal_moves_count"] = r.get("legal_moves_count", board.legal_moves.count())
                else:
                    m = {
                        **empty_metrics,
                        "is_quiet": not bool(r.get("is_capture")) and not bool(r.get("is_check")),
                        "legal_moves_count": r.get("legal_moves_count", board.legal_moves.count()),
                        "piece_count": len(board.piece_map()),
                    }
                enriched.append({**r, **m})
            cached_games.append(enriched)
            continue

        all_cached = True
        for row in game_records:
            hit = pos_cache.get((row["fen_before"], row["move_uci"]))
            if hit is not None:
                row["_cached_metrics"] = _enrich_cached_metrics(
                    row["fen_before"], row["move_uci"], hit
                )
            else:
                all_cached = False

        if all_cached:
            cached_games.append([
                {**{k: v for k, v in r.items() if k != "_cached_metrics"}, **r["_cached_metrics"]}
                for r in game_records
            ])
        else:
            uncached_games.append(game_records)

    print(
        f"[engine_eval] {len(cached_games)}/{len(grouped_games)} games loaded from cache; "
        f"{len(uncached_games)} games queued for Stockfish.",
        flush=True,
    )

    evaluated_games: list[list[dict[str, Any]]] = list(cached_games)

    if uncached_games:
        sf_path = find_stockfish(cfg.stockfish_path)
        max_workers = cfg.workers or max(1, (os.cpu_count() or 2) - 1)
        total_uncached = len(uncached_games)
        start_time = time.monotonic()

        def _log_progress(done: int) -> None:
            elapsed = time.monotonic() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total_uncached - done) / rate if rate > 0 else 0.0
            pct = (done / total_uncached) * 100.0
            print(
                f"[engine_eval] {done}/{total_uncached} games ({pct:.1f}%) | "
                f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
                flush=True,
            )

        if max_workers == 1 or total_uncached == 1:
            _init_worker(
                str(sf_path),
                cfg.depth,
                cfg.shallow_depth,
                cfg.time_per_move,
                cfg.multipv,
                cfg.threads_per_worker,
                cfg.hash_mb_per_worker,
            )
            try:
                for idx, game_records in enumerate(uncached_games, 1):
                    evaluated_games.append(_evaluate_single_game(game_records))
                    _log_progress(idx)
            finally:
                _shutdown_worker()
        else:
            n_workers = min(max_workers, total_uncached)
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(
                    str(sf_path),
                    cfg.depth,
                    cfg.shallow_depth,
                    cfg.time_per_move,
                    cfg.multipv,
                    cfg.threads_per_worker,
                    cfg.hash_mb_per_worker,
                ),
            ) as executor:
                futures = [
                    executor.submit(_evaluate_single_game, game_records)
                    for game_records in uncached_games
                ]
                for idx, future in enumerate(as_completed(futures), 1):
                    evaluated_games.append(future.result())
                    _log_progress(idx)

                stop_futures = [executor.submit(_shutdown_worker) for _ in range(n_workers)]
                for f in stop_futures:
                    try:
                        f.result(timeout=5.0)
                    except Exception:
                        pass

        if cfg.use_cache:
            metric_keys = {
                "best_move",
                "shallow_best_move",
                "move_rank",
                "is_top1",
                "is_top3",
                "is_capture",
                "is_check",
                "is_quiet",
                "eval_before",
                "eval_after",
                "shallow_eval_cp",
                "centipawn_loss",
                "win_prob_before",
                "win_prob_after",
                "win_prob_loss",
                "eval_gap",
                "horizon_gain_cp",
                "is_deep_only_top1",
                "rank_instability",
                "human_prior_prob",
                "machine_surprise",
                "legal_moves_count",
                "piece_count",
            }
            for game_results in evaluated_games[len(cached_games):]:
                for row in game_results:
                    if row.get("eval_before") is not None:
                        pos_cache[(row["fen_before"], row["move_uci"])] = {
                            k: row[k] for k in metric_keys if k in row
                        }
            _save_native_position_cache(cfg.cache_dir, pos_cache)

    all_rows = [row for game in evaluated_games for row in game]
    result_df = pd.DataFrame(all_rows)
    return result_df.sort_values(["game_id", "ply"]).reset_index(drop=True)