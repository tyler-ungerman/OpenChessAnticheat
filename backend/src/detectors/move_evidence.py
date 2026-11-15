"""Chess-specific move evidence for a practical PGN investigation report.

Explanatory heuristic, not a likelihood of assistance. Prefer difficult,
quiet, engine-matching moves over simple recaptures or forced moves.
"""
from __future__ import annotations

import math

import pandas as pd


def _number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _candidate(row: dict) -> dict | None:
    if not bool(row.get("is_top1")):
        return None
    if bool(row.get("is_capture")) or bool(row.get("is_check")):
        return None
    legal = _number(row.get("legal_moves_count"), 0)
    gap = _number(row.get("eval_gap"), 0)
    eval_cp = _number(row.get("eval_before"))
    loss = _number(row.get("centipawn_loss"))
    if legal < 12 or gap < 40 or eval_cp is None or abs(eval_cp) > 250:
        return None
    move_number = int(row.get("move_number") or (int(row["ply"])+1)//2)
    if move_number < 10:
        return None
    if loss is None or loss > 35:
        return None
    # A valuable, non-obvious candidate in a competitive position.
    # This is a review priority, not calibrated statistical evidence.
    seconds = _number(row.get("time_spent"))
    fast = seconds is not None and 0 <= seconds <= 5
    points = min(4.0, gap / 75.0) + (0.7 if fast else 0.0)
    reason = f"Quiet engine-first move; next-best gap {gap:.0f} cp"
    if fast:
        reason += f"; played in {seconds:.1f}s"
    return {
        "ply": int(row["ply"]),
        "move_number": move_number,
        "san": str(row.get("move_san") or row.get("move_uci") or "?"),
        "fen_before": str(row.get("fen_before") or ""),
        "best_move_uci": row.get("best_move"),
        "evaluation_gap_cp": round(gap),
        "time_spent_seconds": seconds,
        "reason": reason,
        "_points": points,
    }


def build_game_evidence(eval_df: pd.DataFrame, scored_df: pd.DataFrame,
                        *, player: str | None = None, top_games: int = 8,
                        top_moves: int = 3) -> list[dict]:
    """Return ranked game evidence with exact pre-move FEN for board replay."""
    if eval_df.empty or scored_df.empty:
        return []
    if top_games <= 0 or top_moves <= 0:
        return []
    scored = scored_df.copy()
    if player is not None:
        scored = scored[scored["player"].astype(str).str.casefold() == player.casefold()]
    result = []
    for game in scored.to_dict(orient="records"):
        frame = eval_df[
            (eval_df["game_id"] == game["game_id"])
            & (eval_df["player"].astype(str).str.casefold() == str(game["player"]).casefold())
        ].sort_values("ply")
        moves = [_candidate(row) for row in frame.to_dict(orient="records")]
        candidates = [move for move in moves if move is not None]
        candidates.sort(key=lambda move: (-move["_points"], move["ply"]))
        shown = [{key: value for key, value in move.items() if key != "_points"}
                 for move in candidates[:top_moves]]
        strength = sum(move["_points"] for move in candidates[:5])
        result.append({
            "game_id": game["game_id"],
            "date": game.get("date"),
            "player": game["player"],
            "opponent": game.get("opponent"),
            "time_control": game.get("time_control"),
            "screening_score": game.get("composite_z"),
            "total_evaluated_moves": int(frame["centipawn_loss"].notna().sum()),
            "candidate_moves": shown,
            "candidate_count": len(candidates),
            "review_priority": round(strength, 2),
            "note": ("Quiet, non-opening engine-first decisions worth inspecting."
                     if candidates else "No notable move evidence found."),
        })
    # Evidence first, not arbitrary anomaly score first.
    result.sort(key=lambda row: (-row["review_priority"],
                                 -float(row.get("screening_score") or 0),
                                 str(row["game_id"])))
    return result[:top_games]
