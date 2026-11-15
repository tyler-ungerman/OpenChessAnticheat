from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.detectors import (
    DetectorConfig,
    aggregate_player_scores,
    build_empirical_baselines,
    score_games,
)
from src.features import build_features

DEFAULT_CONTROLS_MOVES = Path(__file__).resolve().parent.parent / "data" / "controls_moves.jsonl.gz"


def load_control_moves_df(
    moves_path: Path = DEFAULT_CONTROLS_MOVES,
    max_games: int = 1500,
) -> pd.DataFrame:
    """
    Load move-level records from `controls_moves.jsonl.gz` using its exact schema:
    ['ply', 'side', 'fen', 'uci', 'san', 'best_uci', 'n_legal', 'best_cp', 'played_cp',
     'cp_loss', 'win_before', 'win_after', 'rank', 'gap12_cp', 'time_spent',
     'game_id', 'time_control', 'account', 'is_focal', 'mover_rating', 'account_label']
    """
    if not moves_path.is_file():
        raise FileNotFoundError(f"Control moves file not found: {moves_path}")

    rows = []
    game_map: dict[str, int] = {}

    with gzip.open(moves_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # Keep clean accounts (account_label == 0)
            if rec.get("account_label", 0) not in (0, False, None, "clean"):
                continue

            raw_gid = str(rec.get("game_id") or "")
            if not raw_gid:
                continue

            if raw_gid not in game_map:
                if len(game_map) >= max_games:
                    continue
                game_map[raw_gid] = len(game_map) + 1
            gid = game_map[raw_gid]

            side_raw = str(rec.get("side", "white")).lower()
            color = "White" if side_raw.startswith("w") else "Black"

            # Separate focal control account from their opponent so baselines aren't polluted
            account = str(rec.get("account") or f"ctrl_{gid % 40}")
            is_focal = bool(rec.get("is_focal", True))
            player_name = account if is_focal else f"opp_{gid}_{color}"

            ply = int(rec.get("ply", 1))
            win_before = rec.get("win_before")
            win_after = rec.get("win_after")
            wp_loss = (
                max(0.0, float(win_before) - float(win_after))
                if win_before is not None and win_after is not None
                else 0.0
            )

            rank = rec.get("rank")
            best_uci = rec.get("best_uci")
            uci = rec.get("uci")
            is_top1 = (rank == 1) if rank is not None else (best_uci is not None and uci == best_uci)
            is_top3 = (int(rank) <= 3) if rank is not None else is_top1

            # Estimate piece count from FEN placement field (first token) if present
            fen = rec.get("fen")
            if isinstance(fen, str) and " " in fen:
                placement = fen.split(" ", 1)[0]
                piece_count = sum(1 for ch in placement if ch.isalpha())
            else:
                piece_count = 20

            rows.append(
                {
                    "game_id": gid,
                    "date": rec.get("date") or "2026.01.01",
                    "time_control": str(rec.get("time_control") or "rapid"),
                    "ply": ply,
                    "move_number": (ply + 1) // 2,
                    "color": color,
                    "player": player_name,
                    "is_focal": is_focal,
                    "elo": rec.get("mover_rating"),
                    "move_uci": uci or "e2e4",
                    "time_spent": rec.get("time_spent"),
                    "legal_moves_count": rec.get("n_legal", 20),
                    "piece_count": piece_count,
                    "best_move": best_uci,
                    "move_rank": rank,
                    "is_top1": bool(is_top1),
                    "is_top3": bool(is_top3),
                    "eval_before": rec.get("best_cp"),
                    "eval_after": rec.get("played_cp"),
                    "centipawn_loss": rec.get("cp_loss"),
                    "win_prob_loss": wp_loss,
                    "eval_gap": rec.get("gap12_cp"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "eval_before",
        "eval_after",
        "centipawn_loss",
        "win_prob_loss",
        "eval_gap",
        "time_spent",
        "legal_moves_count",
        "piece_count",
        "elo",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    focal_players = df[df["is_focal"]]["player"].nunique()
    print(
        f"[benchmark] Loaded {len(df)} moves | {df['game_id'].nunique()} games | "
        f"{focal_players} focal control accounts."
    )
    return df


def inject_synthetic_cheater(
    clean_eval_df: pd.DataFrame,
    target_player: str,
    mode: str = "selective_critical",
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic assisted player session from clean games:
      - 'full_engine': 85% of non-opening moves replaced with engine best move + flat 5s clock
      - 'selective_critical': 95% of sharp/critical moves (eval_gap >= 80) replaced with engine best move
    """
    rng = np.random.default_rng(rng_seed)
    df = clean_eval_df.copy()
    mask = (df["player"] == target_player) & (df["ply"] > 8) & (df["legal_moves_count"] > 1)

    if mode == "selective_critical":
        mask = mask & (df["eval_gap"].fillna(0) >= 80) & (df["eval_before"].abs().fillna(999) <= 250)
        prob = 0.95
    else:
        prob = 0.85

    inject_idx = df[mask].index[rng.random(mask.sum()) < prob]
    df.loc[inject_idx, "centipawn_loss"] = 0.0
    df.loc[inject_idx, "eval_after"] = df.loc[inject_idx, "eval_before"]
    df.loc[inject_idx, "is_top1"] = True
    df.loc[inject_idx, "is_top3"] = True
    df.loc[inject_idx, "move_rank"] = 1
    df.loc[inject_idx, "time_spent"] = np.round(rng.normal(5.2, 0.7, size=len(inject_idx)).clip(2.0, 9.0), 2)
    df["player"] = df["player"].replace({target_player: f"{target_player}_INJECTED_{mode}"})
    return df


if __name__ == "__main__":
    print("1. Loading control moves sample...")
    ctrl_eval_df = load_control_moves_df(max_games=2500)
    if ctrl_eval_df.empty:
        raise SystemExit("No control moves loaded from controls_moves.jsonl.gz.")

    all_features = build_features(ctrl_eval_df)
    if all_features.empty:
        raise SystemExit("build_features returned 0 rows.")

    # Separate focal control accounts from one-off opponents
    focal_accounts = set(ctrl_eval_df[ctrl_eval_df["is_focal"]]["player"].unique())
    focal_features = all_features[all_features["player"].isin(focal_accounts)].copy()
    print(
        f"   Built {len(focal_features)} focal player-game rows across "
        f"{focal_features['player'].nunique()} control accounts "
        f"(TCs: {focal_features['time_control'].value_counts().to_dict()})."
    )

    if focal_features.empty:
        raise SystemExit("No focal player rows remaining after feature extraction.")

    # Build calibration from clean focal features
    cal = {
        "buckets": build_empirical_baselines(focal_features),
        "global_tc": {},
        "consistency_c": 0.85,
    }

    # 2. Clean False-Positive Check
    clean_scored = score_games(focal_features, eval_df=ctrl_eval_df, calibration=cal)
    clean_players = aggregate_player_scores(clean_scored)

    if clean_players.empty:
        raise SystemExit(
            f"aggregate_player_scores returned empty (time_controls present: "
            f"{clean_scored['time_control'].unique().tolist()})."
        )

    flagged_games_pct = 100.0 * (clean_scored["status"] == "flagged_outlier").mean()
    print(f"\n2. Clean Cohort FPR: {flagged_games_pct:.2f}% of games flagged ({(clean_scored['status'] == 'flagged_outlier').sum()}/{len(clean_scored)}).")
    print("   Top 8 Clean Players by Max CUSUM:")
    print(clean_players.head(8).to_string(index=False))

    # 3. Synthetic Cheat Injection Test (on a player with >= 10 games in their primary time control)
    top_player_row = clean_players.sort_values("games", ascending=False).iloc[0]
    sample_player = top_player_row["player"]
    sample_tc = top_player_row["time_control"]
    print(f"\n   Testing synthetic injection on '{sample_player}' ({sample_tc}, {int(top_player_row['games'])} games)...")

    player_gids = ctrl_eval_df[
        (ctrl_eval_df["player"] == sample_player)
        & (ctrl_eval_df["time_control"].str.lower() == sample_tc)
    ]["game_id"].unique()[:12]
    sample_games = ctrl_eval_df[ctrl_eval_df["game_id"].isin(player_gids)]

    for mode in ("full_engine", "selective_critical"):
        injected_eval = inject_synthetic_cheater(sample_games, sample_player, mode=mode)
        inj_all_feats = build_features(injected_eval)
        target_name = f"{sample_player}_INJECTED_{mode}"
        inj_target_feats = inj_all_feats[inj_all_feats["player"] == target_name].copy()

        inj_scored = score_games(inj_target_feats, eval_df=injected_eval, calibration=cal)
        inj_summary = aggregate_player_scores(inj_scored)
        print(f"\n3. Synthetic Injection ({mode}):")
        print(inj_summary.to_string(index=False))