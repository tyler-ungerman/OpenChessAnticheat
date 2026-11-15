import re
import chess.pgn
import pandas as pd

# Common Chess.com / generic bot username patterns (customize as needed)
BOT_NAME_PATTERNS = re.compile(
    r"(^stockfish|^komodo|^leela|^lc0|^maia\d*|^mittens|^nelson|^martin|^elena|^antonio|^coach|"
    r"-bot$|_bot$)",
    re.IGNORECASE,
)


def is_bot_game(headers: chess.pgn.Headers, extra_bot_names: set[str] | None = None) -> bool:
    """Detect bot/computer games from Lichess and Chess.com PGN headers."""
    # 1. Lichess explicit BOT title
    if headers.get("WhiteTitle", "").upper() == "BOT" or headers.get("BlackTitle", "").upper() == "BOT":
        return True

    # 2. Untimed computer games (common on Chess.com bot challenges)
    tc = headers.get("TimeControl", "").strip()
    if tc in {"-", "?"}:
        return True

    # 3. Event header indicators
    event = headers.get("Event", "")
    if re.search(r"\b(vs\.?\s*computer|bot|engine|stockfish)\b", event, re.IGNORECASE):
        return True

    # 4. Player username checks
    extra = {name.lower() for name in (extra_bot_names or set())}
    for side in ("White", "Black"):
        name = headers.get(side, "").strip()
        if not name:
            continue
        if name.lower() in extra or BOT_NAME_PATTERNS.search(name):
            return True

    return False


def parse_pgn_for_detection(
    pgn_path: str,
    exclude_bots: bool = True,
    extra_bot_names: set[str] | None = None,
) -> pd.DataFrame:
    records = []
    skipped_bots = 0

    with open(pgn_path, encoding="utf-8") as pgn_file:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            headers = game.headers
            bot_flag = is_bot_game(headers, extra_bot_names=extra_bot_names)
            if exclude_bots and bot_flag:
                skipped_bots += 1
                continue

            game_idx += 1
            board = game.board()

            tc = headers.get("TimeControl", "0+0")
            try:
                base_time, increment = map(float, tc.split("+")) if "+" in tc else (float(tc), 0.0)
            except ValueError:
                base_time, increment = 0.0, 0.0

            prev_clocks = {chess.WHITE: base_time, chess.BLACK: base_time}
            node = game
            ply = 0

            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                ply += 1

                turn = board.turn
                player_color = "White" if turn == chess.WHITE else "Black"
                player_name = headers.get(player_color, "Unknown")
                clock_after = next_node.clock()

                time_spent = None
                if clock_after is not None and prev_clocks[turn] is not None:
                    inc = increment if ply > 2 else 0.0
                    time_spent = round(max(0.0, prev_clocks[turn] - clock_after + inc), 3)
                    prev_clocks[turn] = clock_after

                records.append({
                    "game_id": game_idx,
                    "date": headers.get("UTCDate", headers.get("Date")),
                    "event": headers.get("Event"),
                    "time_control": tc,
                    "white_title": headers.get("WhiteTitle"),
                    "black_title": headers.get("BlackTitle"),
                    "is_bot_game": bot_flag,
                    "ply": ply,
                    "move_number": (ply + 1) // 2,
                    "color": player_color,
                    "player": player_name,
                    "elo": headers.get(f"{player_color}Elo"),
                    "fen_before": board.fen(),
                    "move_uci": move.uci(),
                    "move_san": board.san(move),
                    "clock_remaining": clock_after,
                    "time_spent": time_spent,
                    "is_capture": board.is_capture(move),
                    "is_check": board.gives_check(move),
                    "legal_moves_count": board.legal_moves.count(),
                })

                board.push(move)
                node = next_node

    if skipped_bots:
        print(f"[game_history] Skipped {skipped_bots} bot games.", flush=True)

    return pd.DataFrame(records)