"""PGN ingestion and bot-game filtering."""

from .pgn import BOT_NAME_PATTERNS, is_bot_game, parse_pgn_for_detection

__all__ = ["BOT_NAME_PATTERNS", "is_bot_game", "parse_pgn_for_detection"]
