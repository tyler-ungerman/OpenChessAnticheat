"""Pluggable chess fair-play detectors; importing this package registers them."""

from .base import DETECTOR_REGISTRY, DetectorConfig, DetectorFn, register_detector
from .calibration import (
    DEFAULT_BASELINE,
    build_empirical_baselines,
    load_calibration,
)
from .compiler import (
    run_bayesian_sprt_ensemble,
    run_game_style_detector,
    run_hmm_toggle_detector,
    run_statistical_detector,
)
from .game_style import (
    DEFAULT_MAX_PROFILE_GAMES,
    DEFAULT_SKIP_OPENING_PLIES,
    GameInput,
    MoveInput,
    StyleProfile,
    build_style_profile,
    evaluate_style_deviation,
    game_inputs_from_eval_df,
    split_analysis_window,
    style_deviation_score,
)
from .hmm_toggle import fit_game_hmm_toggle
from .reagan_choice import fit_game_sensitivity_s
from .statistical_z import aggregate_player_scores, score_games

__all__ = [
    "DEFAULT_BASELINE",
    "DEFAULT_MAX_PROFILE_GAMES",
    "DEFAULT_SKIP_OPENING_PLIES",
    "DETECTOR_REGISTRY",
    "DetectorConfig",
    "DetectorFn",
    "GameInput",
    "MoveInput",
    "StyleProfile",
    "aggregate_player_scores",
    "build_empirical_baselines",
    "build_style_profile",
    "evaluate_style_deviation",
    "fit_game_hmm_toggle",
    "fit_game_sensitivity_s",
    "game_inputs_from_eval_df",
    "load_calibration",
    "register_detector",
    "run_bayesian_sprt_ensemble",
    "run_game_style_detector",
    "run_hmm_toggle_detector",
    "run_statistical_detector",
    "score_games",
    "split_analysis_window",
    "style_deviation_score",
]

