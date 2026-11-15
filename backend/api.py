from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.detectors import DETECTOR_REGISTRY, load_calibration
from src.engine import EngineConfig, evaluate_games_df
from src.features import build_features
from src.ingestion import parse_pgn_for_detection

app = FastAPI(
    title="Chess Fair-Play & Behavioral Analysis API",
    version="0.3.0",
    description=(
        "Modular algorithmic chess fair-play pipeline featuring Multi-Depth Horizon Gain, "
        "Human-Prior Surprise, 2-State Viterbi HMM Toggle Detection, and Bayesian SPRT."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EngineOptions(BaseModel):
    depth: int = Field(default=14, ge=1, le=30)
    shallow_depth: int = Field(default=6, ge=1, le=12)
    multipv: int = Field(default=3, ge=1, le=10)
    workers: int = Field(default=4, ge=1, le=16)
    use_cache: bool = True


class AnalyzePGNRequest(BaseModel):
    pgn_text: str = Field(..., min_length=10, description="Raw PGN content (one or more games).")
    target_player: str | None = Field(default=None, description="Optional player filter for summary focus.")
    engine: EngineOptions = Field(default_factory=EngineOptions)
    detectors: list[str] | None = Field(
        default=None,
        description="List of registered detector names to run. Defaults to all default-enabled detectors.",
    )
    detector_params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-detector hyperparameter overrides, e.g. {'statistical_z_cusum': {'flag_z_threshold': 2.8}}.",
    )
    include_features: bool = Field(default=False, description="Include game-level engineered feature rows.")
    include_moves: bool = Field(default=False, description="Include ply-level Stockfish + clock rows.")


class AnalyzeFileRequest(BaseModel):
    pgn_path: str = Field(..., description="Local server path to a .pgn archive.")
    target_player: str | None = None
    engine: EngineOptions = Field(default_factory=EngineOptions)
    detectors: list[str] | None = None
    detector_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    include_features: bool = False
    include_moves: bool = False


class AnalysisResponse(BaseModel):
    games_parsed: int
    player_game_rows: int
    detectors_executed: list[str]
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed by detector name; each detector returns its own structured payload.",
    )
    features: list[dict[str, Any]] | None = None
    moves: list[dict[str, Any]] | None = None
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Open container for experimental signals, cohort percentiles, or metadata.",
    )


def _execute_pipeline(
    pgn_path: Path,
    target_player: str | None,
    engine_opts: EngineOptions,
    detector_names: list[str] | None,
    detector_params: dict[str, dict[str, Any]],
    include_features: bool,
    include_moves: bool,
) -> AnalysisResponse:
    raw_df = parse_pgn_for_detection(str(pgn_path))
    if raw_df.empty:
        raise ValueError("No valid games or moves found in the provided PGN.")

    eval_df = evaluate_games_df(
        raw_df,
        config=EngineConfig(
            depth=engine_opts.depth,
            shallow_depth=engine_opts.shallow_depth,
            multipv=engine_opts.multipv,
            workers=engine_opts.workers,
            use_cache=engine_opts.use_cache,
        ),
    )
    features_df = build_features(eval_df)
    if features_df.empty:
        raise ValueError("No games met the minimum evaluated move threshold.")

    selected = detector_names or [
        name for name, meta in DETECTOR_REGISTRY.items() if meta.get("default_enabled", True)
    ]
    unknown = [name for name in selected if name not in DETECTOR_REGISTRY]
    if unknown:
        raise KeyError(f"Unknown detector(s): {unknown}. Available: {list(DETECTOR_REGISTRY.keys())}")

    results: dict[str, Any] = {}
    for name in selected:
        fn = DETECTOR_REGISTRY[name]["fn"]
        params = detector_params.get(name, {})
        out = fn(features_df, eval_df, params)

        if target_player and isinstance(out, dict):
            tf = target_player.casefold()
            if "player_summary" in out:
                out["player_summary"] = [
                    r for r in out["player_summary"] if str(r.get("player", "")).casefold() == tf
                ]
            if "game_scores" in out:
                out["game_scores"] = [
                    r for r in out["game_scores"] if str(r.get("player", "")).casefold() == tf
                ]
        results[name] = out

    feat_records = (
        features_df.where(features_df.notna(), None).to_dict(orient="records")
        if include_features
        else None
    )
    move_records = (
        eval_df.where(eval_df.notna(), None).to_dict(orient="records")
        if include_moves
        else None
    )

    return AnalysisResponse(
        games_parsed=int(raw_df["game_id"].nunique()),
        player_game_rows=len(features_df),
        detectors_executed=selected,
        results=results,
        features=feat_records,
        moves=move_records,
        extensions={
            "engine_depth": engine_opts.depth,
            "shallow_depth": engine_opts.shallow_depth,
            "multipv": engine_opts.multipv,
            "target_player_filter": target_player,
        },
    )


@app.on_event("startup")
def _warm_calibration() -> None:
    """Pre-load the control-cohort calibration into memory on server startup."""
    load_calibration()


@app.get("/health")
def health() -> dict[str, Any]:
    cal = load_calibration()
    return {
        "status": "ok",
        "registered_detectors": list(DETECTOR_REGISTRY.keys()),
        "calibration_loaded": bool(cal.get("buckets") or cal.get("global_tc")),
        "clean_calibration_games": cal.get("n_clean_games", 0),
    }


@app.get("/detectors")
def list_detectors() -> list[dict[str, Any]]:
    """Inspect all available detection algorithms and whether they run by default."""
    return [
        {
            "name": meta["name"],
            "description": meta["description"],
            "default_enabled": meta["default_enabled"],
        }
        for meta in DETECTOR_REGISTRY.values()
    ]


@app.post("/analyze/pgn", response_model=AnalysisResponse)
async def analyze_pgn(req: AnalyzePGNRequest) -> AnalysisResponse:
    """Analyze raw PGN text and return results from all requested detection algorithms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", encoding="utf-8", delete=True) as tmp:
        tmp.write(req.pgn_text)
        tmp.flush()
        try:
            return await run_in_threadpool(
                _execute_pipeline,
                Path(tmp.name),
                req.target_player,
                req.engine,
                req.detectors,
                req.detector_params,
                req.include_features,
                req.include_moves,
            )
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/analyze/file", response_model=AnalysisResponse)
async def analyze_file(req: AnalyzeFileRequest) -> AnalysisResponse:
    """Analyze an existing local PGN file path (leveraging `.cache` for instant results)."""
    path = Path(req.pgn_path).expanduser().resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"PGN file not found: {path}")
    try:
        return await run_in_threadpool(
            _execute_pipeline,
            path,
            req.target_player,
            req.engine,
            req.detectors,
            req.detector_params,
            req.include_features,
            req.include_moves,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc