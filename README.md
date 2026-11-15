# Chess Fair-Play Detection

Behavioral screening for chess games. Stockfish evaluates every move, engineered
features summarize how a player performs under pressure, and behavioral detectors flag
exploratory deviations against personal history and a presumed-clean rating-matched cohort.

> **Research-only.** The current heuristic anomaly scores are **not** calibrated
> cheating probabilities or adjudications. A high score does not establish misconduct;
> a low score does not establish innocence. No independent account-level false-positive
> rate has been measured. Outputs have `null_p_value: null` and
> `calibration_status: uncalibrated` until an independently held-out account-level
> reference has been validated. The legacy `bayesian_sprt_ensemble` API name remains
> for compatibility, but its accumulated score is not a calibrated Bayesian posterior
> or validated sequential likelihood ratio.

## What it does

- **Move evaluation** — Stockfish MultiPV at shallow and full depth, with a persistent cache.
- **Behavioral features** — accuracy/CPL, critical and quiet-critical precision, Regan-style
  move-choice sensitivity, HMM toggling, clock pacing, phase and deficit/scramble shifts.
- **Detectors (exploratory only)** — `statistical_z_cusum`, `hmm_toggle_detector`, `bayesian_sprt_ensemble`, and
  `game_style_fingerprint` (RMS deviation from a player's prior style profile: accuracy,
  engine-match rate, tactical precision, volatility, blunder rate, opening profile).
- **HTTP API** — analyze raw PGN text or a server-side `.pgn` file. All four detectors run by default.

## Layout

```text
backend/
  api.py            FastAPI app
  benchmark.py      synthetic-injection benchmark
  src/
    detectors/      scoring, calibration, HMM, SPRT registry
    engine/         Stockfish evaluation and cache
    features/       game-level feature engineering
    ingestion/      PGN parsing and bot filtering
scripts/
  fetch_games.py    fetch public Chess.com games into PGN files
  build_baseline.py build an optional control-cohort calibration
examples/
  analyze_pgn.py    run the pipeline from Python (no server)
  sample_game.pgn
data/               generated calibration and move exports (gitignored)
```

## Requirements

- Python 3.12+
- Stockfish (for example `brew install stockfish`), found on `PATH` or via `STOCKFISH_PATH`
- `pip install -r requirements.txt`

## Get your own games

The repository ships one tiny sample PGN. Fetch your own public Chess.com games:

```bash
python scripts/fetch_games.py --user alice --out datasets/games.pgn
```

Useful options: `--months` (default 3, `0` = all archives), `--max-games` (default 100),
`--time-class rapid`. Only standard chess is fetched; bot games are filtered by the parser.

## Optional: control-cohort calibration

Cohort calibration compares a player against presumed-clean accounts of similar rating and
time control. Fetch a set of controls and build the baseline:

```bash
printf "control_user_1\ncontrol_user_2\n" > controls.txt
python scripts/fetch_games.py --users-file controls.txt \
  --out-dir datasets/controls --months 3 --max-games 50
python scripts/build_baseline.py --controls datasets/controls \
  --out data/baseline.json --workers 6
```

`build_baseline.py` runs Stockfish over the control games (slow on the first run, cached
afterwards). It writes `data/baseline.json`; roughly 15+ player-games per (time control,
100-Elo bucket) are needed for cohort statistics. Without that file the pipeline uses
Elo-based fallbacks.

To also create the fixture used by `backend/benchmark.py`, add
`--export-moves data/controls_moves.jsonl.gz`, then run `python backend/benchmark.py`.

## Run the API

```bash
cd backend
uvicorn api:app --reload
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/detectors
```

Interactive docs: <http://127.0.0.1:8000/docs>.

Analyze a PGN file that exists on the API host:

```bash
curl -X POST http://127.0.0.1:8000/analyze/file \
  -H 'Content-Type: application/json' \
  -d '{"pgn_path": "/path/to/games.pgn", "target_player": "alice"}'
```

Use `/analyze/pgn` with `pgn_text` to send PGN content instead of a path.

## Example without the API

```bash
python examples/analyze_pgn.py examples/sample_game.pgn
python examples/analyze_pgn.py datasets/games.pgn --player alice --depth 12 --json-out results.json
```

The first run evaluates uncached plies with Stockfish; later runs reuse
`backend/.cache/position_evals.json`. Pass `--no-cache` to bypass the cache.

## Calibration

`backend/src/detectors/calibration.py` loads `data/baseline.json` if it exists at API startup
or on the first scoring call. `GET /health` reports whether cohort tables loaded. Without the
file, scoring falls back to Elo-based curves.

## Notes

- `game_style_fingerprint` needs at least two games per player for a nonzero deviation. Its

