# OpenChessAnticheat

Run a local chess fair-play screen on a PGN archive. No login, web service,
manual calibration, model training, or ML runtime is required.

```bash
# Install once (macOS)
brew install stockfish
python3 -m pip install -r requirements-cli.txt

# Inspect a player's games
python3 examples/analyze_pgn.py data/personal/games.pgn \
  --player tyrjung --json-out results.json
```

The default output gives an account assessment and ranked **five-move
human-adjusted engine-choice sequences** in 3+0, 3+2, 5+0 or 5+3 blitz.
A small bundled JSON model estimates which engine moves comparable human
players would tend to choose in each position. The application contrasts
actual sustained engine-first decisions with that human baseline.
Other time controls (including 1+0 bullet and 10+0 rapid) continue using
the existing Stockfish screening rather than extrapolated model scores.

Technical game scores and individual moves are still available with
`--technical`. Disable the human-prior ranking with `--no-human-prior`.
No percentage is a calibrated cheating probability, and any ranking is
a lead for chess review—not a determination of assistance.

**Evidence for the bundled experimental model:** trained on 70,000 genuine
human decisions, tested against ChessFraud assisted-play annotations. On two
player-disjoint controlled-tournament cohorts, its five-move sequence AUC was
0.694 vs 0.610 and 0.711 vs 0.643 for simple Stockfish agreement. The
second-cohort follow-up was exploratory, and neither cohort measures the
false-accusation rate on public Chess.com accounts. The model is experimental.
The original strict move-level gain gate did not pass.

## Advanced components

The remaining sections document the older exploratory detectors, optional
calibration scripts and reproducible research utilities. None is required
for the one-command PGN workflow.

## Design principles

- **No automated verdicts.** Game and player outputs are exploratory tiers, not accusations.
  `null_p_value` remains empty until an independent, held-out account-level reference is
  configured.
- **Calibration is explicit.** No `cheat_probability` is emitted. An empirical account-level tail
  requires at least 99 independent null accounts; resolving a 0.1% tail requires 999+.
- **Explainable by layer.** Each detector reports its own scores, thresholds, and trigger reasons,
  so a reviewer can see *why* a case was surfaced.
- **Conservative by default.** Missing clock data, thin history, sparse cohorts, and small control
  sets produce weaker evidence rather than invented precision.
- **Reproducible and privacy-aware.** The engine cache, feature pipeline, safety tests, and data
  tooling are in the repository; fetched games, control archives, and generated calibration files
  stay local.

## Project layout

- `backend/` — FastAPI service, scoring pipeline, and research safeguards
- `backend/src/engine/` — Stockfish MultiPV evaluation and persistent cache
- `backend/src/features/` — game-level behavioral feature engineering
- `backend/src/detectors/` — Z/CUSUM/SPRT scoring, HMM regime detection, style fingerprints,
  calibration, and validation utilities
- `backend/src/ingestion/` — PGN parsing and bot filtering
- `examples/analyze_pgn.py` — human-readable CLI report without the API server
- `scripts/fetch_games.py` — fetch public Chess.com games into PGN files
- `scripts/build_baseline.py` — build an optional control-cohort calibration
- `tests/` — safety and contract regression tests
- `.github/workflows/safety-tests.yml` — scoring-safety test workflow
- `data/` — local-only games, controls, caches, and generated calibration (gitignored)
- `LICENSE` — MIT

## What's implemented

**Analysis engine**
- Multi-depth Stockfish evaluation (MultiPV, shallow/full snapshots) with a persistent FEN/move cache
- Behavioral features: accuracy/CPL, critical and quiet-critical precision, Regan-style move-choice
  sensitivity, timing/pacing, phase shifts, deficit and scramble behavior, and style-profile metrics
- Four detectors: `statistical_z_cusum`, `hmm_toggle_detector`, `bayesian_sprt_ensemble`, and
  `game_style_fingerprint`
- Game-level and player-level reports with per-signal Z-scores, CUSUM/SPRT summaries, and trigger
  descriptions

**API**
- `GET /health`, `GET /detectors`
- `POST /analyze/pgn` for raw PGN text
- `POST /analyze/file` for a server-side `.pgn` path
- Structured detector payloads plus optional feature/move rows

**Validation safeguards**
- `backend/src/detectors/validation.py` provides account-level empirical-tail utilities and a
  leakage/label-manifest validator
- Outputs explicitly carry `null_p_value`, `calibration_status`, and `calibration_note`
- Safety regression tests reject cheat probabilities, duplicate/self calibration controls, and
  cross-split account leakage

**Data tooling**
- Chess.com public-game fetch for a player or a control-account list
- Control-cohort baseline builder with optional move export for `backend/benchmark.py`
- Synthetic engine-injection benchmark for sanity checking, not external validation

## Current limitations

- **No probability of misconduct is emitted.** The project does not estimate
  `P(misconduct | data)`. The legacy `bayesian_sprt_ensemble` name is retained for compatibility,
  but its score is not a Bayesian posterior or a validated sequential likelihood ratio.
- **Account-level calibration is in progress.** Control accounts are presumed typical, not
  verified. An empirical tail needs at least 99 independent accounts; 0.1% resolution needs 999+.
  No qualified cohort is shipped yet.
- **External validation is pending.** No held-out adjudicated cases, false-positive counts with
  intervals, recall at fixed FPR, Brier score, or calibration error are reported yet. The synthetic
  injection benchmark is a robustness check, not proof of accuracy.
- **No UI, auth, persistence, or audit workflow.** The API is stateless; it produces analysis
  payloads only.
- **Legacy output names remain.** `account_verdict` and `flagged_*` are retained for compatibility;
  read them as heuristic screening tiers, not findings.
- **Window-length comparability.** Historical per-game maxima depend on how many games an
  account has; compare accounts only at matched window lengths.
- **Data licensing/privacy is the operator’s responsibility.** Public game archives may still be
  sensitive; do not republish third-party data.
- **Safety workflow coverage is incomplete.** The workflow currently targets the feature branch and
  pull requests; broaden it to `main` before relying on it as a merge gate.

## Setup

Requirements: Python 3.12+, Stockfish, and the packages in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS example; install Stockfish however you prefer
brew install stockfish
# or: export STOCKFISH_PATH=/path/to/stockfish
```

Run the safety tests:

```bash
python -m unittest discover -s tests -v
```

## Quickstart

### CLI report (no server)

```bash
python examples/analyze_pgn.py examples/sample_game.pgn --depth 2
python examples/analyze_pgn.py data/games.pgn --player alice
```

The report leads with the screening summary, shows the highest-priority players, and prints the
calibration caveat once. `--json-out results.json` writes the full rows; `--top`, `--top-games`, and
`--no-color` control the presentation.

Abridged example:

```text
Chess fair-play screening
────────────────────────────────────────────
  File        examples/sample_game.pgn
  Games       1   Player-game rows 2   Players 2
  Calibration no local baseline

alice  ·  rapid  ·  No outlier detected
  ──────────────────────────────────────────
  Screening       No outlier detected
  Games           1   Mean ACPL 14.4   Mean Elo 1600
  Max signal      SPRT 2.90   CUSUM 1.67   Z 2.17
  Outlier games   0 / 1 (0.0%)
  Empirical null  not configured

Games crossing the exploratory tiers
  No individual games crossed the exploratory tiers.

Heuristic tiers only; not a cheating determination.
```

### API

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

## Detectors

| Detector | What it examines | Output |
|---|---|---|
| `statistical_z_cusum` | Seven behavioral signals standardized against personal history and a rating/time-control cohort | Game tiers, composite Z, CUSUM, SPRT summary |
| `hmm_toggle_detector` | Two-state HMM for within-game shifts between human-like and engine-like move regimes | Engine-like fraction, longest streak, path LLR |
| `bayesian_sprt_ensemble` | SPRT-style fusion across surprise, HMM, and meta-history signals (legacy name) | Heuristic account tier |
| `game_style_fingerprint` | Deviation from a player’s prior style profile | RMS Z-score and per-game style metrics |

## API response semantics

| Field | Meaning |
|---|---|
| `status` | Game-level: `exploratory_outlier`, `exploratory_review`, `no_outlier_detected`, `untimed_excluded` |
| `account_verdict` | Player-level heuristic tier: `manual_review_recommended`, `statistical_outlier`, `no_outlier_detected` |
| `composite_z` | Weighted combination of available signal Z-scores |
| `max_sprt_llr` | Heuristic sequential accumulation; **not** a validated likelihood ratio |
| `null_p_value` | `null` until an independent account-level null is available |
| `calibration_status` | Machine-readable state: `cohort_baseline` when `data/baseline.json` has cohort statistics for the analyzed time control; `empirical_null` for an account-level reference |
| `flagged_games` / `flagged_pct` | Legacy names for games that crossed the game-level tier |

## Data and calibration

Fetch your own public Chess.com games:

```bash
python scripts/fetch_games.py --user alice --out data/games.pgn
```

Fetch control accounts (one file per account) and build the optional baseline:

```bash
printf "control_user_1\ncontrol_user_2\n" > controls.txt
python scripts/fetch_games.py --users-file controls.txt \
  --out-dir data/controls --months 3 --max-games 50
python scripts/build_baseline.py --controls data/controls \
  --out data/baseline.json --workers 6
```

The baseline is optional; without it, scoring falls back to Elo curves. Control accounts are treated
as presumed typical, not adjudicated. `build_baseline.py` needs roughly 15+ player-games per
(time control, 100-Elo bucket) for cohort statistics and warns when the cohort is too sparse.
Coverage is per time control: a baseline built only from blitz games leaves rapid and daily on the
Elo fallback, and `calibration_status` stays `uncalibrated` for those rows.
Add `--export-moves data/controls_moves.jsonl.gz` to create the fixture consumed by
`backend/benchmark.py`.

`backend/src/detectors/calibration.py` loads `data/baseline.json` if present at API startup or on
the first scoring call; `GET /health` reports whether cohort tables loaded.



## Experimental pretrained assistance classifier (opt-in)

A compact model is trained on [ChessFraud's controlled-assistance tournament](https://huggingface.co/datasets/artemlepin/chess-fraud).
The **published JSON weights, when present**, are generated by GitHub Actions
from a pinned upstream dataset revision; clone users **never need to train or
calibrate anything**. Only eligible **300+0 five-minute blitz** games are scored,
because that is the benchmark's sole time control.

```bash
# Normal, recommended PGN analysis — no classifier or calibration
python examples/analyze_pgn.py data/personal/games.pgn --player tyrjung

# Opt in to the experimental pretrained model if the weights are shipped
python examples/analyze_pgn.py data/personal/games.pgn --player tyrjung \\
  --experimental-model --json-out data/personal/results.json
```

The model scores individual eligible games on a 0–100 **benchmark-relative**
scale. It **does not** output the chance that an arbitrary Chess.com/Lichess
user cheated, nor does it determine innocence. Different time controls,
insufficient game history, and games with fewer than 12 qualifying moves
do not get a pretrained score.

Training/evaluation: `scripts/train_chessfraud_model.py`.
GitHub Actions selects between logistic regression and a shallow tree ensemble
on a player-disjoint validation set, evaluates against a plain
Stockfish-best-move agreement baseline on held-out players, then publishes
source hashes, evaluation metrics, and experimental weights only if the model
meets the experimental minimum AUC. The stricter primary product gate
requires a further improvement of 0.02 AUC over the simple baseline.

**Verified initial results (49 players, 941 eligible player-games):** selected
logistic model test ROC AUC 0.6800 on 183 player-games from 10 held-out
players; simple engine-agreement AUC 0.6696. Its incremental detection benefit
is small, so this is not the default screening assessment.

No GitHub API key or Hugging Face account is required to *use* the bundled
weights. To reproduce training locally, install `scikit-learn` and
`datasets`, then run the script. The repository's regular runtime does not
require either package.

## License

MIT — see [LICENSE](LICENSE).

## Calibration is optional — everyday use never downloads a dataset

The default command is the entire user workflow:

```bash
python examples/analyze_pgn.py my_games.pgn --player alice
```

It works with Stockfish and the supplied PGN, without creating or providing a
reference dataset. Results are heuristic rankings (not a probability of cheating).

**Optional one-time empirical reference**: use the public-account collection,
scoring and reference scripts documented above, or provide a previously frozen
`openchess-proxy-null-v1` JSON reference. The frozen file can be reused across
runs; it is NOT recalculated per PGN.

```bash
python examples/analyze_pgn.py my_games.pgn --player alice \\
  --reference data/references/my-first-reference.json \\
  --platform chesscom --json-out results.json
```

The reference must match **platform, time control, 200-Elo rating bucket and
exact number of games**. Otherwise the optional comparison reports why it
cannot produce a percentile. For an existing default 30-game reference,
provide exactly 30 eligible same-time-control games for the player; otherwise
omit the reference flag and keep the ordinary analysis.

A displayed `proxy_tail_p_value` describes extremeness among the publicly
**unconfirmed** accounts used to build the reference, not proven-clean accounts.
It is NOT a probability that the player is cheating, nor a validated false
positive rate. The independent ChessFraud dataset can be used for evaluating
assistance detection once its benchmark adapter is completed; downloading it
does not automatically calibrate this detector.

The code intentionally does not require (or silently load) the optional
account-level reference. Third-party account datasets are not shipped with the
repository.
