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
  `maia_likelihood` metric stays 0 unless the evaluation data includes a `maia_probability` column.


## Independent validation required before fair-play decisions

The shipped control cohort is *presumed* clean, not verified innocent. A platform
account closure for fair-play reasons supports the label "platform action taken",
not proof that each game contains assistance. Archive account status, source URL,
retrieval timestamp, platform and the game date range. Never treat unclosed accounts
as verified negatives, and never assume an account's date of closure can be inferred
from its current public API status.

`backend/src/detectors/validation.py` provides two opt-in utilities:

- `validate_label_manifest` rejects contradictory labels and accounts appearing
  across train, calibration and test partitions.
- `empirical_account_tail` ranks **one account score** among distinct, held-out,
  same-time-control null account scores; it excludes the queried account, uses a
  conservative add-one rank, and refuses to emit a p-value for fewer than 99 controls.
  It is **not** wired into the public detector by default because the repository
  does not yet ship an independently qualified calibration cohort.

A 0.1% one-sided tail requires at least **999 independent null accounts** even
to have numerical resolution (1/(N+1) <= 0.001); adequate confidence in the
operating false-positive rate requires substantially more. Match evaluation
window length/game count and rating/time-control strata before using ranks.
Historical maximum-over-games statistics vary with the number of games and
cannot be compared fairly across arbitrary window sizes. Freeze feature/scoring
code and account partitions **before** constructing the evaluation cohort.

Before any probability calibration, evaluate account-disjoint held-out cases and
report false-positive counts/denominators with intervals, recall at preselected
FPR thresholds, Brier score, and calibration error. An estimated platform-action
probability also depends on the cohort's prevalence and label-selection process;
it is not automatically the probability of cheating.

Run safety regression tests with `python -m unittest discover -s tests -v`.

## Public-label dataset quickstart (no API key)

For a **reproducible evidence snapshot** rather than hand-curated cheating claims:

```bash
# From repo root, with requirements installed:
python scripts/collect_public_labels.py collect \\
  --seed-user YOUR_CHESSCOM_USERNAME --platform chesscom \\
  --out data/snapshots/my-first-snapshot \\
  --user-agent "OpenChessAnticheat-research (contact: you@example.com)" \\
  --fetch-games --max-accounts 200 --max-games 50

# Offline audit checks stored identity/split and PGN byte checksums:
python scripts/collect_public_labels.py verify \\
  --out data/snapshots/my-first-snapshot
```

Alternatively supply a UTF-8 CSV with the columns `platform,player` (platform:
`chesscom` or `lichess`) using `--accounts accounts.csv`. \`--seed-user\` downloads public games from that account, then selects players visible in them. An existing PGN can also be supplied with \`--seed-pgn path/to/games.pgn --platform chesscom\`. A PGN seed selects\naccounts already visible in those games; it does not guarantee any fair-play
closures. Collection is serial to respect public API rate limits, retries
transient errors, and records failures in `manifest.json`. A partially failed
run is **not** a complete corpus. Each output directory is immutable: use a
new directory to collect again. Snapshots are intentionally gitignored by
the existing `data/` ignore rule. Do not upload account-level accusations.

Account label semantics:
- Chess.com `status=closed:fair_play_violations` becomes **1**,
  documented public platform fair-play closure.
- Lichess `tosViolation=true` becomes **1** but labelled generic
  platform terms-of-service action; it is not verified to mean fair-play
  assistance specifically.
- All other statuses become **0 = unconfirmed**, NOT verified innocent.
  Other account closure reasons are not positive fair-play examples.

**Do not train or calibrate a cheating probability from this snapshot as-is.**
It supplies traceable weak labels and PGNs; it does not establish the actual
violation dates or which pre-closure games were assisted. Chess.com game exports
retrieve up to the three latest month archives; Lichess exports the latest
rated blitz/rapid games. Use a frozen acquisition cutoff and documented
selection rules for any benchmark. In particular, avoid using games
recorded after a closure or from widely differing time-control/game-count strata.

### What must be pinned for a reproducible calibration

Persist the snapshot `manifest.json` and PGNs, Git commit, Python and package
versions, Stockfish binary/version and configuration (depth, nodes, threads,
MultiPV), game-selection window, clean-reference account IDs, scoring and
threshold code version, train/calibration/test account partitions, and all
computed account scores. Re-running against **live** player statuses is a new
snapshot, not a reproduction of the original one. Generate per-account reports
with `python examples/analyze_pgn.py path/to/account.pgn --player ACCOUNT`
once an account has enough eligible games.

The current empirical-tail utility deliberately requires an independent
account-score table and will return no p-value for fewer than 99 controls.
It is **not automatically enabled**, because no audited, stratified control
set or account-level false-positive study has yet been validated.

Public sources: https://support.chess.com/en/articles/9650547-what-is-the-pubapi-and-how-do-i-use-it
and https://lichess-org.github.io/api/.

### Freeze account windows and build an empirical reference

After collecting the snapshot, run the existing engine pipeline once per qualifying
account and time-control stratum. This requires Stockfish, and will take substantial
CPU time for a large corpus:

```bash
python scripts/score_public_snapshot.py \\
  --snapshot data/snapshots/my-first-snapshot \\
  --out data/scored/my-first-run \\
  --depth 12 --games-per-time-control 30 --workers 2

python scripts/build_empirical_reference.py \\
  --scores data/scored/my-first-run/score_manifest.json \\
  --out data/references/my-first-reference.json \\
  --games 30 --elo-width 200 --min-controls 99
```

The scoring stage freezes the latest **30 eligible games in each time-control
category** for every player; insufficient histories are excluded. It writes
the exact subset PGNs and model reports, plus engine/version/settings and code
hashes for replay. The reference stage uses only the account-disjoint
`calibration` split and reports empirical ranks for the untouched `test`
split; accounts are matched by platform, time control, 200-Elo bucket, and
exact game count. It refuses incomplete scoring runs and skips reference
strata with fewer than 99 accounts.

**This is a proxy reference, not a verified clean-account null.** Its
`proxy_tail_p_value` answers only how extreme a score is relative to the
sample of publicly *unconfirmed* accounts in that stratum. It does not
establish a genuine fair-play false-positive rate. It is never automatically
used to declare guilt. To resolve a 0.1% tail in a stratum requires at least
999 independent reference accounts; the default hash split assigns only
20% of discovered accounts to calibration, so a much larger input pool is
usually necessary. This is a population-size problem, not an optimizer bug.
