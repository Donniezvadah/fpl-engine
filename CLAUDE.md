# FPL Engine

AI-powered Fantasy Premier League decision-support platform. Goal: maximize expected
season points (target: top-100 overall). Local-first: SQLite, no Docker/Redis.

## Commands
- `uv sync` — install deps
- `uv run pytest` — run tests
- `uv run fpl --help` — CLI (update / backtest / recommend / plan / api)

## Architecture
- `src/fpl_engine/rules/` — pure-function FPL rules engine (scoring, squad legality,
  transfers, sell prices, autosubs, chips). Season-versioned constants in `constants.py`.
  Everything downstream (optimizer, simulator) must validate against this module.
- `src/fpl_engine/ingest/` — official FPL API client + vaastav historical CSV loader → SQLite.
- `src/fpl_engine/models/` — prediction layer. Every model outputs (mean, variance).
  Poisson-composed xP + LightGBM direct xP, blended by backtest-fit ensemble weights.
- `src/fpl_engine/optim/` — PuLP MILP: squad/XI/captain, multi-GW transfer planner, chip timing.
- `src/fpl_engine/sim/` — vectorized Monte Carlo over correlated scorelines.
- `src/fpl_engine/backtest/` — walk-forward replay of past seasons; the gate for model changes.
- `src/fpl_engine/recommend/` — assembles explained weekly recommendations.

## Conventions
- Prices are stored in 0.1m integer units (API convention: 55 == £5.5m). Never floats.
- Player positions: 1=GK, 2=DEF, 3=MID, 4=FWD (API element_type convention).
- All predictions carry uncertainty; no point estimates without variance.
- Rules constants are per-season (`SEASON_RULES` registry); check official FPL rules
  each July for changes before the new season. Chip windows mirror the API's
  `chips` array (start_event/stop_event), not a hand-written half-season split.
- The season label is derived from the date (`config.current_season`) and training
  seasons are auto-detected from the DB, so the engine rolls over each summer with
  no config edit. Override with `FPL_SEASON` if needed.
- Player identity across seasons is the API's stable `code`, never `player_id`
  (element ids are reassigned every season).
- Model changes must not merge unless the walk-forward backtest does not regress.
  Evaluate over GW1-38, not GW7+: changes aimed at early-season behaviour look
  falsely harmful if the gameweeks they target are excluded.
