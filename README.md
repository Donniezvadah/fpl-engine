# fpl-engine

AI-powered Fantasy Premier League decision-support platform. Local-first, evidence-based,
uncertainty-aware. Target: maximize expected season points.

## What it does

- **Data engine** — ingests the official FPL API (players, fixtures, per-fixture stats
  incl. xG/xA/defensive contributions) plus multi-season history from the community
  archive, into a local SQLite database.
- **Prediction models** — a minutes model (LightGBM: P(start), P(60+)), a Dixon-Coles
  team goal model (time-decayed), empirical-Bayes player rates, and a direct
  gradient-boosted points model, blended into per-player per-GW expected points with
  variance.
- **Walk-forward backtest** — replays past seasons using only information available at
  each deadline; model changes must not regress it.
- **MILP optimizer** — best legal squad/XI/captain (PuLP/CBC), a multi-gameweek transfer
  planner with free-transfer banking and hit EV, and chip-timing evaluation
  (wildcard / bench boost / triple captain / free hit).
- **Monte Carlo simulator** — correlated scoreline sampling for captaincy ceilings/floors,
  haul probabilities, and chip context.
- **Recommendation engine** — weekly report with transfers, captain, chips, differentials,
  price watch — every call with its evidence.

## Quick start

```bash
uv sync
uv run pytest                                  # rules engine + optimizer + sim tests

# data (current season from the official API; history archive optional)
uv run fpl update
uv run fpl backfill 2022-23 2023-24 2024-25
echo "FPL_TEAM_ID=1234567" > .env             # from your Points-page URL
uv run fpl sync-team

# validate the models
uv run fpl backtest --season 2025-26 --train-season 2023-24 --train-season 2024-25

# weekly decisions
uv run fpl recommend
uv run fpl api                                 # JSON API for the local dashboard
```

## The dashboard

A GitHub Actions workflow rebuilds the site twice a week and publishes it to
GitHub Pages, so the current gameweek's numbers are always a URL away. The page
is a **static export** — the engine bakes its output into a single `data.json`
and every filter, sort and re-ranking runs in the browser, so there is no server
to pay for or keep alive.

The player explorer is the point of it: filter by position, price, ownership and
start probability, then re-rank the whole league over any horizon from one to six
gameweeks, sorted by expected points or by points per £m.

Build it locally:

```bash
uv run fpl export                              # writes web/public/data/data.json
npm ci --prefix web && npm run build --prefix web
python3 -m http.server 4321 --directory web/out
```

To point the published site at your own squad instead of a from-scratch draft,
set an `FPL_TEAM_ID` repository variable — the workflow picks it up and switches
to linked mode once a deadline has passed.

## My Team

With `FPL_TEAM_ID` set, the dashboard opens on a **My Team** tab:

- a pitch view of your synced squad (formation, captain, bench, injury flags, next opponent);
- the Wildcard rebuild the engine recommends, as a second pitch plus an out → in table;
- transfer suggestions that are scored on the whole team (best XI + captain), not player
  by player, so bench-only upgrades don't look like big wins;
- an editor: tap a player to swap him, make him captain, or press **Use this as my team**
  on the Wildcard squad. Budget, 3-per-club and position rules are checked as you go.
  Edits are saved in your browser only and are never uploaded.

Refresh everything in one step (data, team sync, site data):

```bash
uv run fpl refresh
```

## Design principles

1. **Never guess, always compute** — every recommendation traces to model outputs.
2. **Rules are law** — a season-versioned rules engine validates every emitted squad
   independently of the optimizer.
3. **Uncertainty first** — all predictions carry variance; the simulator models the
   correlations that matter (teammates, clean sheets, captaincy).
4. **Backtest is the gate** — walk-forward replay decides ensemble weights and vetoes
   regressions.
