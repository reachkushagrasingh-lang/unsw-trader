# AGENTS.md - CLAUDE

## What this is
A FinTech pipeline for FINS3645 Part B. It ingests equity and crypto price panels plus a
news feed, builds a family of optimised funds with a walk-forward backtest, computes a
news-sentiment index, and fuses lagged sentiment into the equity fund as an extension.
Results are precomputed to CSV/PNG and served read-only by a Streamlit app.

Data sources and schema are defined in `context/DATA_GUIDE.md` — that file is the source
of truth for column names, tickers, sectors, and date coverage. Do not guess a field.

GITHUB - https://github.com/reachkushagrasingh-lang/unsw-trader/blob/main/results/data/fund_returns.csv
STREAMLIT APP- https://reachkushagrasingh-lang-unsw-trader-streamlit-app-c9vx6t.streamlit.app/

## Architecture (build vs serve — this split is load-bearing)
- **Build** (`scripts/run_part_b.py`, `scripts/innovation_lexicon.py`): does all heavy
  work — backtests, VADER, figures — and writes `results/data/*.csv`,
  `results/tables/*.csv`, `results/figures/*.png`.
- **Serve** (`streamlit_app.py`): only **reads** those artefacts. It never runs a
  backtest and never runs VADER. VADER's `nltk.download('vader_lexicon')` is a build-time
  step and must never be triggered from the deployed app.

`src/` modules by station: `etl.py` (load/clean), `features.py` (returns, headline panel),
`portfolios.py` (4 optimisers + OOS backtest), `sentiment.py` (VADER + sector index),
`fusion.py` (sentiment tilt), `viz.py` (styling). Prompt logs live in `ai/`.

## Coding conventions
- Python 3.13, pandas + DuckDB for data, scipy `SLSQP` for the optimisers, matplotlib via
  `viz.apply_style()`. Keep imports at top; `# noqa: E402` only for the `sys.path` shim in
  scripts.
- Pure DataFrame-in/DataFrame-out transforms in `src/`; scripts orchestrate and do I/O.
- Relative paths off `ROOT` only — no absolute Mac paths (it has to run on Streamlit Cloud).
- Annualisation is a parameter, not a constant: 252 for equity/combined, 365 for crypto.

## Rules you must follow
1. **No look-ahead — anywhere.** Backtest weights come from a trailing window of *past*
   returns only, held to the next rebalance (`portfolios.oos_backtest`). Sentiment is
   lagged (`sentiment.lag_signal`, `lag=1`) before it can influence a trade. Never join
   same-day news onto a same-day trading decision — it's only known after the close.
2. **Returns = `adjClose` + `.pct_change()`**, pivoted wide, first row per ticker dropped
   (`features.daily_returns`). Never raw `close`, never `.diff()`.
3. **Crypto is aligned onto the equity calendar** (`features.combined_returns_panel`):
   compute crypto returns on its own 365-day calendar, then reindex to equity days.
   Weekend-only crypto moves are intentionally dropped — a fund trading on equity days
   couldn't act on them.
4. **Finance lexicon merges, never replaces** (`sentiment._analyzer` uses
   `sia.lexicon.update(...)`). Replacing VADER's base lexicon would destroy its general
   scoring; we only *augment* it for finance terms.
5. **Never invent a column, ticker, citation, or number** (see `context/verify_ai_output.md`
   — this is the KPMG/EY hallucination rule). If a field isn't in the data guide, stop and
   ask. If a method might not exist, say so. Show your working for any number you produce.
6. **You write code; I write the graded prose.** The report interpretation in `report/`
   is mine, in my own words. You can draft figures and list findings I react to, not the
   analysis I submit.

## How I check and correct your output
- Re-run `python scripts/run_part_b.py` end to end; confirm all CSVs/figures regenerate.
- Spot-check one return by hand (`adjClose` t vs t−1) against `features.daily_returns`.
- Confirm first row per ticker is NaN/dropped, and that no fund has weights before its
  `first_live_date`.
- Eyeball each new signal against dates for look-ahead.
- Confirm `streamlit_app.py` runs on a clean checkout with no network/VADER calls.
- Every non-trivial assist gets a prompt-log entry in `ai/`.
