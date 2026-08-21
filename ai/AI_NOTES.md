# AI_NOTES — how I directed and checked the assistant

## How I used AI
I used Claude Code as a coding assistant, not an author. I put my project rules in
`CLAUDE.md` (no look-ahead, `adjClose`+`pct_change`, lexicon merges not replaces, app reads
precomputed results, never invent a column or number) and had it draft functions station by
station: ETL, features/returns, the four optimisers and the walk-forward backtest, the VADER
sentiment index, the sentiment-fusion extension, and the figures.

## Where it went wrong and how I caught it
The prompt logs in this folder record the specific catches. The recurring themes were
**look-ahead** (it wanted to use same-day sentiment and full-sample weights — I lagged the
signal and made the backtest walk-forward), **a destructive shortcut** (it replaced VADER's
lexicon instead of merging. I used `.update()` and then wrote an ablation script to prove
the finance boost actually reduced false-neutrals), **calendar mismatch** (crypto weekend
rows leaking into the equity-day fund — I reindexed onto the trading calendar), and a
**deploy trap** (recomputing VADER/backtests inside the Streamlit app — I split build from
serve).

## How I verified before trusting anything
Following `context/verify_ai_output.md`: I re-ran `python scripts/run_part_b.py` end to end
and confirmed every CSV and figure regenerated; I hand-checked one daily return against
`adjClose`; I confirmed the first return per ticker drops out and no fund holds weights
before its first live date; I eyeballed each new signal against dates for look-ahead; and I
confirmed the deployed app runs on a clean checkout with no VADER or network calls. No number
went into the report because "the AI said so", each traces to a computation I can re-run.

## What was mine vs the AI's
The architecture decisions (build-vs-serve split, lagging, the annualisation-by-calendar
choice, the innovation lexicon ablation as evidence) and all report interpretation are mine.
The AI accelerated the boilerplate and first drafts of the functions, which I then corrected
against the rules above.