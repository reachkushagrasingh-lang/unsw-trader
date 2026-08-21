# Prompt logs — Project B (z5632343)


## Prompt log — sentiment look-ahead (lagging the signal)

### What I wanted
A daily news-sentiment signal I could fuse into the equity fund without leaking future
information into a trading decision.

### Prompt(s)
"Build a daily sector sentiment index from the scored headlines and join it to the
returns so I can tilt the fund by sentiment."

### The risk I designed against
A headline dated day t is only known after the close, so tilting day-t weights by day-t
sentiment trades on information the fund couldn't have had. I required a lag in CLAUDE.md
up front.

### How I verified it
The fusion consumes `sentiment.lag_signal(index_df, lag=1)` only, so day t sees t−1 and
earlier. I confirmed the signal moves after a price move, not before.

---

## Prompt log — finance lexicon (merge, not replace)

### What I wanted
Lift finance headlines out of false-neutral — VADER under-reads market language like
"miss", "beat", "downgrade" — without breaking its scoring of everything else.

### Prompt(s)
"Add a finance sentiment lexicon to VADER so market headlines aren't scored neutral."

### The risk I designed against
Replacing VADER's base lexicon collapses any headline without a finance keyword to 0. I
required merge-not-replace up front.

### How I verified it
`sentiment._analyzer` uses `sia.lexicon.update(...)`, and `scripts/innovation_lexicon.py`
proves the boost cut the neutral rate (49.6%→46.8%) with rescued examples.

---

## Prompt log — crypto on the equity calendar

### What I wanted
A combined equity+crypto fund whose returns line up on the calendar the fund actually
trades on.

### Prompt(s)
"Combine the equity and crypto returns into one panel for the combined funds."

### What I built
A combined returns panel joining equity returns with crypto returns.

### The risk I designed against
Crypto trades 365 days; equities don't. Keeping Saturday/Sunday crypto rows means the
combined fund would "act" on weekend moves it could never trade on an equity day, and it
mixes two annualisation bases (365 vs 252). I required calendar alignment up front.

### How I verified it
`features.combined_returns_panel` computes crypto returns on their own 365 calendar first,
then reindexes onto the equity trading days and left-joins — weekend-only moves are
dropped. Crypto-only funds keep `periods_per_year=365`; equity/combined use 252.

---

## Prompt log — walk-forward backtest (no in-sample weights)

### What I wanted
An out-of-sample backtest for the four optimisers where weights only use past data.

### Prompt(s)
"Backtest the min-variance / max-Sharpe / risk-parity / equal-weight funds and report
Sharpe, annual return and max drawdown."

### What I built
A walk-forward backtest across the four optimisers.

### The risk I designed against
Estimating the covariance and weights on the full return history is in-sample look-ahead —
Sharpe is inflated and the numbers aren't a fair OOS result. I required trailing-window
estimation up front.

### How I verified it
`portfolios.oos_backtest` estimates weights from a trailing 252-day window of past returns
at each monthly rebalance, held until the next, and drops returns before `first_live_date`.
I confirmed no fund holds weights dated before its first live date.

---

## Prompt log — deploy: app must not recompute

### What I wanted
The Streamlit app to load fast and deploy cleanly on share.streamlit.io.

### Prompt(s)
"Make the Streamlit app show the funds, the sentiment index and the fusion comparison."

### What I built
A Streamlit app presenting the funds, the sentiment index and the fusion comparison.

### The risk I designed against
Running the backtest and VADER (with an nltk download) inside the deployed app is slow and
fragile on the Cloud box — the download can fail behind the sandbox and every visitor
re-runs heavy compute. I required a build-vs-serve split up front.

### How I verified it
`scripts/run_part_b.py` precomputes everything to `results/data/`, `results/tables/` and
`results/figures/`; `streamlit_app.py` only reads those artefacts — no backtest, no VADER,
no network at serve time. I confirmed the app runs on a clean checkout.

---

## Bug fixed — SSL / VADER lexicon wouldn't download

### What went wrong
`nltk.download('vader_lexicon')` failed with `CERTIFICATE_VERIFY_FAILED` and failed
silently, then crashed later with a `LookupError` when the analyzer couldn't find the
lexicon it thought had downloaded.

### How I fixed and verified it
Pointed Python's SSL context at certifi's CA bundle so the download completes; confirmed
the sentiment index then built end to end.

### What I learned
A silent failure is worse than a loud one — the "successful" download that left nothing on
disk is what turned a cert problem into a baffling `LookupError` later, so I now confirm a
dependency actually landed rather than trusting the call returned cleanly.

---

## Bug fixed — deployed app rendered a black screen

### What went wrong
The app booted but rendered blank. Funds with staggered start dates left NaNs that broke
the allocation blend on load.

### How I fixed and verified it
Made the blend NaN-safe; verified by clicking through all four tabs on the deployed app,
not just locally.

### What I learned
A deployed app can boot and still show nothing — NaNs from staggered start dates have to be
handled at the blend step rather than assumed away, and the only real test is the live app,
not my machine.

---

## Bug fixed — missing required figure (weights over time)

### What went wrong
Checking my outputs against the brief's exhibit list, the weights-over-time figure wasn't
being saved — the script was keeping only the last row of weights.

### How I fixed and verified it
Kept the full weight history and drew a stacked-area chart; confirmed
`results/figures/weights_over_time.png` regenerates on a clean run.

### What I learned
Checking output against the brief's exhibit list, not my own memory, is what caught a
required figure I'd silently dropped.

---

## Bug fixed — `.venv` committed into the repo

### What went wrong
`scripts/check_handin.py` flagged parquet files sitting inside `.venv` that had been
committed.

### How I fixed and verified it
Added `.venv/` to `.gitignore` and ran `git rm --cached` to untrack it; re-ran
`check_handin.py` clean.

### What I learned
An automated hand-in check catches what I'd miss by eye — a virtualenv full of data files
slipping into the repo — so I run `check_handin.py` before zipping, not after.