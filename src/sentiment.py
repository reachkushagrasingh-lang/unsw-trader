"""Station 3 - sentiment model and sector index from news headlines.

VADER scores each headline; we average to a ticker-day score, then to an
equal-weight sector index. The signal is LAGGED before it is ever used to trade
(day t uses sentiment from t-1 or earlier).

VADER needs a one-time nltk.download('vader_lexicon'). That is a BUILD step
(run_part_b.py / this module), never the deployed app - the app reads the
precomputed results/data/sector_sentiment_index.csv instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Optional finance-lexicon boost (innovation hook). VADER under-reads finance
# words; a small domain lexicon reduces false neutrals. Extend this yourself and
# log the change - that is exactly the kind of edit the innovation band rewards.
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "surge": 2.5, "soars": 3.0, "soar": 3.0,
    "rally": 2.0, "upgrade": 2.0, "upgraded": 2.0, "outperform": 2.0,
    "record": 1.5, "profit": 1.5, "bullish": 2.5, "jumps": 2.0,
    "miss": -2.0, "misses": -2.0, "plunge": -3.0, "plunges": -3.0,
    "slump": -2.5, "downgrade": -2.0, "downgraded": -2.0, "bearish": -2.5,
    "lawsuit": -1.5, "probe": -1.5, "recall": -2.0, "bankruptcy": -3.5,
    "cut": -1.0, "warns": -2.0, "warning": -1.5, "loss": -1.5, "slashes": -2.0,
}


def _analyzer(use_finance_lexicon: bool = True):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    if use_finance_lexicon:
        sia.lexicon.update(FINANCE_LEXICON)     # merge, don't replace
    return sia


def score_headlines(panel: pd.DataFrame, use_finance_lexicon: bool = True) -> pd.DataFrame:
    """Score each headline's compound sentiment, average to ticker-day.

    Input : long panel [trading_day, ticker, sector, title] (from features).
    Output: [trading_day, ticker, sector, sentiment, n_headlines].
    """
    sia = _analyzer(use_finance_lexicon)
    scores = panel["title"].map(lambda t: sia.polarity_scores(t)["compound"])
    df = panel.assign(sentiment=scores)
    agg = (df.groupby(["trading_day", "ticker", "sector"])
             .agg(sentiment=("sentiment", "mean"),
                  n_headlines=("sentiment", "size"))
             .reset_index())
    return agg


def sector_sentiment_index(scores: pd.DataFrame, trading_days: pd.DatetimeIndex,
                           fill: str = "ffill") -> pd.DataFrame:
    """Daily equal-weight (across tickers) sentiment per sector, on the trading grid.

    fill: how to treat sector-days with no headlines:
      'ffill'   -> carry the last observed value forward (persistence assumption)
      'neutral' -> 0.0 (no-news = neutral)
      'drop'    -> leave NaN
    Returned wide: index = trading_day, columns = sectors.
    """
    daily = (scores.groupby(["trading_day", "sector"])["sentiment"]
                   .mean().unstack("sector").sort_index())
    grid = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_days).unique()))
    daily = daily.reindex(grid)
    if fill == "ffill":
        daily = daily.ffill()
    elif fill == "neutral":
        daily = daily.fillna(0.0)
    daily.index.name = "date"
    return daily


def lag_signal(index_df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Shift the sentiment signal forward so day t only sees t-lag and earlier."""
    return index_df.shift(lag)


def ticker_sentiment_wide(scores: pd.DataFrame, trading_days: pd.DatetimeIndex,
                          fill: str = "ffill") -> pd.DataFrame:
    """Per-ticker daily sentiment (used by the fusion tilt), same fill rules."""
    wide = (scores.pivot_table(index="trading_day", columns="ticker",
                               values="sentiment", aggfunc="mean").sort_index())
    grid = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_days).unique()))
    wide = wide.reindex(grid)
    if fill == "ffill":
        wide = wide.ffill()
    elif fill == "neutral":
        wide = wide.fillna(0.0)
    return wide