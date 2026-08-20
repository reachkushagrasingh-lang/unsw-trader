"""Station 2 - features and text assembly.

Returns are the one feature the optimiser needs. We compute them *within each
panel first*, then left-merge crypto onto the equity trading calendar. The
headline panel is assembled and date-aligned here; scoring is Station 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Wide (date x ticker) simple daily returns. First obs per ticker is dropped."""
    wide = prices.pivot(index="date", columns="ticker", values=price_col).sort_index()
    return wide.pct_change().iloc[1:]


def combined_returns_panel(equities: pd.DataFrame, crypto: pd.DataFrame) -> pd.DataFrame:
    """Equity returns with crypto returns LEFT-MERGED onto the equity calendar.

    Crypto returns are computed on crypto's own 365-day calendar first, then
    reindexed to equity trading days. Weekend-only crypto moves are intentionally
    dropped (a fund trading on equity days could not act on them).
    """
    eq_ret = daily_returns(equities)
    cr_ret = daily_returns(crypto)
    cr_on_eq = cr_ret.reindex(eq_ret.index)          # left-join on equity dates
    combined = eq_ret.join(cr_on_eq, how="left")
    return combined


def returns_descriptive_stats(equities: pd.DataFrame, crypto: pd.DataFrame) -> pd.DataFrame:
    """Mean / vol / min / max / skew / kurtosis of daily returns by asset class."""
    eq = daily_returns(equities).stack()
    cr = daily_returns(crypto).stack()
    rows = []
    for name, s in [("Equity", eq), ("Crypto", cr)]:
        rows.append({
            "asset_class": name, "n_obs": int(s.shape[0]),
            "mean_daily": s.mean(), "vol_daily": s.std(),
            "min": s.min(), "max": s.max(),
            "skew": s.skew(), "excess_kurtosis": s.kurt(),
        })
    return pd.DataFrame(rows).set_index("asset_class")


def assemble_headline_panel(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Align each headline to its equity trading day.

    Same day if it is a trading day, otherwise the NEXT trading day. Raw title
    text is preserved (VADER needs casing/punctuation/stopwords). Returns long
    rows: [trading_day, ticker, sector, title].
    """
    td = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_days).unique()))
    df = news.copy()
    pos = td.searchsorted(df["date"].values, side="left")
    pos = np.clip(pos, 0, len(td) - 1)
    df["trading_day"] = td[pos]
    df = df[df["date"] <= td[-1]]
    return (df[["trading_day", "ticker", "sector", "title"]]
            .sort_values(["trading_day", "ticker"]).reset_index(drop=True))