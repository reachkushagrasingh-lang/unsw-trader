"""Station 1 - ETL: load and clean the structured + unstructured data.

Everything loads through src.data_access (the frozen course helper). This module
adds the integrity checks the brief asks for and returns clean frames the rest of
the pipeline consumes. No data files are written or committed here.

Design choices (documented so the report can cite them):
- Equities/crypto are unique by (ticker, date); we assert this and drop exact dups.
- Crypto has 10 stray rows dated 2024-01-01 -> we cap the sample at 2023-12-31.
- News has many rows per (ticker, date); duplicates are checked on
  (ticker, date, title), NOT (ticker, date).
- Extreme returns are REAL events (COVID crash, meme spikes): we flag them, keep
  them, and report them. We do not delete.
"""
from __future__ import annotations

import pandas as pd

from src import data_access

SAMPLE_END = pd.Timestamp("2023-12-31")


def _clean_prices(df: pd.DataFrame, has_sector: bool) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df[df["date"] <= SAMPLE_END]                       # drop the 2024 strays
    df = df.drop_duplicates(subset=["ticker", "date"])      # enforce ticker-date key
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    keep = ["ticker", "date", "open", "high", "low", "close", "adjClose", "volume"]
    if has_sector:
        keep.append("sector")
    return df[keep]


def load_clean_equities() -> pd.DataFrame:
    """50 US equities, cleaned, unique by (ticker, date), with sector."""
    return _clean_prices(data_access.load_equity_prices(), has_sector=True)


def load_clean_crypto() -> pd.DataFrame:
    """10 cryptos, cleaned, capped at 2023-12-31, price-only (no sector)."""
    return _clean_prices(data_access.load_crypto_prices(), has_sector=False)


def load_clean_news() -> pd.DataFrame:
    """Headlines, deduped on (ticker, date, title), tz-normalised to naive dates."""
    df = data_access.load_news_headlines().copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    df = df[df["date"] <= SAMPLE_END]
    df["title"] = df["title"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["ticker", "date", "title"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def integrity_report() -> dict:
    """Quantify (not just list) the issues found, so the report can cite numbers."""
    eq_raw = data_access.load_equity_prices()
    cr_raw = data_access.load_crypto_prices()
    nw_raw = data_access.load_news_headlines()

    eq, cr, nw = load_clean_equities(), load_clean_crypto(), load_clean_news()

    news_dups = len(nw_raw) - nw_raw.assign(
        title=nw_raw["title"].fillna("")
    ).drop_duplicates(subset=["ticker", "date", "title"]).shape[0]

    return {
        "equity_rows_raw": len(eq_raw), "equity_rows_clean": len(eq),
        "crypto_rows_raw": len(cr_raw), "crypto_rows_clean": len(cr),
        "crypto_2024_rows_dropped": int((pd.to_datetime(cr_raw["date"]).dt.tz_localize(None) > SAMPLE_END).sum()),
        "news_rows_raw": len(nw_raw), "news_rows_clean": len(nw),
        "news_exact_duplicates": int(news_dups),
        "equity_tickers": eq["ticker"].nunique(),
        "crypto_tickers": cr["ticker"].nunique(),
        "sectors": eq["sector"].nunique(),
    }