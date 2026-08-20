"""Station 3 (extension) - fuse sentiment into the equity fund, look-ahead safe.

Baseline fusion: at each rebalance, tilt the optimiser's equity weights toward
names with positive LAGGED sentiment, then renormalise. We backtest base vs
tilted on the same dates and report the difference. A negative result, honestly
explained, still earns the fusion mark - the point is a measured, safe attempt.

Sentiment applies to EQUITY names only (crypto has no news).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import portfolios


def tilt_weights(weights_row: pd.Series, sent_row: pd.Series, strength: float = 0.5) -> pd.Series:
    """Multiplicative tilt: w_i * (1 + strength * z_i), clipped >=0, renormalised.

    z_i is the cross-sectional sentiment for the names held that day. Crypto names
    (absent from sent_row) get a neutral z of 0, so their weights are unchanged
    before renormalisation.
    """
    z = sent_row.reindex(weights_row.index).fillna(0.0)
    tilted = weights_row * (1.0 + strength * z)
    tilted = tilted.clip(lower=0.0)
    s = tilted.sum()
    return tilted / s if s > 0 else weights_row


def backtest_with_sentiment(returns: pd.DataFrame, sentiment_lagged: pd.DataFrame,
                            method: str = "max_sharpe", estimation_window: int = 252,
                            periods_per_year: int = 252, strength: float = 0.5):
    """Re-run the walk-forward backtest, tilting each rebalance's weights by the
    sentiment known on the rebalance date (already lagged upstream).
    """
    base = portfolios.oos_backtest(returns, method, estimation_window, periods_per_year)
    base_w = base["weights"]

    tilted_w = {}
    for d, row in base_w.iterrows():
        sent_row = (sentiment_lagged.loc[d] if d in sentiment_lagged.index
                    else pd.Series(dtype=float))
        tilted_w[d] = tilt_weights(row, sent_row, strength)
    tilted_w = pd.DataFrame(tilted_w).T.reindex(columns=returns.columns).fillna(0.0)

    idx = returns.index
    held = tilted_w.reindex(idx, method="ffill").dropna(how="all")
    port_daily = (held * returns.loc[held.index]).sum(axis=1)
    first_live = tilted_w.index[0]
    port_daily = port_daily[port_daily.index > first_live]

    return {
        "base": base,
        "tilted": {
            "method": method,
            "daily_returns": port_daily,
            "weights": tilted_w,
            "growth": (1 + port_daily).cumprod(),
            "metrics": portfolios.performance_metrics(port_daily, periods_per_year),
        },
    }