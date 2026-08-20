"""Station 3 - funds: optimal portfolios + walk-forward out-of-sample backtest.

Four long-only optimisers (equal-weight, minimum-variance, maximum-Sharpe /
mean-variance tangency, risk parity). The backtest is walk-forward: at each
monthly rebalance, weights are estimated from a trailing window of PAST returns
only and held until the next rebalance. No look-ahead anywhere.

Annualisation factor is a parameter: 252 for equity/combined funds (which trade
on the equity calendar), 365 for a crypto-only fund on its own calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

METHODS = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]


def _clean_window(win: pd.DataFrame) -> pd.DataFrame:
    """Drop assets with any missing return in the window (keeps Sigma well-posed)."""
    return win.dropna(axis=1, how="any")


def w_equal(win: pd.DataFrame) -> pd.Series:
    cols = _clean_window(win).columns
    return pd.Series(1.0 / len(cols), index=cols)


def w_min_variance(win: pd.DataFrame) -> pd.Series:
    win = _clean_window(win)
    Sigma = win.cov().values
    n = Sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n
    res = minimize(lambda w: w @ Sigma @ w, w0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = res.x if res.success else w0
    return pd.Series(_project(w), index=win.columns)


def w_max_sharpe(win: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Long-only mean-variance tangency (maximise the Sharpe ratio)."""
    win = _clean_window(win)
    mu = win.mean().values
    Sigma = win.cov().values
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * n

    def neg_sharpe(w):
        ret = w @ mu - rf
        vol = np.sqrt(w @ Sigma @ w)
        return -ret / vol if vol > 1e-12 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"maxiter": 1000, "ftol": 1e-12})
    w = res.x if res.success else w0
    return pd.Series(_project(w), index=win.columns)


def w_risk_parity(win: pd.DataFrame) -> pd.Series:
    """Equal risk contribution (long-only), solved as a smooth minimisation."""
    win = _clean_window(win)
    Sigma = win.cov().values
    n = Sigma.shape[0]
    w0 = np.full(n, 1.0 / n)

    def obj(w):
        port_var = w @ Sigma @ w
        mrc = Sigma @ w                    # marginal risk contribution
        rc = w * mrc                       # risk contribution
        target = port_var / n
        return np.sum((rc - target) ** 2)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(1e-6, 1.0)] * n
    res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-14})
    w = res.x if res.success else w0
    return pd.Series(_project(w), index=win.columns)


def _project(w: np.ndarray) -> np.ndarray:
    """Clip tiny negatives from the solver and renormalise to sum to 1."""
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


_OPTIMISERS = {
    "equal_weight": w_equal,
    "min_variance": w_min_variance,
    "max_sharpe": w_max_sharpe,
    "risk_parity": w_risk_parity,
}


def performance_metrics(daily: pd.Series, periods_per_year: int = 252) -> dict:
    daily = daily.dropna()
    if daily.empty:
        return {k: np.nan for k in
                ["ann_return", "ann_vol", "sharpe", "max_drawdown", "cumulative_return"]}
    growth = (1 + daily).cumprod()
    ann_ret = daily.mean() * periods_per_year
    ann_vol = daily.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    running_max = growth.cummax()
    max_dd = (growth / running_max - 1).min()
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "cumulative_return": float(growth.iloc[-1] - 1),
    }


def _rebalance_dates(index: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """First trading day of each month within the index."""
    s = pd.Series(index, index=index)
    grp = s.groupby([index.year, index.month]).first()
    return pd.DatetimeIndex(grp.values)


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance",
                 estimation_window: int = 252, periods_per_year: int = 252,
                 rf: float = 0.0):
    """Walk-forward backtest. Weights use only returns strictly BEFORE each date.

    Returns dict with daily_returns, weights (rebalance_date x asset), growth of
    $1, and metrics. First live date = the first rebalance on/after the window.
    """
    if method not in _OPTIMISERS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    opt = _OPTIMISERS[method]
    idx = returns.index
    rebal = _rebalance_dates(idx)
    rebal = rebal[rebal >= idx[estimation_window]]

    weights_log = {}
    for d in rebal:
        loc = idx.get_loc(d)
        window = returns.iloc[loc - estimation_window:loc]     # PAST data only
        if window.dropna(axis=1, how="any").shape[1] < 2:
            continue
        weights_log[d] = opt(window)

    if not weights_log:
        raise RuntimeError("no rebalances produced weights - check window length")

    weights = pd.DataFrame(weights_log).T.reindex(columns=returns.columns).fillna(0.0)

    held = weights.reindex(idx, method="ffill").dropna(how="all")
    aligned_ret = returns.loc[held.index]
    port_daily = (held * aligned_ret).sum(axis=1)
    first_live = weights.index[0]
    port_daily = port_daily[port_daily.index > first_live]

    growth = (1 + port_daily).cumprod()
    return {
        "method": method,
        "first_live_date": first_live,
        "daily_returns": port_daily,
        "weights": weights,
        "growth": growth,
        "metrics": performance_metrics(port_daily, periods_per_year),
    }