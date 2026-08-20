"""Tests that matter for the marks: weights are valid and there is NO look-ahead.

    python tests/test_backtest.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import portfolios


def _fake_returns(seed=0, n=600, k=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(rng.normal(0.0004, 0.02, (n, k)),
                        index=idx, columns=[f"A{i}" for i in range(k)])


def test_weights_valid():
    r = _fake_returns()
    for m in portfolios.METHODS:
        bt = portfolios.oos_backtest(r, m, estimation_window=252)
        w = bt["weights"]
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6), f"{m} weights don't sum to 1"
        assert (w.values >= -1e-9).all(), f"{m} has negative weights"
    print("weights valid for all methods OK")


def test_no_lookahead():
    """Change returns AFTER a rebalance date; weights on/before it must not move."""
    r = _fake_returns()
    bt1 = portfolios.oos_backtest(r, "max_sharpe", estimation_window=252)
    cut = bt1["weights"].index[3]                      # a known rebalance date
    r2 = r.copy()
    r2.loc[r2.index > cut] *= 5.0                       # blow up the FUTURE only
    bt2 = portfolios.oos_backtest(r2, "max_sharpe", estimation_window=252)
    early = bt1["weights"].index[bt1["weights"].index <= cut]
    a = bt1["weights"].loc[early]
    b = bt2["weights"].loc[early]
    assert np.allclose(a.values, b.values, atol=1e-8), "LOOK-AHEAD: past weights changed!"
    print("no look-ahead OK (future data does not affect past weights)")


def test_methods_differ():
    """Guard against the solver-stall trap the brief warns about."""
    r = _fake_returns(seed=7)
    mv = portfolios.oos_backtest(r, "min_variance", 252)["weights"].iloc[-1]
    ew = portfolios.oos_backtest(r, "equal_weight", 252)["weights"].iloc[-1]
    assert not np.allclose(mv.values, ew.values, atol=1e-3), "min-var == equal-weight (solver stalled?)"
    print("methods produce distinct weights OK")


if __name__ == "__main__":
    test_weights_valid()
    test_no_lookahead()
    test_methods_differ()
    print("\nAll backtest tests passed.")