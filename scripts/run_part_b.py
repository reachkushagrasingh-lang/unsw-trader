"""Reproduce all Part B results. Run from the project root:

    python scripts/run_part_b.py

Writes the app-readable CSVs to results/data/, report tables to results/tables/,
and figures to results/figures/. The deployed app only READS these - it never
recomputes a backtest or runs VADER.
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402

from src import etl, features, portfolios, sentiment, fusion, viz  # noqa: E402

DATA = ROOT / "results" / "data"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
for d in (DATA, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

viz.apply_style()
EST_WINDOW = 252


def build_funds():
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()

    eq_ret = features.daily_returns(eq)
    cr_ret_native = features.daily_returns(cr)              # crypto's own 365 calendar
    combined = features.combined_returns_panel(eq, cr)      # equity calendar

    families = {
        "Equity":   (eq_ret,        252),
        "Crypto":   (cr_ret_native, 365),
        "Combined": (combined,      252),
    }
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]

    fund_returns, fund_weights_long, metrics_rows = {}, [], []
    for fam, (ret, ppy) in families.items():
        for m in methods:
            name = f"{fam}_{m}"
            bt = portfolios.oos_backtest(ret, m, EST_WINDOW, ppy)
            fund_returns[name] = bt["daily_returns"]
            latest = bt["weights"].iloc[-1]
            latest = latest[latest > 1e-4]
            for asset, w in latest.items():
                fund_weights_long.append(
                    {"fund": name, "date": bt["weights"].index[-1],
                     "asset": asset, "weight": float(w)})
            metrics_rows.append({"fund": name, **bt["metrics"],
                                 "first_live": bt["first_live_date"].date(),
                                 "periods_per_year": ppy})
            print(f"  {name:26s} Sharpe={bt['metrics']['sharpe']:.2f}  "
                  f"AnnRet={bt['metrics']['ann_return']:.2%}")

    fr = pd.DataFrame(fund_returns)
    fr.index.name = "date"
    fr.to_csv(DATA / "fund_returns.csv")
    pd.DataFrame(fund_weights_long).to_csv(DATA / "fund_weights.csv", index=False)
    metrics = pd.DataFrame(metrics_rows).set_index("fund")
    metrics.to_csv(TAB / "performance_metrics.csv")
    combined_minvar_weights = portfolios.oos_backtest(
        combined, "min_variance", EST_WINDOW, 252)["weights"]
    return eq, cr, eq_ret, combined, fr, metrics, combined_minvar_weights


def build_sentiment(eq, news_trading_days):
    news = etl.load_clean_news()
    panel = features.assemble_headline_panel(news, news_trading_days)
    scores = sentiment.score_headlines(panel, use_finance_lexicon=True)
    sector_idx = sentiment.sector_sentiment_index(scores, news_trading_days, fill="ffill")
    sector_idx.to_csv(DATA / "sector_sentiment_index.csv")
    ticker_sent = sentiment.ticker_sentiment_wide(scores, news_trading_days, fill="ffill")
    return scores, sector_idx, ticker_sent


def build_fusion(eq_ret, ticker_sent):
    sent_lagged = sentiment.lag_signal(ticker_sent, lag=1)      # look-ahead safe
    res = fusion.backtest_with_sentiment(eq_ret, sent_lagged, method="max_sharpe",
                                         estimation_window=EST_WINDOW, strength=0.5)
    base_m = res["base"]["metrics"]
    tilt_m = res["tilted"]["metrics"]
    comp = pd.DataFrame({"base_max_sharpe": base_m, "sentiment_tilted": tilt_m}).T
    comp.to_csv(TAB / "fusion_comparison.csv")
    return res, comp


def make_figures(metrics, fr, sector_idx, fusion_res, weights_over_time):
    fig, ax = plt.subplots()
    for col in [c for c in fr.columns if c.startswith("Combined")]:
        (1 + fr[col].dropna()).cumprod().plot(ax=ax, label=col.replace("Combined_", ""))
    ax.set_title("Growth of $1 - Combined funds (out-of-sample)")
    ax.set_ylabel("Value of $1"); ax.set_xlabel("Date"); ax.legend()
    viz.savefig(fig, FIG / "growth_of_1.png", "Combined equity+crypto, OOS, 2020-2023")

    fig, ax = plt.subplots()
    g = (1 + fr["Combined_min_variance"].dropna()).cumprod()
    ((g / g.cummax()) - 1).plot(ax=ax, color=viz.PALETTE[3])
    ax.set_title("Drawdown - Combined Minimum-Variance")
    ax.set_ylabel("Drawdown"); ax.set_xlabel("Date")
    viz.savefig(fig, FIG / "drawdown.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    metrics["sharpe"].plot.bar(ax=ax, color=viz.PALETTE[0])
    ax.set_title("Sharpe ratio across funds and methods")
    ax.set_ylabel("Sharpe (rf=0)"); ax.axhline(0, color="#333", lw=0.8)
    plt.xticks(rotation=45, ha="right")
    viz.savefig(fig, FIG / "sharpe_barplot.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sector_idx.rolling(21).mean().plot(ax=ax, lw=1)
    ax.set_title("Sector news-sentiment index (21-day mean)")
    ax.set_ylabel("VADER compound"); ax.set_xlabel("Date")
    ax.legend(ncol=5, fontsize=7)
    viz.savefig(fig, FIG / "sentiment_index.png")

    fig, ax = plt.subplots()
    fusion_res["base"]["growth"].plot(ax=ax, label="Base max-Sharpe")
    fusion_res["tilted"]["growth"].plot(ax=ax, label="Sentiment-tilted")
    ax.set_title("Fusion: base vs sentiment-tilted (equity fund)")
    ax.set_ylabel("Value of $1"); ax.set_xlabel("Date"); ax.legend()
    viz.savefig(fig, FIG / "fusion_before_after.png")

    # --- required exhibit #4: portfolio weights over time (stacked area) ---
    w = weights_over_time.copy()
    top = w.mean().sort_values(ascending=False).head(8).index      # 8 biggest holdings
    plotw = w[top].copy()
    plotw["Other"] = w.drop(columns=top).sum(axis=1)               # collapse the rest
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(plotw.index, plotw.T.values, labels=plotw.columns)
    ax.set_title("Portfolio weights over time - Combined Minimum-Variance")
    ax.set_ylabel("Weight"); ax.set_xlabel("Rebalance date")
    ax.set_ylim(0, 1); ax.legend(ncol=3, fontsize=7, loc="upper left")
    viz.savefig(fig, FIG / "weights_over_time.png")


def main():
    print("Building funds...")
    eq, cr, eq_ret, combined, fr, metrics, cmv_weights = build_funds()
    trading_days = pd.DatetimeIndex(eq_ret.index)

    print("Building sentiment index...")
    scores, sector_idx, ticker_sent = build_sentiment(eq, trading_days)

    print("Building fusion extension...")
    fusion_res, comp = build_fusion(eq_ret, ticker_sent)
    print(comp[["ann_return", "sharpe", "max_drawdown"]].to_string())

    print("Rendering figures...")
    make_figures(metrics, fr, sector_idx, fusion_res, cmv_weights)
    print("Done. Outputs in results/.")


if __name__ == "__main__":
    main()