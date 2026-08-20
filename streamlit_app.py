"""UNSW-Trader - systematic multi-asset funds with news-sentiment analytics.

The deployed app is READ-ONLY: it loads precomputed artifacts from results/ and
the raw prices via the data-access helper. It never runs a backtest or VADER
(the free tier cannot), so it stays fast. Rebuild artifacts with
`python scripts/run_part_b.py`.
"""
import pathlib

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "results" / "data"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

st.set_page_config(page_title="UNSW-Trader", layout="wide")

PALETTE = ["#1b3a5b", "#2e8b8b", "#e08a3c", "#a83e5b", "#6b7a8f", "#3c8c5a"]


@st.cache_data(show_spinner=False)
def load_artifacts():
    fr = pd.read_csv(DATA / "fund_returns.csv", index_col=0, parse_dates=True)
    fw = pd.read_csv(DATA / "fund_weights.csv", parse_dates=["date"])
    metrics = pd.read_csv(TAB / "performance_metrics.csv", index_col=0)
    sent = pd.read_csv(DATA / "sector_sentiment_index.csv", index_col=0, parse_dates=True)
    return fr, fw, metrics, sent


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:.2%}"


def fact_sheet(fund, fr, fw, metrics):
    r = fr[fund].dropna()
    if r.empty:
        st.info("No return history for this fund.")
        return
    growth = (1 + r).cumprod()
    m = metrics.loc[fund]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", fmt_pct(m["ann_return"]))
    c2.metric("Annualised volatility", fmt_pct(m["ann_vol"]))
    c3.metric("Sharpe (rf=0)", f"{m['sharpe']:.2f}" if pd.notna(m["sharpe"]) else "-")
    c4.metric("Max drawdown", fmt_pct(m["max_drawdown"]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Growth of $1**")
        st.line_chart(growth.rename("Value of $1"))
    with right:
        st.markdown("**Drawdown**")
        dd = growth / growth.cummax() - 1
        st.area_chart(dd.rename("Drawdown"))

    st.markdown("**Current holdings** (target weights, most recent rebalance)")
    hold = (fw[fw["fund"] == fund][["asset", "weight"]]
            .sort_values("weight", ascending=False).reset_index(drop=True))
    if hold.empty:
        st.info("No holdings recorded for this fund.")
    else:
        hold["weight"] = hold["weight"].map(lambda w: f"{w:.1%}")
        st.dataframe(hold, use_container_width=True, height=280)


def main():
    if not (DATA / "fund_returns.csv").exists():
        st.error("No precomputed results found. Run `python scripts/run_part_b.py` first.")
        return
    fr, fw, metrics, sent = load_artifacts()
    funds = list(fr.columns)

    st.title("UNSW-Trader")
    st.caption("Systematically managed multi-asset funds, evaluated out-of-sample. "
               "Compare funds, read a fact sheet, set an allocation, and follow sector sentiment.")

    tab_compare, tab_fact, tab_alloc, tab_sent = st.tabs(
        ["Compare funds", "Fund fact sheet", "Build allocation", "Sentiment"])

    with tab_compare:
        st.subheader("All funds at a glance")
        show = metrics[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].copy()
        for c in ["ann_return", "ann_vol", "max_drawdown"]:
            show[c] = show[c].map(fmt_pct)
        show["sharpe"] = metrics["sharpe"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        st.dataframe(show, use_container_width=True)
        st.bar_chart(metrics["sharpe"])

    with tab_fact:
        default = "Combined_max_sharpe" if "Combined_max_sharpe" in funds else funds[0]
        fund = st.selectbox("Choose a fund", funds, index=funds.index(default))
        st.subheader(fund.replace("_", " "))
        fact_sheet(fund, fr, fw, metrics)

    with tab_alloc:
        st.subheader("Split your money across funds")
        st.caption("Set a percentage for each fund; the blended out-of-sample track record updates live.")
        picks = st.multiselect(
            "Funds to include", funds,
            default=[f for f in ["Combined_max_sharpe", "Combined_min_variance"] if f in funds])
        alloc = {}
        if picks:
            cols = st.columns(len(picks))
            for i, f in enumerate(picks):
                alloc[f] = cols[i].slider(f, 0, 100, int(100 / len(picks)), key=f)
        total = sum(alloc.values())
        if picks and total > 0:
            w = {f: a / total for f, a in alloc.items()}
            st.write("Normalised weights: " + ", ".join(f"{k}={v:.0%}" for k, v in w.items()))
            sub = fr[picks].dropna(how="all")          # only dates where >=1 fund is live
            if sub.empty:
                st.info("No overlapping return history for the selected funds.")
            else:
                blend = sum(sub[f].fillna(0) * wt for f, wt in w.items())
                blend = blend.loc[sub.index]
                g = (1 + blend).cumprod()
                st.line_chart(g.rename("Blended $1"))
                ann = blend.mean() * 252
                vol = blend.std() * np.sqrt(252)
                c1, c2, c3 = st.columns(3)
                c1.metric("Blended ann. return", fmt_pct(ann))
                c2.metric("Blended ann. vol", fmt_pct(vol))
                c3.metric("Blended Sharpe", f"{ann/vol:.2f}" if vol else "-")
        else:
            st.info("Pick at least one fund and set a non-zero allocation.")

    with tab_sent:
        st.subheader("Sector news-sentiment index")
        st.caption("VADER compound score, equal-weighted across tickers in each sector, "
                   "lagged for trading. Higher = more positive headline tone.")
        sectors = list(sent.columns)
        chosen = st.multiselect("Sectors", sectors, default=sectors[:4])
        smooth = st.slider("Smoothing (rolling days)", 1, 63, 21)
        if chosen:
            st.line_chart(sent[chosen].rolling(smooth).mean())


if __name__ == "__main__":
    main()