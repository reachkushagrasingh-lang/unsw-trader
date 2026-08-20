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
    growth = (1 + r).cumprod()
    m = metrics.loc[fund]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised return", fmt_pct(m["ann_return"]))
    c2.metric("Annualised volatility", fmt_pct(m["ann_vol"]))
    c3.metric("Sharpe (rf=0)", f"{m['sharpe']:.2f}")
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
        show["sharpe"] = metrics["sharpe"].map(lambda x: f"{x:.2f}")
        st.dataframe(show, use_container_width=True)
        st.bar_chart(metrics["sharpe"])