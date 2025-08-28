# streamlit app
# comments are in English
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest, default_config

st.set_page_config(page_title="Earnings Drift Backtest", layout="wide")

st.title("Earnings Drift Backtest")

with st.sidebar:
    st.header("Inputs")
    tickers_str = st.text_input("Tickers comma separated", value="AAPL, MSFT, NVDA, AMZN, META, AVGO, GOOGL")
    benchmark = st.text_input("Benchmark ETF", value="SPY")
    start_date = st.date_input("Start date", value=pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
    end_date = st.date_input("End date", value=pd.to_datetime("today")).strftime("%Y-%m-%d")
    threshold = st.number_input("Three month signal threshold", value=0.00, step=0.01, format="%.2f")
    min_hist_days = st.number_input("Min price history days", value=400, step=10)
    calendar_pad = st.number_input("Calendar pad days", value=5, step=1)
    run_btn = st.button("Run backtest")

st.caption("Rule. buy at three months after earnings if the three month return from the earnings date is at least the threshold. hold until twelve months after the earnings date. equal weight across all open positions.")

if run_btn:
    cfg = default_config().copy()
    cfg["tickers"] = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    cfg["benchmark"] = benchmark.strip().upper()
    cfg["start_date"] = start_date
    cfg["end_date"] = end_date
    cfg["three_month_signal_threshold"] = float(threshold)
    cfg["min_price_history_days"] = int(min_hist_days)
    cfg["calendar_pad_days"] = int(calendar_pad)
    cfg["plot"] = False

    with st.spinner("Running backtest"):
        stats, trades, equity, bench_equity = run_backtest(cfg)

    st.subheader("Summary")
    pretty = pd.DataFrame({
        "metric": [
            "trades",
            "win rate",
            "average trade return",
            "median trade return",
            "strategy CAGR",
            "benchmark CAGR",
            "strategy Sharpe",
            "benchmark Sharpe",
            "strategy max drawdown",
            "benchmark max drawdown",
        ],
        "value": [
            stats["trades"],
            f"{stats['win_rate']:.2%}",
            f"{stats['avg_trade_return']:.2%}",
            f"{stats['median_trade_return']:.2%}",
            f"{stats['equity_cagr']:.2%}",
            f"{stats['bench_cagr']:.2%}",
            f"{stats['sharpe']:.2f}",
            f"{stats['bench_sharpe']:.2f}",
            f"{stats['max_drawdown']:.2%}",
            f"{stats['bench_max_drawdown']:.2%}",
        ]
    })
    st.dataframe(pretty, use_container_width=True)

    st.subheader("Equity curve")
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, label="strategy")
    bench_equity.plot(ax=ax, label=cfg["benchmark"])
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.set_title("equity curve")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Trades")
    st.dataframe(trades, use_container_width=True)

    st.download_button("Download trades CSV", data=trades.to_csv(index=False), file_name="trades.csv", mime="text_csv")
