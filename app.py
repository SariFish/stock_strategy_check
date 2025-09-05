# streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest import run_backtest, default_config, run_backtest_baseline

st.set_page_config(page_title="Earnings Drift Backtest", layout="wide")

st.title("Earnings Drift Backtest")

with st.sidebar:
    st.header("Inputs")
    tickers_str = st.text_input("Tickers comma separated", value="EQIX, AVG, META, GE, HWM, UBER, APP")
    benchmark = st.text_input("Benchmark ETF", value="SPY")
    start_date = st.date_input("Start date", value=pd.to_datetime("2010-01-01")).strftime("%Y-%m-%d")
    end_date = st.date_input("End date", value=pd.to_datetime("today")).strftime("%Y-%m-%d")
    threshold = st.number_input("Three month signal threshold", value=0.00, step=0.01, format="%.2f")
    min_hist_days = st.number_input("Min price history days", value=400, step=10)
    calendar_pad = st.number_input("Calendar pad days", value=5, step=1)
    run_btn = st.button("Run backtest")

st.caption(
    "Rule: buy at three months after earnings if the three month return from the earnings date is at least the threshold. "
    "Hold until twelve months after the earnings date. Equal weight across all open positions.\n\n"
    "**Baseline:** buy three months after every earnings report, hold for nine months, regardless of performance."
)

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
        # אסטרטגיה ראשית
        stats, trades, equity, bench_equity = run_backtest(cfg)
        # אסטרטגיית בסיס
        baseline_stats, baseline_trades, baseline_equity, _ = run_backtest_baseline(cfg)

    st.subheader("Summary")
    summary = pd.DataFrame({
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
        "Earnings Drift": [
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
        ],
        "Baseline Buy&Hold": [
            baseline_stats["trades"],
            f"{baseline_stats['win_rate']:.2%}",
            f"{baseline_stats['avg_trade_return']:.2%}",
            f"{baseline_stats['median_trade_return']:.2%}",
            f"{baseline_stats['equity_cagr']:.2%}",
            f"{baseline_stats['bench_cagr']:.2%}",
            f"{baseline_stats['sharpe']:.2f}",
            f"{baseline_stats['bench_sharpe']:.2f}",
            f"{baseline_stats['max_drawdown']:.2%}",
            f"{baseline_stats['bench_max_drawdown']:.2%}",
        ]
    })
    st.dataframe(summary, use_container_width=True)

    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(10, 5))
    equity.plot(ax=ax, label="Earnings Drift Strategy", color="limegreen", linewidth=2.5)
    baseline_equity.plot(ax=ax, label="Baseline Buy&Hold", color="royalblue", linewidth=2)
    bench_equity.plot(ax=ax, label=cfg["benchmark"], color="orange", linewidth=2, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curve Comparison")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Trades – Earnings Drift")
    st.dataframe(trades, use_container_width=True)
    st.download_button("Download trades CSV (Earnings Drift)", data=trades.to_csv(index=False), file_name="trades.csv", mime="text_csv")

    st.subheader("Trades – Baseline Buy&Hold")
    st.dataframe(baseline_trades, use_container_width=True)
    st.download_button("Download trades CSV (Baseline)", data=baseline_trades.to_csv(index=False), file_name="trades_baseline.csv", mime="text_csv")
