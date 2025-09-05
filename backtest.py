# reusable backtest engine
# comments are in English
import math
import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

def default_config():
    return {
        "tickers": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL"],
        "benchmark": "SPY",
        "start_date": "2010-01-01",
        "end_date": None,
        "three_month_signal_threshold": 0.00,
        "min_price_history_days": 400,
        "plot": False,
        "figure_path": "equity_curve.png",
        "results_csv": "backtest_trades_and_stats.csv",
        "calendar_pad_days": 5
    }

def safe_end_date(end_date_str):
    if end_date_str is None or str(end_date_str).lower() == "none":
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(end_date_str)

def download_prices(unique_tickers, start_date, end_date):
    data = yf.download(unique_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.sort_index().ffill()

def get_earnings_dates(ticker, start_date, end_date):
    tk = yf.Ticker(ticker)
    try:
        df = tk.get_earnings_dates(limit=240)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    dates = df.index.tz_localize(None)
    dates = dates[(dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))]
    return sorted(pd.to_datetime(dates).to_pydatetime())

def pct_return(series, start_dt, end_dt):
    try:
        s = series.loc[:end_dt].iloc[-1]
        b = series.loc[:start_dt].iloc[-1]
        return float(s / b - 1.0)
    except Exception:
        return np.nan

def run_backtest(cfg):
    tickers = list(dict.fromkeys(cfg["tickers"]))
    bench = cfg["benchmark"]
    all_tickers = sorted(set(tickers + [bench]))
    end_date = safe_end_date(cfg["end_date"])
    start_date = pd.Timestamp(cfg["start_date"])
    pad_days = int(cfg.get("calendar_pad_days", 0))
    effective_end_for_signals = end_date - pd.Timedelta(days=pad_days) if pad_days > 0 else end_date

    prices = download_prices(all_tickers, start_date - pd.Timedelta(days=500), end_date + pd.Timedelta(days=2))
    bench_prices = prices[bench].dropna()

    trade_ledger = []
    for tkr in tickers:
        px = prices[tkr].dropna()
        if px.empty:
            continue
        if (px.index.max() - px.index.min()).days < cfg["min_price_history_days"]:
            continue

        earn_dates = get_earnings_dates(tkr, start_date, effective_end_for_signals)
        for E in earn_dates:
            E = pd.Timestamp(E).normalize()
            if E < px.index.min() or E > effective_end_for_signals:
                continue
            entry_dt = E + relativedelta(months=3)
            exit_dt = E + relativedelta(months=12)
            try:
                entry_ts = px.index[px.index.get_indexer([entry_dt], method="nearest")[0]]
                exit_ts = px.index[px.index.get_indexer([exit_dt], method="nearest")[0]]
            except Exception:
                continue
            if entry_ts >= exit_ts or entry_ts >= prices.index.max():
                continue
            r3m = pct_return(px, E, entry_ts)
            if np.isnan(r3m):
                continue
            if r3m >= cfg["three_month_signal_threshold"]:
                entry_price = float(px.loc[entry_ts])
                exit_price = float(px.loc[exit_ts])
                r_hold = float(exit_price / entry_price - 1.0)
                trade_ledger.append({
                    "ticker": tkr,
                    "earn_date": E.date().isoformat(),
                    "entry_date": pd.Timestamp(entry_ts).date().isoformat(),
                    "exit_date": pd.Timestamp(exit_ts).date().isoformat(),
                    "r_3m": r3m,
                    "r_hold": r_hold
                })

    if trade_ledger:
        trades = pd.DataFrame(trade_ledger).sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    else:
        trades = pd.DataFrame(columns=["ticker", "earn_date", "entry_date", "exit_date", "r_3m", "r_hold"])

    daily_index = prices.index[(prices.index >= start_date) & (prices.index <= end_date)]
    equity = pd.Series(index=daily_index, dtype=float, data=1.0)

    positions = []
    for _, row in trades.iterrows():
        seg = prices[row["ticker"]].loc[pd.Timestamp(row["entry_date"]):pd.Timestamp(row["exit_date"])].pct_change().fillna(0.0)
        positions.append(seg)
    if positions:
        pos_df = pd.concat(positions, axis=1)
        daily_ret = pos_df.mean(axis=1).reindex(daily_index).fillna(0.0)
        equity = (1.0 + daily_ret).cumprod()

    bench_seg = bench_prices.reindex(daily_index).ffill().pct_change().fillna(0.0)
    bench_equity = (1.0 + bench_seg).cumprod()

    def sharpe(returns, freq=252):
        mu = returns.mean() * freq
        sig = returns.std(ddof=0) * math.sqrt(freq)
        return float(mu / sig) if sig > 0 else np.nan

    bt_ret = equity.pct_change().dropna()
    bm_ret = bench_equity.pct_change().dropna()
    stats = {
        "trades": int(len(trades)),
        "win_rate": float((trades["r_hold"] > 0).mean()) if len(trades) else np.nan,
        "avg_trade_return": float(trades["r_hold"].mean()) if len(trades) else np.nan,
        "median_trade_return": float(trades["r_hold"].median()) if len(trades) else np.nan,
        "equity_cagr": float((equity.iloc[-1]) ** (252.0 / len(equity)) - 1.0) if len(equity) > 1 else np.nan,
        "bench_cagr": float((bench_equity.iloc[-1]) ** (252.0 / len(bench_equity)) - 1.0) if len(bench_equity) > 1 else np.nan,
        "sharpe": sharpe(bt_ret),
        "bench_sharpe": sharpe(bm_ret),
        "max_drawdown": float(((equity / equity.cummax()) - 1.0).min()) if len(equity) else np.nan,
        "bench_max_drawdown": float(((bench_equity / bench_equity.cummax()) - 1.0).min()) if len(bench_equity) else np.nan
    }

    return stats, trades, equity, bench_equity

def run_backtest_baseline(cfg):
    tickers = list(dict.fromkeys(cfg["tickers"]))
    bench = cfg["benchmark"]
    all_tickers = sorted(set(tickers + [bench]))
    end_date = safe_end_date(cfg["end_date"])
    start_date = pd.Timestamp(cfg["start_date"])
    pad_days = int(cfg.get("calendar_pad_days", 0))
    effective_end_for_signals = end_date - pd.Timedelta(days=pad_days) if pad_days > 0 else end_date

    prices = download_prices(all_tickers, start_date - pd.Timedelta(days=500), end_date + pd.Timedelta(days=2))
    bench_prices = prices[bench].dropna()

    trade_ledger = []
    for tkr in tickers:
        px = prices[tkr].dropna()
        if px.empty:
            continue
        if (px.index.max() - px.index.min()).days < cfg["min_price_history_days"]:
            continue

        earn_dates = get_earnings_dates(tkr, start_date, effective_end_for_signals)
        for E in earn_dates:
            E = pd.Timestamp(E).normalize()
            if E < px.index.min() or E > effective_end_for_signals:
                continue
            entry_dt = E + relativedelta(months=3)
            exit_dt = E + relativedelta(months=12)
            try:
                entry_ts = px.index[px.index.get_indexer([entry_dt], method="nearest")[0]]
                exit_ts = px.index[px.index.get_indexer([exit_dt], method="nearest")[0]]
            except Exception:
                continue
            if entry_ts >= exit_ts or entry_ts >= prices.index.max():
                continue
            entry_price = float(px.loc[entry_ts])
            exit_price = float(px.loc[exit_ts])
            r_3m = pct_return(px, E, entry_ts)
            r_hold = float(exit_price / entry_price - 1.0)
            trade_ledger.append({
                "ticker": tkr,
                "earn_date": E.date().isoformat(),
                "entry_date": pd.Timestamp(entry_ts).date().isoformat(),
                "exit_date": pd.Timestamp(exit_ts).date().isoformat(),
                "r_3m": r_3m,
                "r_hold": r_hold
            })

    if trade_ledger:
        trades = pd.DataFrame(trade_ledger).sort_values(["entry_date", "ticker"]).reset_index(drop=True)
    else:
        trades = pd.DataFrame(columns=["ticker", "earn_date", "entry_date", "exit_date", "r_3m", "r_hold"])

    daily_index = prices.index[(prices.index >= start_date) & (prices.index <= end_date)]
    equity = pd.Series(index=daily_index, dtype=float, data=1.0)

    positions = []
    for _, row in trades.iterrows():
        seg = prices[row["ticker"]].loc[pd.Timestamp(row["entry_date"]):pd.Timestamp(row["exit_date"])].pct_change().fillna(0.0)
        positions.append(seg)
    if positions:
        pos_df = pd.concat(positions, axis=1)
        daily_ret = pos_df.mean(axis=1).reindex(daily_index).fillna(0.0)
        equity = (1.0 + daily_ret).cumprod()

    bench_seg = bench_prices.reindex(daily_index).ffill().pct_change().fillna(0.0)
    bench_equity = (1.0 + bench_seg).cumprod()

    def sharpe(returns, freq=252):
        mu = returns.mean() * freq
        sig = returns.std(ddof=0) * math.sqrt(freq)
        return float(mu / sig) if sig > 0 else np.nan

    bt_ret = equity.pct_change().dropna()
    bm_ret = bench_equity.pct_change().dropna()
    stats = {
        "trades": int(len(trades)),
        "win_rate": float((trades["r_hold"] > 0).mean()) if len(trades) else np.nan,
        "avg_trade_return": float(trades["r_hold"].mean()) if len(trades) else np.nan,
        "median_trade_return": float(trades["r_hold"].median()) if len(trades) else np.nan,
        "equity_cagr": float((equity.iloc[-1]) ** (252.0 / len(equity)) - 1.0) if len(equity) > 1 else np.nan,
        "bench_cagr": float((bench_equity.iloc[-1]) ** (252.0 / len(bench_equity)) - 1.0) if len(bench_equity) > 1 else np.nan,
        "sharpe": sharpe(bt_ret),
        "bench_sharpe": sharpe(bm_ret),
        "max_drawdown": float(((equity / equity.cummax()) - 1.0).min()) if len(equity) else np.nan,
        "bench_max_drawdown": float(((bench_equity / bench_equity.cummax()) - 1.0).min()) if len(bench_equity) else np.nan
    }

    return stats, trades, equity, bench_equity
