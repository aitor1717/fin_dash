"""Orchestrates the pipeline: feed -> sized trades -> metrics -> site/data JSON.

Usage: python pipeline/build.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

from parse_feed import parse_feed, open_trades
from sizing import size_trades, daily_position_count
from market_data import fetch_benchmark_history, fetch_market_snapshot, benchmark_daily_returns, fetch_intraday_today
import metrics as M


def normalize_dates(series: pd.Series, tz: str) -> pd.Series:
    out = series.copy()
    out.index = out.index.tz_convert(tz).tz_localize(None).normalize()
    return out.groupby(level=0).last()


def build(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tz = cfg["timezone"]
    account_size = float(cfg["account_size"])
    weight_pct = float(cfg["position_weight_pct"])
    benchmarks = cfg["benchmarks"]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Parsing feed: {cfg['feed_path']}")
    trades = parse_feed(cfg["feed_path"])
    print(f"  {len(trades)} trades, {len(open_trades(trades))} currently open")

    print(f"Sizing trades (account_size={account_size}, weight={weight_pct}%)")
    sizing_result = size_trades(trades, account_size, weight_pct)
    print(f"  {sizing_result.pct_skipped:.1f}% of trades skipped (capital cap)")

    pnl = M.daily_pnl(sizing_result.sized_trades, tz)
    equity = M.equity_curve(pnl, account_size)
    cumulative_return_pct = 100 * (equity.iloc[-1] - account_size) / account_size if len(equity) else 0.0

    # Long-only and short-only slices, isolated as if each were the only book
    # traded -- reuses the same sized trades already computed above, just
    # filtered by side, so this is a re-aggregation, not a new calculation.
    long_sized = [s for s in sizing_result.sized_trades if s.trade.position == "long"]
    short_sized = [s for s in sizing_result.sized_trades if s.trade.position == "short"]
    pnl_long = M.daily_pnl(long_sized, tz)
    pnl_short = M.daily_pnl(short_sized, tz)

    start = trades[0].open_dt.tz_convert(tz).date()
    today = datetime.now(timezone.utc).date()
    # freeze_asof: for a static/synthetic book (see config.sample.yaml) that
    # will never get new trades, "today" should be pinned to the last real
    # activity this data actually has -- not real wall-clock time, which
    # would silently drift the benchmark range and live mark-to-market
    # forward on every later run while the underlying feed stays frozen
    # (the source going stale while parts of the display keep moving).
    # Deliberately built from actual opens/closes only, not open positions'
    # own speculative *scheduled* close_dt, which can land arbitrarily far
    # in the future and isn't something that's actually happened yet.
    asof_date = None
    if cfg.get("freeze_asof", False):
        last_activity_dt = max(
            [t.open_dt for t in trades] + [t.close_dt for t in trades if not t.is_open]
        )
        asof_date = min(last_activity_dt.tz_convert(tz).date(), today)
        end = asof_date
        print(f"  freeze_asof: pinning 'today' to {asof_date} (last known feed activity)")
    else:
        end = max(t.close_dt for t in trades).tz_convert(tz).date()
        end = max(end, today)

    print(f"Fetching benchmark history for {benchmarks} ({start} to {end})")
    try:
        bench_close = fetch_benchmark_history(benchmarks, start, end)
        bench_close.index = pd.DatetimeIndex(bench_close.index).tz_localize(None).normalize()
        # yfinance can return a column per ticker with zero real rows (all-NaN)
        # on a failed/rate-limited fetch instead of raising -- drop those
        # columns so downstream code treats them as "unavailable", not silently
        # propagates NaN (which isn't even valid JSON) into the dashboard.
        bench_close = bench_close.dropna(axis=1, how="all")
        bench_ret = benchmark_daily_returns(bench_close)
    except Exception as e:
        print(f"  WARNING: benchmark fetch failed ({e}); historic chart will be strategy-only", file=sys.stderr)
        bench_close = pd.DataFrame()
        bench_ret = pd.DataFrame()

    trading_days = bench_close.index if len(bench_close) else pd.date_range(start, end, freq="D")

    equity_naive = equity.copy()
    equity_naive.index = pd.DatetimeIndex(equity_naive.index).tz_localize(None).normalize()
    equity_aligned = equity_naive.reindex(trading_days).ffill().fillna(account_size)

    util_naive = normalize_dates(sizing_result.utilization, tz) if len(sizing_result.utilization) else pd.Series(dtype=float)
    util_aligned = util_naive.reindex(trading_days).ffill().fillna(0.0)

    pos_count_series = daily_position_count(sizing_result.sized_trades)
    pos_count_naive = normalize_dates(pos_count_series, tz) if len(pos_count_series) else pd.Series(dtype=float)
    pos_count_aligned = pos_count_naive.reindex(trading_days).ffill().fillna(0.0)

    def side_cum_pct(side_pnl: pd.Series) -> pd.Series:
        if side_pnl.empty:
            return pd.Series(0.0, index=trading_days)
        naive = side_pnl.copy()
        naive.index = pd.DatetimeIndex(naive.index).tz_localize(None).normalize()
        full_index = pd.date_range(naive.index.min(), naive.index.max(), freq="D")
        cum = naive.reindex(full_index, fill_value=0.0).cumsum()
        return (100 * cum / account_size).reindex(trading_days).ffill().fillna(0.0)

    long_return_aligned = side_cum_pct(pnl_long)
    short_return_aligned = side_cum_pct(pnl_short)

    # True drawdown, from the full calendar-day curve -- not the benchmark
    # trading-day-aligned one. The feed's scheduled close_datetimes can land
    # on non-trading days (weekends); reindexing onto only benchmark trading
    # days silently drops those P&L events from the running peak/trough,
    # understating real drawdown. The aligned series below is still used for
    # charting (so its x-axis matches the benchmark overlay) and for
    # alpha/beta (which must line up with benchmark trading days).
    dd = M.max_drawdown(equity_naive)
    strategy_daily_ret_aligned = equity_aligned.pct_change().fillna(0.0)
    roll_sharpe_60 = M.rolling_sharpe(strategy_daily_ret_aligned, 60)

    alpha_beta_by_bench = {}
    normalized_series = {"strategy": (100 * equity_aligned / equity_aligned.iloc[0]).round(3).tolist() if len(equity_aligned) else []}
    for b in benchmarks:
        if b in bench_close.columns:
            bench_series = bench_close[b].reindex(trading_days).ffill()
            normalized_series[b] = (100 * bench_series / bench_series.iloc[0]).round(3).tolist()
            alpha_beta_by_bench[b] = M.alpha_beta(strategy_daily_ret_aligned, bench_ret[b].reindex(trading_days).fillna(0.0))

    trade_stats = M.trade_level_stats(trades)
    ticker_conc = M.ticker_concentration(sizing_result.sized_trades)
    closed_returns_pct = [t.pct_change for t in trades if not t.is_open]

    # Synthetic what-if tickers (see simulate_scenario.py) aren't real symbols --
    # skip them here rather than spending one failed network lookup each.
    all_tickers = sorted(set(t.ticker for t in trades if not t.ticker.startswith("SIM-")))
    print(f"Fetching market snapshot for {len(all_tickers)} tickers (price/volume compliance + mark-to-market)")
    try:
        snapshot = fetch_market_snapshot(all_tickers, asof_date=asof_date)
    except Exception as e:
        print(f"  WARNING: market snapshot fetch failed ({e})", file=sys.stderr)
        snapshot = {t: {"last_price": None, "avg_volume": None} for t in all_tickers}

    compliance = M.compliance_panel(
        cumulative_return_pct, dd["max_drawdown_pct"], sizing_result.sized_trades,
        sizing_result.pct_skipped, snapshot, cfg["compliance"],
    )

    sized_by_id = {id(s.trade): s for s in sizing_result.sized_trades}
    open_positions = []
    for t in open_trades(trades):
        snap = snapshot.get(t.ticker, {"last_price": None, "avg_volume": None})
        live_price = snap["last_price"]
        if live_price is not None:
            price_move_pct = 100 * (live_price - t.entry_price) / t.entry_price
            # A short profits from a price DROP -- the raw price move above
            # is the opposite sign of the position's own P&L for a short.
            unrealized_pct = -price_move_pct if t.position == "short" else price_move_pct
        else:
            unrealized_pct = t.pct_change
        sized = sized_by_id.get(id(t))
        notional = sized.notional if sized and sized.admitted else 0.0
        open_positions.append({
            "ticker": t.ticker,
            "side": t.position,
            "open_datetime": t.open_dt.isoformat(),
            "scheduled_close_datetime": t.close_dt.isoformat(),
            "entry_price": t.entry_price,
            "live_price": live_price,
            "unrealized_pct": round(unrealized_pct, 3),
            "notional_dollars": round(notional, 2),
            "unrealized_dollars": round(notional * unrealized_pct / 100, 2),
            "capital_admitted": bool(sized and sized.admitted),
            "status": t.status,
        })

    today_tickers = sorted(set(benchmarks) | {
        p["ticker"] for p in open_positions if not p["ticker"].startswith("SIM-")
    })
    print(f"Fetching intraday 'today' data for {len(today_tickers)} tickers")
    try:
        intraday = fetch_intraday_today(today_tickers, asof_date=asof_date)
    except Exception as e:
        print(f"  WARNING: intraday fetch failed ({e})", file=sys.stderr)
        intraday = pd.DataFrame()

    today_chart = {"timestamps": [], "series": {}}
    if not intraday.empty:
        idx = intraday.index
        today_chart["timestamps"] = [ts.isoformat() for ts in idx]
        # Every fetched ticker, not just benchmarks -- open positions' own
        # intraday series were already being fetched above (today_tickers
        # includes them) but discarded here; the per-position mini-chart
        # needs them exposed too.
        for col in intraday.columns:
            today_chart["series"][col] = [
                round(float(v), 3) if pd.notna(v) else None for v in intraday[col]
            ]
        # Notional-weighted portfolio move so far today, over the open book.
        weights = {
            p["ticker"]: p["notional_dollars"] for p in open_positions
            if p["notional_dollars"] > 0 and p["ticker"] in intraday.columns
        }
        if weights:
            port_series = []
            for ts in idx:
                num, denom = 0.0, 0.0
                for tkr, w in weights.items():
                    val = intraday.loc[ts, tkr]
                    if pd.notna(val):
                        num += w * val
                        denom += w
                port_series.append(round(num / denom, 3) if denom > 0 else None)
            today_chart["series"]["portfolio"] = port_series

    dashboard = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "account_size": account_size,
            "position_weight_pct": weight_pct,
            "benchmarks": benchmarks,
        },
        "summary": {
            "cumulative_return_pct": round(cumulative_return_pct, 3),
            "equity_dollars": round(float(equity_aligned.iloc[-1]), 2) if len(equity_aligned) else account_size,
            "max_drawdown": dd,
            "sharpe_ratio": round(M.sharpe_ratio(strategy_daily_ret_aligned), 3),
            "sortino_ratio": round(M.sortino_ratio(strategy_daily_ret_aligned), 3),
            "alpha_beta": alpha_beta_by_bench,
            **{k: v for k, v in trade_stats.items()},
        },
        "historic": {
            "dates": [str(d.date()) for d in trading_days],
            "equity_normalized": normalized_series,
            "equity_strategy_dollars": [round(float(v), 2) for v in equity_aligned.tolist()] if len(equity_aligned) else [],
            "drawdown_pct": [round(float(v) * 100, 3) for v in ((equity_aligned - equity_aligned.cummax()) / equity_aligned.cummax()).tolist()] if len(equity_aligned) else [],
            "capital_utilization_pct": [round(float(v) * 100, 2) for v in util_aligned.tolist()] if len(util_aligned) else [],
            "open_position_count": [int(round(float(v))) for v in pos_count_aligned.tolist()] if len(pos_count_aligned) else [],
            "long_return_pct": [round(float(v), 3) for v in long_return_aligned.tolist()],
            "short_return_pct": [round(float(v), 3) for v in short_return_aligned.tolist()],
            "rolling_sharpe_60d": {
                "dates": [str(d.date()) for d in roll_sharpe_60.index],
                "values": [round(float(v), 3) for v in roll_sharpe_60.tolist()],
            },
        },
        "ticker_concentration": ticker_conc,
        "trade_returns_pct": closed_returns_pct,
        "open_positions": open_positions,
        "compliance": compliance,
        "today": today_chart,
    }

    out_path = os.path.join(output_dir, "dashboard.json")
    with open(out_path, "w") as f:
        json.dump(dashboard, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build(args.config)
