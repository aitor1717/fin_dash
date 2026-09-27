"""Orchestrates the pipeline: feed -> sized trades -> metrics -> docs/data JSON.

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

from parse_feed import parse_feed, open_trades, cap_premature_close_dates
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
    trades = cap_premature_close_dates(trades, pd.Timestamp.now(tz="UTC"))
    print(f"  {len(trades)} trades, {len(open_trades(trades))} currently open")

    print(f"Sizing trades (account_size={account_size}, weight={weight_pct}%)")
    sizing_result = size_trades(trades, account_size, weight_pct)
    print(f"  {sizing_result.pct_skipped:.1f}% of trades skipped (capital cap)")

    pnl = M.daily_pnl(sizing_result.sized_trades, tz)
    equity = M.equity_curve(pnl, account_size)

    # Long-only and short-only slices, isolated as if each were the only book
    # traded. This filters the sized trades already computed above by side --
    # a re-aggregation, not a new calculation.
    long_sized = [s for s in sizing_result.sized_trades if s.trade.position == "long"]
    short_sized = [s for s in sizing_result.sized_trades if s.trade.position == "short"]
    pnl_long = M.daily_pnl(long_sized, tz)
    pnl_short = M.daily_pnl(short_sized, tz)

    start = trades[0].open_dt.tz_convert(tz).date()
    today = datetime.now(timezone.utc).date()
    # freeze_asof: for a static book (see config.sample.yaml) that never
    # gets new trades, pin "today" to the feed's last real activity, not
    # real wall-clock time. Otherwise the benchmark range and live
    # mark-to-market drift forward on every later run while the feed stays
    # frozen. Built from actual opens/closes only -- not an open position's
    # own scheduled close_dt, which can land arbitrarily far in the future.
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

    # Market snapshot + open positions' live unrealized $ -- fetched here
    # (moved up from originally coming after the benchmark/equity-curve
    # block below) so the "Live mark-to-market" adjustment further down can
    # use it. asof_date/all_tickers only need `trades`, not anything
    # derived from the benchmark fetch, so this only changes *when* the
    # snapshot is fetched, not what's fetched.
    # Synthetic what-if tickers (see simulate_scenario.py) aren't real
    # symbols. Skip them here instead of spending one failed network
    # lookup each.
    all_tickers = sorted(set(t.ticker for t in trades if not t.ticker.startswith("SIM-")))
    print(f"Fetching market snapshot for {len(all_tickers)} tickers (price/volume compliance + mark-to-market)")
    try:
        snapshot = fetch_market_snapshot(all_tickers, asof_date=asof_date)
    except Exception as e:
        print(f"  WARNING: market snapshot fetch failed ({e})", file=sys.stderr)
        snapshot = {t: {"last_price": None, "avg_volume": None} for t in all_tickers}

    # Open positions + their live unrealized $, computed before the equity
    # curve below so today's equity-curve point can include it. The equity
    # curve is otherwise realized-closes-only (M.daily_pnl only counts
    # admitted, *closed* trades grouped by close_dt) -- an open position
    # that's up or down big today would otherwise never move the main chart
    # until it actually closes, days or weeks later.
    sized_by_id = {id(s.trade): s for s in sizing_result.sized_trades}
    open_positions = []
    for t in open_trades(trades):
        snap = snapshot.get(t.ticker, {"last_price": None, "avg_volume": None})
        live_price = snap["last_price"]
        if live_price is not None:
            price_move_pct = 100 * (live_price - t.entry_price) / t.entry_price
            # A short profits from a price drop. The raw price move above
            # has the opposite sign of the position's own P&L for a short.
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
        })
    # Only admitted positions carry real notional -- matches how
    # equity/utilization elsewhere only count admitted trades, so this
    # can't move the curve by more than what's actually sized into the book.
    total_unrealized_dollars = sum(p["unrealized_dollars"] for p in open_positions if p["capital_admitted"])

    print(f"Fetching benchmark history for {benchmarks} ({start} to {end})")
    try:
        bench_close = fetch_benchmark_history(benchmarks, start, end)
        bench_close.index = pd.DatetimeIndex(bench_close.index).tz_localize(None).normalize()
        # yfinance can return an all-NaN column on a failed/rate-limited fetch
        # instead of raising. Drop those columns so downstream code treats
        # them as unavailable, instead of propagating NaN (not valid JSON)
        # into the dashboard.
        bench_close = bench_close.dropna(axis=1, how="all")
        bench_ret = benchmark_daily_returns(bench_close)
    except Exception as e:
        print(f"  WARNING: benchmark fetch failed ({e}); historic chart will be strategy-only", file=sys.stderr)
        bench_close = pd.DataFrame()
        bench_ret = pd.DataFrame()

    trading_days = bench_close.index if len(bench_close) else pd.date_range(start, end, freq="D")
    # yfinance can lag posting the current session's own daily bar (e.g.
    # running mid-session or shortly after close). If `end` (today, or the
    # frozen asof date) isn't in the benchmark's own index yet, any trade
    # whose close just got capped to `end` by cap_premature_close_dates
    # above would silently fall outside every series reindexed onto
    # trading_days -- not deferred to a future date, just dropped, until a
    # later run finally sees the bar and the P&L jumps in retroactively.
    # Union it in explicitly rather than leaving it to chance.
    end_ts = pd.Timestamp(end)
    if end_ts not in trading_days:
        trading_days = trading_days.append(pd.DatetimeIndex([end_ts])).sort_values()

    equity_naive = equity.copy()
    equity_naive.index = pd.DatetimeIndex(equity_naive.index).tz_localize(None).normalize()
    equity_aligned = equity_naive.reindex(trading_days).ffill().fillna(account_size)

    # Live mark-to-market: fold currently-open positions' unrealized $
    # (computed above) into today's own point on the curve -- recomputed
    # fresh every run from the live snapshot, same as open_positions itself.
    # Only the LAST point (today) is touched; every prior day is
    # realized-only and stays exactly as it was once a trade actually
    # closed on it -- see M.daily_pnl.
    if len(equity_aligned):
        equity_aligned.iloc[-1] = equity_aligned.iloc[-1] + total_unrealized_dollars

    cumulative_return_pct = 100 * (equity_aligned.iloc[-1] - account_size) / account_size if len(equity_aligned) else 0.0

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
    # trading-day-aligned one. A trade's close_datetime can land on a
    # weekend. Reindexing onto trading days only would drop those P&L
    # events from the running peak/trough and understate drawdown. The
    # aligned series below still drives charting (x-axis matches the
    # benchmark overlay) and alpha/beta (must line up with trading days).
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

    compliance = M.compliance_panel(
        cumulative_return_pct, dd["max_drawdown_pct"], sizing_result.sized_trades,
        sizing_result.pct_skipped, snapshot, cfg["compliance"],
    )

    # Recently-closed tickers, not just currently-open ones: a trade that
    # closed earlier in the session being measured needs its own intraday
    # price path (to mark it up to its close time) even though it's no
    # longer in open_positions. A calendar-day window rather than "closed
    # today" specifically, since which date IS "today" (the session
    # fetch_intraday_today actually returns) isn't known until after that
    # fetch -- see its own before-the-open/weekend-close_dt handling.
    recent_cutoff = (
        pd.Timestamp(asof_date, tz="UTC") if asof_date is not None else pd.Timestamp.now(tz="UTC")
    ) - pd.Timedelta(days=5)
    recently_closed_tickers = {
        t.ticker for t in trades
        if not t.is_open and t.close_dt >= recent_cutoff and not t.ticker.startswith("SIM-")
    }
    today_tickers = sorted(set(benchmarks) | {
        p["ticker"] for p in open_positions if not p["ticker"].startswith("SIM-")
    } | recently_closed_tickers)
    print(f"Fetching intraday 'today' data for {len(today_tickers)} tickers")
    try:
        intraday_prices, session_date = fetch_intraday_today(today_tickers, tz, asof_date=asof_date)
    except Exception as e:
        print(f"  WARNING: intraday fetch failed ({e})", file=sys.stderr)
        intraday_prices, session_date = pd.DataFrame(), None

    today_chart = {"timestamps": [], "series": {}, "session_date": None, "is_current_session": False}
    if session_date is not None and not intraday_prices.empty:
        idx = intraday_prices.index
        today_chart["timestamps"] = [ts.isoformat() for ts in idx]
        today_chart["session_date"] = str(session_date)
        effective_today = asof_date if asof_date is not None else pd.Timestamp.now(tz=tz).date()
        today_chart["is_current_session"] = session_date == effective_today

        # Every fetched ticker, not just benchmarks. today_tickers already
        # includes open positions' own intraday series; expose them here too
        # for the per-position mini-chart. % from that ticker's own first
        # bar of the session -- distinct from the portfolio series below,
        # which is marked from the previous close / each position's entry,
        # not from the session's first bar.
        for col in intraday_prices.columns:
            series = intraday_prices[col].reindex(idx)
            valid = series.dropna()
            if valid.empty:
                continue
            first = valid.iloc[0]
            today_chart["series"][col] = [
                round(100 * (v - first) / first, 3) if pd.notna(v) else None for v in series
            ]

        # Account day change: dollar move since the previous session's
        # close, as a % of equity at that close -- a broker's "day change",
        # not the notional-weighted move-since-session-open this replaced.
        prior_equity = M.prior_close_equity(equity_naive, session_date, account_size)
        prev_close_by_ticker: dict[str, float | None] = {}
        for tkr in today_tickers:
            daily_closes = snapshot.get(tkr, {}).get("daily_closes") or {}
            dates_before = [d for d in daily_closes if pd.Timestamp(d).date() < session_date]
            prev_close_by_ticker[tkr] = (
                daily_closes[max(dates_before, key=lambda d: pd.Timestamp(d))] if dates_before else None
            )
        day_change_dollars = M.account_day_change_dollars(
            sizing_result.sized_trades, idx, intraday_prices, prev_close_by_ticker, session_date, tz,
        )
        today_chart["series"]["portfolio"] = [
            round(100 * v / prior_equity, 3) if prior_equity else None
            for v in day_change_dollars.tolist()
        ]

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
