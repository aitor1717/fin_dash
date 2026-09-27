"""Benchmark history and live-price/volume snapshots via yfinance.

Two kinds of data are needed:
- Benchmark daily closes over the full feed date range (historic view, alpha/beta).
- A snapshot (last price + ~3mo avg volume) for every ticker traded. Used to
  mark open positions to market and for the price/volume compliance floors.
  Fetched as one batched call, not per-ticker: a feed can touch 200+ tickers.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def _ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV columns out of a group_by="ticker" download.

    yfinance returns MultiIndex columns keyed by ticker regardless of how
    many tickers were requested -- a single-ticker request still comes back
    this way, not flattened. Indexing straight into `data` for that case
    (an earlier version of this code did, on the assumption a lone ticker
    wouldn't be grouped) silently returns nothing: `data["Close"]` doesn't
    exist when the top-level column key is the ticker, not the field.
    """
    return data[ticker] if isinstance(data.columns, pd.MultiIndex) else data


def fetch_benchmark_history(tickers: list[str], start, end) -> pd.DataFrame:
    """Daily close price per benchmark ticker, indexed by date."""
    data = yf.download(
        tickers, start=start, end=end, interval="1d",
        progress=False, auto_adjust=True,
    )
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all")


def benchmark_daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change().dropna(how="all")


def fetch_intraday_today(tickers: list[str], tz: str, asof_date=None) -> tuple[pd.DataFrame, object]:
    """Raw 5-minute close prices for each ticker's most recent trading
    session. Columns are tickers, index is intraday timestamps for that one
    session; returns (prices, session_date) -- session_date is the calendar
    date (in `tz`) that session falls on, or None if nothing came back.

    Raw prices, not a % series: different callers need different reference
    points (first bar of the session for benchmarks/per-ticker mini-charts;
    the previous session's close or a trade's own entry price for the
    account day-change -- see build.py and metrics.account_day_change_dollars).

    asof_date pins this to one past session instead of real wall-clock
    "today". Needed for a frozen/static book (see build.py's freeze_asof):
    re-running the pipeline on a later day must reproduce the same "today"
    view, not drift forward with live prices for a book that stopped
    trading. Works only within yfinance's ~60-day 5-minute retention
    window; an out-of-window asof_date returns an empty frame.

    asof_date may fall on a non-trading day -- it comes from the feed's
    last open/close timestamp, and a trade can close on a weekend. Also,
    before the market opens on a live trading day, the most recent
    complete session is still yesterday's. Both cases are why this always
    pulls a trailing window and keeps only the single most recent session
    in it (at or before asof_date when given), rather than trusting
    yfinance's own "1 day" framing to line up with one session.
    """
    tickers = sorted(set(tickers))
    end = (asof_date if asof_date is not None else pd.Timestamp.now(tz=tz).normalize().date())
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    start_ts = end_ts - pd.Timedelta(days=8)
    data = yf.download(
        tickers, start=str(start_ts.date()), end=str(end_ts.date()),
        interval="5m", progress=False, auto_adjust=True, group_by="ticker", threads=False,
    )
    if not len(data):
        return pd.DataFrame(), None
    idx_local = pd.DatetimeIndex(data.index).tz_convert(tz)
    session_date = idx_local.normalize().max().date()
    data = data[idx_local.normalize() == pd.Timestamp(session_date, tz=tz)]

    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            sub = _ticker_frame(data, t)
            closes = sub["Close"].dropna()
            if len(closes):
                out[t] = closes
        except Exception:
            continue
    return pd.DataFrame(out), session_date


def fetch_market_snapshot(tickers: list[str], asof_date=None) -> dict[str, dict]:
    """Last close price and ~3-month average volume for each ticker.

    asof_date pins "last price" to a specific past close, and the volume
    average to the ~3 months ending there, instead of a live snapshot. Same
    reasoning as fetch_intraday_today: a frozen book shouldn't mark itself
    to market against prices that keep moving after it stopped trading.
    """
    tickers = sorted(set(tickers))
    # threads=False: yfinance's local sqlite cache isn't safe under concurrent
    # per-ticker requests and silently drops tickers with "database is locked".
    if asof_date is not None:
        data = yf.download(
            tickers, start=str(asof_date - pd.Timedelta(days=95)), end=str(asof_date + pd.Timedelta(days=1)),
            interval="1d", progress=False, auto_adjust=True, group_by="ticker", threads=False,
        )
    else:
        data = yf.download(
            tickers, period="3mo", interval="1d",
            progress=False, auto_adjust=True, group_by="ticker", threads=False,
        )
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            sub = _ticker_frame(data, t)
            closes = sub["Close"].dropna()
            volumes = sub["Volume"].dropna()
            out[t] = {
                "last_price": float(closes.iloc[-1]) if len(closes) else None,
                "avg_volume": float(volumes.mean()) if len(volumes) else None,
                # Every daily close in the fetched window, keyed by ISO
                # date string. A broker marks positions at the official
                # close, so the account day-change (build.py /
                # metrics.account_day_change_dollars) needs the previous
                # session's close as its reference point for anything held
                # overnight -- not just the single latest one.
                "daily_closes": {str(d.date()): float(c) for d, c in closes.items()},
            }
        except Exception:
            out[t] = {"last_price": None, "avg_volume": None, "daily_closes": {}}

    # The daily bar above is regular-session-only and never updates again
    # once it prints at the close, so a genuinely live snapshot (not a
    # frozen/backtested one) goes stale for the rest of the trading day
    # while the stock keeps moving. Overlay the last available *regular-
    # session* minute bar on top of last_price only -- avg_volume stays
    # daily-based. prepost=False deliberately: a broker values positions at
    # the official close, so an extended-hours print would make this drift
    # away from the real account every evening. When the market is closed,
    # the regular-session daily close above is simply what stays on screen.
    # Best-effort: a failed live fetch just leaves the daily close in place.
    if asof_date is None:
        try:
            live = yf.download(
                tickers, period="1d", interval="1m", prepost=False,
                progress=False, auto_adjust=True, group_by="ticker", threads=False,
            )
            for t in tickers:
                try:
                    sub = _ticker_frame(live, t)
                    closes = sub["Close"].dropna()
                    if len(closes):
                        out[t]["last_price"] = float(closes.iloc[-1])
                except Exception:
                    continue
        except Exception:
            pass

    return out
