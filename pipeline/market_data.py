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


def fetch_intraday_today(tickers: list[str], asof_date=None) -> pd.DataFrame:
    """5-minute % change from each ticker's first bar of its most recent
    session. Columns are tickers, index is intraday timestamps. Compares the
    open book's notional-weighted move today against benchmarks on the same
    % basis.

    asof_date pins this to one past session instead of real wall-clock
    "today". Needed for a frozen/static book (see build.py's freeze_asof):
    re-running the pipeline on a later day must reproduce the same "today"
    view, not drift forward with live prices for a book that stopped
    trading. Works only within yfinance's ~60-day 5-minute retention
    window; an out-of-window asof_date returns an empty frame.

    asof_date may fall on a non-trading day -- it comes from the feed's
    last open/close timestamp, and a trade can close on a weekend. A
    single-day query then returns nothing, so this pulls a trailing week
    and keeps only the most recent session at or before asof_date.
    """
    tickers = sorted(set(tickers))
    if asof_date is not None:
        data = yf.download(
            tickers, start=str(asof_date - pd.Timedelta(days=7)), end=str(asof_date + pd.Timedelta(days=1)),
            interval="5m", progress=False, auto_adjust=True, group_by="ticker", threads=False,
        )
        if len(data):
            last_session = pd.DatetimeIndex(data.index).normalize().max()
            data = data[pd.DatetimeIndex(data.index).normalize() == last_session]
    else:
        data = yf.download(
            tickers, period="1d", interval="5m",
            progress=False, auto_adjust=True, group_by="ticker", threads=False,
        )
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            sub = data[t] if len(tickers) > 1 else data
            closes = sub["Close"].dropna()
            if len(closes) == 0:
                continue
            first = closes.iloc[0]
            out[t] = 100 * (closes - first) / first
        except Exception:
            continue
    return pd.DataFrame(out)


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
            sub = data[t] if len(tickers) > 1 else data
            closes = sub["Close"].dropna()
            volumes = sub["Volume"].dropna()
            out[t] = {
                "last_price": float(closes.iloc[-1]) if len(closes) else None,
                "avg_volume": float(volumes.mean()) if len(volumes) else None,
            }
        except Exception:
            out[t] = {"last_price": None, "avg_volume": None}

    # The daily bar above is regular-session-only and never updates again
    # once it prints at the close, so a genuinely live snapshot (not a
    # frozen/backtested one) goes stale for the rest of the day/evening
    # while the stock keeps moving in extended-hours trading. Overlay the
    # last available extended-hours minute bar on top of last_price only --
    # avg_volume stays regular-session-based. Best-effort: a failed live
    # fetch just leaves the daily close in place.
    if asof_date is None:
        try:
            live = yf.download(
                tickers, period="1d", interval="1m", prepost=True,
                progress=False, auto_adjust=True, group_by="ticker", threads=False,
            )
            for t in tickers:
                try:
                    sub = live[t] if len(tickers) > 1 else live
                    closes = sub["Close"].dropna()
                    if len(closes):
                        out[t]["last_price"] = float(closes.iloc[-1])
                except Exception:
                    continue
        except Exception:
            pass

    return out
