"""Benchmark history and live-price/volume snapshots via yfinance.

Two kinds of data are needed:
- Benchmark daily closes over the full feed date range (historic view, alpha/beta).
- A snapshot (last price + ~3mo avg volume) for every ticker traded, used both
  to mark open (pending) positions to market and for the price/volume
  compliance floors. Fetched as one batched call rather than per-ticker, since
  a feed can easily touch 200+ tickers.
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
    session (columns = tickers, index = intraday timestamps). Used to compare
    the open book's notional-weighted move so far today against benchmarks on
    the same, apples-to-apples % basis.

    asof_date pins this to a SPECIFIC past session instead of whatever
    "today" happens to be in real wall-clock time -- needed for a frozen/
    static book (see build.py's freeze_asof), where re-running the pipeline
    on a later real calendar day should reproduce the exact same "today"
    view, not silently drift forward with real intraday prices for a book
    that stopped trading on a fixed date. Only works within yfinance's
    5-minute retention window (~60 days); an out-of-window asof_date just
    yields an empty frame, same as any other fetch failure here.

    asof_date itself isn't guaranteed to be a trading day -- it's derived
    from the feed's own last open/close timestamp (build.py), and a trade
    can close on a weekend even though the market can't. A single-day query
    on a non-trading date returns nothing, so this pulls a trailing week and
    keeps only the most recent session at or before asof_date -- the same
    "most recent session" behavior period="1d" gives for real "today".
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

    asof_date pins "last price" to a specific past close and the volume
    average to the ~3 months ending there, instead of a literal live
    snapshot -- same reasoning as fetch_intraday_today's asof_date: a frozen
    book shouldn't mark itself to market against real-time prices that keep
    moving on days after the book itself stopped trading.
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
    return out
