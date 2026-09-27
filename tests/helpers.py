"""Shared synthetic-data builders for the test suite. No network, no real
trade data -- round numbers and invented tickers only, same spirit as
pipeline/check_integrity.py.
"""
from __future__ import annotations

import pandas as pd

from parse_feed import Trade
from sizing import SizedTrade


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_trade(
    open_dt: str, close_dt: str, ticker: str, pct_change: float,
    is_open: bool = False, position: str = "long", entry_price: float = 100.0,
) -> Trade:
    exit_price = None if is_open else entry_price * (1 + pct_change / 100.0)
    return Trade(
        open_dt=ts(open_dt),
        close_dt=ts(close_dt),
        ticker=ticker,
        entry_price=entry_price,
        exit_price=exit_price,
        pct_change=pct_change,
        is_open=is_open,
        status="open" if is_open else "closed",
        position=position,
    )


def make_sized(trade: Trade, notional: float = 5000.0, admitted: bool = True) -> SizedTrade:
    pnl = 0.0 if (not admitted or trade.is_open) else notional * (trade.pct_change / 100.0)
    return SizedTrade(trade=trade, admitted=admitted, notional=notional if admitted else 0.0, pnl_dollars=pnl)
