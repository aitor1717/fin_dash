"""Parse the raw trade-log CSV feed into normalized trade records.

The feed schema (see feed/example_feed.csv) has two eras: early rows only
populate open_ask/close_bid (single-sided price), later rows populate the
full open_bid/open_ask/close_bid/close_ask. In both eras the entry cost is
open_ask and the exit proceeds are close_bid (these are long-only trades),
which is what makes the `change` column reconcile as a % return.

`status` is the authoritative signal that a trade is still open: standardized
to exactly "open" or "closed" (case-insensitive) in the synthetic sample
feed. A real broker-exported feed whose status values don't map that cleanly
would need its own status column cleaned up to match before parsing it here
-- there's no longer a close_bid-based fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Trade:
    open_dt: pd.Timestamp
    close_dt: pd.Timestamp  # actual close time if closed, else scheduled/last-seen close time
    ticker: str
    entry_price: float
    exit_price: float | None  # None while open
    pct_change: float  # realized % if closed, last-marked unrealized % if open
    is_open: bool
    status: str
    position: str


def parse_feed(csv_path: str) -> list[Trade]:
    df = pd.read_csv(csv_path)
    trades: list[Trade] = []
    for _, row in df.iterrows():
        is_open = str(row["status"]).strip().lower() == "open"
        exit_price = None if is_open else float(row["close_bid"])
        # Normalize to UTC: the feed mixes -04:00/-05:00 offsets across DST
        # transitions, and a DatetimeIndex built from mixed fixed offsets
        # errors in pandas unless every timestamp shares one tz.
        trades.append(
            Trade(
                open_dt=pd.Timestamp(row["open_datetime"]).tz_convert("UTC"),
                close_dt=pd.Timestamp(row["close_datetime"]).tz_convert("UTC"),
                ticker=str(row["ticker"]).strip(),
                entry_price=float(row["open_ask"]),
                exit_price=exit_price,
                pct_change=float(row["change"]),
                is_open=is_open,
                status=str(row["status"]).strip(),
                position=str(row["position"]).strip(),
            )
        )
    trades.sort(key=lambda t: t.open_dt)
    return trades


def closed_trades(trades: list[Trade]) -> list[Trade]:
    return [t for t in trades if not t.is_open]


def open_trades(trades: list[Trade]) -> list[Trade]:
    return [t for t in trades if t.is_open]
