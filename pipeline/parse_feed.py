"""Parse the raw trade-log CSV feed into normalized trade records.

The feed has two eras. Early rows populate only open_ask/close_bid. Later
rows also populate open_bid/close_ask. Both eras use open_ask as entry cost
and close_bid as exit proceeds (long-only trades). This is why `change`
reconciles as a % return.

`status` decides open vs. closed. Use exactly "open" or "closed"
(case-insensitive). Clean up a real broker feed's status column to these
two values before parsing it here.
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
        # Normalize to UTC. The feed mixes -04:00/-05:00 offsets across DST
        # transitions. Pandas errors on a DatetimeIndex built from mixed
        # fixed offsets unless every timestamp shares one tz.
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
