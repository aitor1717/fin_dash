"""Parse the raw trade-log CSV feed into normalized trade records.

The feed has two eras. Early rows populate only open_ask/close_bid. Later
rows also populate open_bid/close_ask. Both eras use open_ask as entry cost
and close_bid as exit proceeds (long-only trades). This is why `change`
reconciles as a % return.

`status` decides open vs. closed: "open" or "closed" (case-insensitive).
A real broker export can also carry a stray third value, "register" --
not a real lifecycle state, just uncleaned data -- which is normalized
by content instead of failing: a numeric `close_bid` means the trade is
actually closed, the literal "pending" placeholder means it's still
open. Any other status value is a genuine data problem and raises.
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
        status_norm = str(row["status"]).strip().lower()
        if status_norm == "open":
            is_open = True
        elif status_norm == "closed":
            is_open = False
        elif status_norm == "register":
            # Stray uncleaned status, not a real third lifecycle state --
            # resolve it from close_bid's content instead.
            is_open = str(row["close_bid"]).strip().lower() == "pending"
        else:
            raise ValueError(
                f"Unrecognized status {row['status']!r} for {row['ticker']} "
                f"opened {row['open_datetime']} -- clean the feed's status "
                f"column to 'open'/'closed' before parsing it here."
            )
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
