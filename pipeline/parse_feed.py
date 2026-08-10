"""Parse the raw trade-log CSV feed into normalized trade records.

The feed has two eras. Early rows populate only open_ask/close_bid. Later
rows also populate open_bid/close_ask. A long trade reconciles on open_ask
(entry cost) and close_bid (exit proceeds) -- realistic execution against
the spread, buying at the ask and selling at the bid. A short trade
reconciles the other way: open_bid (entry proceeds -- selling to open) and
close_ask (exit cost -- buying to cover). No real trade has been short yet,
so this side is only exercised by the synthetic generators; an old-era row
missing the bid/ask a short needs raises rather than reconciling on the
wrong side.

`status` decides open vs. closed: "open" or "closed" (case-insensitive).
Any other status value is a genuine data problem and raises.
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
        else:
            raise ValueError(
                f"Unrecognized status {row['status']!r} for {row['ticker']} "
                f"opened {row['open_datetime']} -- clean the feed's status "
                f"column to 'open'/'closed' before parsing it here."
            )

        position = str(row["position"]).strip()
        is_short = position.lower() == "short"
        entry_col = "open_bid" if is_short else "open_ask"
        exit_col = "close_ask" if is_short else "close_bid"

        entry_raw = row[entry_col]
        if pd.isna(entry_raw):
            raise ValueError(
                f"Short trade for {row['ticker']} opened {row['open_datetime']} "
                f"has no {entry_col} -- a short needs its own side of the "
                f"spread to reconcile; this row only has the long-side columns."
            )
        entry_price = float(entry_raw)

        exit_price = None
        if not is_open:
            exit_raw = row[exit_col]
            if pd.isna(exit_raw):
                raise ValueError(
                    f"Short trade for {row['ticker']} closed {row['close_datetime']} "
                    f"has no {exit_col} -- a short needs its own side of the "
                    f"spread to reconcile; this row only has the long-side columns."
                )
            exit_price = float(exit_raw)

        # Normalize to UTC. The feed mixes -04:00/-05:00 offsets across DST
        # transitions. Pandas errors on a DatetimeIndex built from mixed
        # fixed offsets unless every timestamp shares one tz.
        trades.append(
            Trade(
                open_dt=pd.Timestamp(row["open_datetime"]).tz_convert("UTC"),
                close_dt=pd.Timestamp(row["close_datetime"]).tz_convert("UTC"),
                ticker=str(row["ticker"]).strip(),
                entry_price=entry_price,
                exit_price=exit_price,
                pct_change=float(row["change"]),
                is_open=is_open,
                status=str(row["status"]).strip(),
                position=position,
            )
        )
    trades.sort(key=lambda t: t.open_dt)
    return trades


def closed_trades(trades: list[Trade]) -> list[Trade]:
    return [t for t in trades if not t.is_open]


def open_trades(trades: list[Trade]) -> list[Trade]:
    return [t for t in trades if t.is_open]
