"""Buying-power-capped position sizing allocator.

The feed has no size data. Trades overlap heavily -- up to ~49 concurrent
in the sample book. Each admitted trade risks a fixed weight `w` of
account_size. Trades are walked in open-time order; a new trade is skipped
(zero-filled) if it would push total open notional over 100% of
account_size at the moment it opens. This models a real buying-power
constraint, not unlimited leverage.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import pandas as pd

from parse_feed import Trade


@dataclass
class SizedTrade:
    trade: Trade
    admitted: bool
    notional: float  # 0.0 if skipped
    pnl_dollars: float  # 0.0 if skipped or still open


@dataclass
class SizingResult:
    sized_trades: list[SizedTrade]
    utilization: pd.Series  # open notional / account_size, indexed by open time
    pct_skipped: float


def size_trades(trades: list[Trade], account_size: float, weight_pct: float) -> SizingResult:
    notional_per_trade = account_size * (weight_pct / 100.0)
    trades_by_open = sorted(trades, key=lambda t: t.open_dt)

    open_heap: list[tuple[pd.Timestamp, float]] = []  # (close_dt, notional)
    current_exposure = 0.0
    sized: list[SizedTrade] = []
    util_index: list[pd.Timestamp] = []
    util_values: list[float] = []

    def release_until(t: pd.Timestamp) -> None:
        nonlocal current_exposure
        while open_heap and open_heap[0][0] <= t:
            _, notional = heapq.heappop(open_heap)
            current_exposure -= notional

    for trade in trades_by_open:
        release_until(trade.open_dt)
        if current_exposure + notional_per_trade > account_size + 1e-9:
            sized.append(SizedTrade(trade=trade, admitted=False, notional=0.0, pnl_dollars=0.0))
            continue
        current_exposure += notional_per_trade
        heapq.heappush(open_heap, (trade.close_dt, notional_per_trade))
        pnl = 0.0 if trade.is_open else notional_per_trade * (trade.pct_change / 100.0)
        sized.append(SizedTrade(trade=trade, admitted=True, notional=notional_per_trade, pnl_dollars=pnl))
        util_index.append(trade.open_dt)
        util_values.append(current_exposure / account_size)

    utilization = pd.Series(util_values, index=pd.DatetimeIndex(util_index)).sort_index()
    n_total = len(trades)
    n_skipped = sum(1 for s in sized if not s.admitted)
    pct_skipped = 100.0 * n_skipped / n_total if n_total else 0.0
    return SizingResult(sized_trades=sized, utilization=utilization, pct_skipped=pct_skipped)


def daily_position_count(sized_trades: list[SizedTrade]) -> pd.Series:
    """Timestamped snapshot of admitted (capital-cleared) concurrent open
    positions after each open/close event. Same event-sweep shape as
    utilization above, but a headcount, not a dollar exposure. Callers
    reindex/ffill this onto their own daily axis, same as utilization.
    """
    events = []
    for s in sized_trades:
        if not s.admitted:
            continue
        events.append((s.trade.open_dt, 1))
        events.append((s.trade.close_dt, -1))
    if not events:
        return pd.Series(dtype=float)
    events.sort(key=lambda e: (e[0], -e[1]))
    running = 0
    idx: list[pd.Timestamp] = []
    vals: list[int] = []
    for t, delta in events:
        running += delta
        idx.append(t)
        vals.append(running)
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()


def concurrency_series(trades: list[Trade]) -> list[int]:
    """Raw (uncapped) concurrent-open-position count sampled at every open event."""
    events = []
    for t in trades:
        events.append((t.open_dt, 1))
        events.append((t.close_dt, -1))
    events.sort(key=lambda e: (e[0], -e[1]))  # opens before closes at same instant
    running = 0
    samples = []
    for _, delta in events:
        running += delta
        if delta == 1:
            samples.append(running)
    return samples
