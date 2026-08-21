"""Standalone correctness check, independent of any real/live data.

"Runs without crashing" and "produces the mathematically correct number"
are different claims. build.py running clean, or a property being set,
proves the first -- not the second. This builds a small hand-picked
synthetic trade set (round numbers, so every expected output can be
verified with a calculator) and checks the pipeline's own sizing/metrics
functions against:

  1. A second, independently written reimplementation of the same math
     (plain-Python loops, not pandas -- a genuinely different code path,
     not a re-invocation of sizing.py/metrics.py's own logic).
  2. Structural invariants that must hold regardless of the specific data
     (e.g. the published cumulative return must equal what you'd compute
     by hand from the published equity curve).
  3. A regression guard on cap_premature_close_dates (pipeline/parse_feed.py):
     a "closed" trade's own timestamp gets corrected, but its status,
     is_open, and exit_price must never be touched by that correction --
     an earlier attempt at that fix flipped is_open to True instead, which
     silently discarded a real, already-realized close (see git history /
     the backport notes for pipeline/parse_feed.py::cap_premature_close_dates).

No network access, no real trade data. Run directly: python pipeline/check_integrity.py
"""
from __future__ import annotations

import datetime as dt
import statistics
import sys

import pandas as pd

from parse_feed import Trade, cap_premature_close_dates
from sizing import size_trades, SizedTrade
import metrics as M

ACCOUNT_SIZE = 10_000.0
WEIGHT_PCT = 50.0  # notional_per_trade = 5000, so 2 concurrent trades exactly fill the cap
TZ = "UTC"
FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(f"{label}: {detail}")


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_trade(open_dt, close_dt, ticker, pct_change, is_open=False, position="long") -> Trade:
    return Trade(
        open_dt=ts(open_dt),
        close_dt=ts(close_dt),
        ticker=ticker,
        entry_price=100.0,
        exit_price=None if is_open else 100.0 * (1 + pct_change / 100.0),
        pct_change=pct_change,
        is_open=is_open,
        status="open" if is_open else "closed",
        position=position,
    )


# ---------------------------------------------------------------------------
# Synthetic dataset, hand-picked so every expected output is calculator-
# verifiable. weight_pct=50% on a $10,000 account means exactly two
# concurrent trades fill the buying-power cap.
# ---------------------------------------------------------------------------

def build_dataset(now: pd.Timestamp) -> list[Trade]:
    return [
        # T1, T2 overlap and exactly fill the cap (5000 + 5000 = 10000).
        make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0),   # admitted, pnl=+500
        make_trade("2024-01-02", "2024-01-06", "T2", pct_change=-4.0),   # admitted, pnl=-200
        # T3 opens while T1+T2 are both still open (10000 already committed)
        # -- pushing to 15000 exceeds the cap, so it must be skipped, not
        # partially sized.
        make_trade("2024-01-03", "2024-01-04", "T3", pct_change=50.0),   # skipped, pnl=0
        # T4 opens after T1+T2 have both closed (exposure back to 0).
        make_trade("2024-01-07", "2024-01-08", "T4", pct_change=8.0, position="short"),  # admitted, pnl=+400
        # T5: a "closed" trade whose recorded close_dt is still in the
        # future relative to `now` -- the item-3 bug. cap_premature_close_dates
        # must bucket its P&L on `now`'s date, not silently defer it to 2030.
        make_trade("2024-01-09", "2030-01-01", "T5", pct_change=15.0),   # admitted, pnl=+750, capped to `now`
    ]


# ---------------------------------------------------------------------------
# Independent reimplementation -- plain Python, no pandas, no reuse of
# sizing.py's heap sweep or metrics.py's groupby/cumsum.
# ---------------------------------------------------------------------------

def independent_size(trades: list[Trade], account_size: float, weight_pct: float):
    notional = account_size * (weight_pct / 100.0)
    by_open = sorted(trades, key=lambda t: t.open_dt)
    admitted_intervals: list[tuple] = []  # (open_dt, close_dt) of admitted trades
    results = {}
    for t in by_open:
        exposure = sum(
            notional for (o, c) in admitted_intervals if o <= t.open_dt < c
        )
        if exposure + notional > account_size + 1e-9:
            results[id(t)] = (False, 0.0, 0.0)
        else:
            admitted_intervals.append((t.open_dt, t.close_dt))
            pnl = 0.0 if t.is_open else notional * (t.pct_change / 100.0)
            results[id(t)] = (True, notional, pnl)
    return results


def independent_equity_curve(trades: list[Trade], admit_results: dict, account_size: float):
    daily_pnl: dict[dt.date, float] = {}
    for t in trades:
        admitted, _, pnl = admit_results[id(t)]
        if not admitted or t.is_open:
            continue
        d = t.close_dt.tz_convert(TZ).normalize().date()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl

    if not daily_pnl:
        return {}, 0.0, None, None

    start, end = min(daily_pnl), max(daily_pnl)
    equity: dict[dt.date, float] = {}
    running = account_size
    d = start
    while d <= end:
        running += daily_pnl.get(d, 0.0)
        equity[d] = running
        d += dt.timedelta(days=1)

    peak = None
    peak_val = float("-inf")
    trough_val = float("inf")
    trough = None
    running_peak_val = float("-inf")
    running_peak_date = None
    worst_dd = 0.0
    worst_dd_date = None
    for d in sorted(equity):
        v = equity[d]
        if v > running_peak_val:
            running_peak_val = v
            running_peak_date = d
        dd = (v - running_peak_val) / running_peak_val
        if dd < worst_dd:
            worst_dd = dd
            worst_dd_date = d
            peak = running_peak_date
    return equity, worst_dd * 100, peak, worst_dd_date


def independent_sharpe_sortino(equity_by_date: dict) -> tuple[float, float]:
    dates = sorted(equity_by_date)
    values = [equity_by_date[d] for d in dates]
    # First day's day-over-day return is undefined; metrics.py's own
    # convention (equity.pct_change().fillna(0.0)) treats it as 0.0 rather
    # than dropping it, so match that here for a fair comparison.
    rets = [0.0] + [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]
    if len(rets) < 2:
        return 0.0, 0.0
    mean = statistics.mean(rets)
    std = statistics.stdev(rets)
    sharpe = 0.0 if std == 0 else mean / std * (252 ** 0.5)
    downside = [r for r in rets if r < 0]
    if len(downside) < 2:
        sortino = 0.0
    else:
        dstd = statistics.stdev(downside)
        sortino = 0.0 if dstd == 0 else mean / dstd * (252 ** 0.5)
    return sharpe, sortino


def main() -> int:
    now = ts("2024-01-09T12:00:00")

    print("Regression guard: cap_premature_close_dates leaves status/is_open/exit_price alone")
    raw = build_dataset(now)
    t5_before = next(t for t in raw if t.ticker == "T5")
    check("T5 recorded with a future close_dt before capping",
          t5_before.close_dt > now, str(t5_before.close_dt))

    capped = cap_premature_close_dates(raw, now)
    t5_after = next(t for t in capped if t.ticker == "T5")
    check("capped close_dt equals `now`", t5_after.close_dt == now, str(t5_after.close_dt))
    check("status untouched by capping", t5_after.status == "closed", t5_after.status)
    check("is_open untouched by capping (the wrong-fix regression)", t5_after.is_open is False,
          "capping must not reclassify a closed trade as open")
    check("exit_price untouched by capping", t5_after.exit_price == t5_before.exit_price)
    other_ids = [id(a) for a in raw if a.ticker != "T5"]
    check("only T5's close_dt changed",
          all(a.close_dt == b.close_dt for a, b in zip(raw, capped) if a.ticker != "T5"))

    print("\nPipeline path: sizing.size_trades + metrics.daily_pnl/equity_curve/max_drawdown")
    sizing_result = size_trades(capped, ACCOUNT_SIZE, WEIGHT_PCT)
    by_ticker = {s.trade.ticker: s for s in sizing_result.sized_trades}

    expected_admit = {"T1": True, "T2": True, "T3": False, "T4": True, "T5": True}
    for tkr, exp in expected_admit.items():
        check(f"{tkr} admission == {exp}", by_ticker[tkr].admitted == exp,
              f"got {by_ticker[tkr].admitted}")
    check("pct_skipped == 20% (1 of 5 trades)", abs(sizing_result.pct_skipped - 20.0) < 1e-9,
          str(sizing_result.pct_skipped))

    pnl = M.daily_pnl(sizing_result.sized_trades, TZ)
    equity = M.equity_curve(pnl, ACCOUNT_SIZE)
    dd = M.max_drawdown(equity)
    cumulative_return_pct = 100 * (equity.iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE

    print("\nIndependent path: plain-Python reimplementation (no pandas)")
    admit_results = independent_size(capped, ACCOUNT_SIZE, WEIGHT_PCT)
    for tkr, exp in expected_admit.items():
        t = next(x for x in capped if x.ticker == tkr)
        got = admit_results[id(t)][0]
        check(f"independent: {tkr} admission == {exp}", got == exp, f"got {got}")

    ind_equity, ind_dd_pct, ind_peak, ind_trough = independent_equity_curve(capped, admit_results, ACCOUNT_SIZE)
    ind_final = ind_equity[max(ind_equity)]
    ind_cum_return = 100 * (ind_final - ACCOUNT_SIZE) / ACCOUNT_SIZE
    ind_sharpe, ind_sortino = independent_sharpe_sortino(ind_equity)

    print("\nCross-check: pipeline result vs. independent reimplementation")
    check("final equity matches", abs(float(equity.iloc[-1]) - ind_final) < 1e-6,
          f"pipeline={float(equity.iloc[-1])} independent={ind_final}")
    check("cumulative return % matches", abs(cumulative_return_pct - ind_cum_return) < 1e-6,
          f"pipeline={cumulative_return_pct} independent={ind_cum_return}")
    check("max drawdown % matches", abs(dd['max_drawdown_pct'] - ind_dd_pct) < 1e-6,
          f"pipeline={dd['max_drawdown_pct']} independent={ind_dd_pct}")
    check("peak date matches", str(dd['peak_date']) == str(ind_peak),
          f"pipeline={dd['peak_date']} independent={ind_peak}")

    strategy_daily_ret = equity.pct_change().fillna(0.0)
    sharpe = M.sharpe_ratio(strategy_daily_ret)
    sortino = M.sortino_ratio(strategy_daily_ret)
    check("Sharpe matches independent hand-rolled formula", abs(sharpe - ind_sharpe) < 1e-6,
          f"pipeline={sharpe} independent={ind_sharpe}")
    check("Sortino matches independent hand-rolled formula", abs(sortino - ind_sortino) < 1e-6,
          f"pipeline={sortino} independent={ind_sortino}")

    print("\nStructural invariants (must hold regardless of the specific data)")
    check("published cumulative return == (final - baseline) / baseline from the published equity series",
          abs(cumulative_return_pct - 100 * (equity.iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE) < 1e-9)
    check("T5's P&L was bucketed on the capped date, not deferred to 2030",
          equity.index.max().year < 2030, str(equity.index.max()))
    check("T3 (skipped) contributed zero to equity",
          by_ticker["T3"].pnl_dollars == 0.0 and by_ticker["T3"].notional == 0.0)

    # Hand-verified exact values for this specific dataset, computed
    # independently of both code paths above (by calculator, per the
    # module docstring).
    print("\nHand-calculated expected values for this exact dataset")
    check("final equity == 11450.0 (10000 + 500 - 200 + 0 + 400 + 750)",
          abs(float(equity.iloc[-1]) - 11450.0) < 1e-6, str(float(equity.iloc[-1])))
    check("cumulative return == 14.5%", abs(cumulative_return_pct - 14.5) < 1e-6,
          str(cumulative_return_pct))
    check("max drawdown == -1.9047619...% (10300 vs 10500 peak on 2024-01-06)",
          abs(dd['max_drawdown_pct'] - (-100 * 200 / 10500)) < 1e-6, str(dd['max_drawdown_pct']))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
