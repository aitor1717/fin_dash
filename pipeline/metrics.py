"""Standard portfolio/trade metrics plus the evaluation-style compliance panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from parse_feed import Trade, closed_trades
from sizing import SizedTrade, concurrency_series

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Daily $ P&L / equity curve, from admitted+closed trades grouped by
# close date.
# ---------------------------------------------------------------------------

def daily_pnl(sized_trades: list[SizedTrade], tz: str) -> pd.Series:
    rows = [
        (s.trade.close_dt.tz_convert(tz).normalize(), s.pnl_dollars)
        for s in sized_trades
        if s.admitted and not s.trade.is_open
    ]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["date", "pnl"])
    return df.groupby("date")["pnl"].sum().sort_index()


def equity_curve(pnl: pd.Series, account_size: float) -> pd.Series:
    if pnl.empty:
        return pd.Series([account_size], index=[pd.Timestamp.now()])
    full_index = pd.date_range(pnl.index.min(), pnl.index.max(), freq="D")
    daily = pnl.reindex(full_index, fill_value=0.0)
    return account_size + daily.cumsum()


def prior_close_equity(equity_naive: pd.Series, session_date, account_size: float) -> float:
    """Account equity as of the close of the trading day immediately before
    `session_date`, from the realized-only calendar-day equity curve
    (equity_curve() above -- *not* the live-mark-to-market-overlaid series
    build.py derives from it). This is the account_day_change_dollars
    denominator and the reference point for anything held overnight: a
    position opened before `session_date` and still open, or closed during
    it, gets marked from here, not from its own entry price.
    """
    if equity_naive is None or not len(equity_naive):
        return account_size
    prior = equity_naive[equity_naive.index < pd.Timestamp(session_date)]
    return float(prior.iloc[-1]) if len(prior) else account_size


def account_day_change_dollars(
    sized_trades: list[SizedTrade],
    session_index: pd.DatetimeIndex,
    intraday_prices: pd.DataFrame,
    prev_close_by_ticker: dict[str, float | None],
    session_date,
    tz: str,
) -> pd.Series:
    """Dollar move of the account since the previous session's close, at
    each intraday timestamp of `session_date` -- a broker's "day change",
    not the notional-weighted move-since-session-open this replaced.

    Per admitted position:
    - opened before `session_date` and closes during it (held overnight,
      then closed today): marked from prev_close_by_ticker[ticker] until
      its close time, then holds its realized dollar P&L flat.
    - opened before `session_date` and still open at the end of it (held
      overnight, never closes today): excluded. Its since-prev-close move
      would only be its own day's slice, but build.py's separate
      total_unrealized_dollars overlay (which is what the published equity
      total actually reflects for a still-open position) marks the whole
      book from entry price instead -- the position's entire lifetime
      move, however many days it's been open. Mixing those two bases would
      make prior_close_equity() + this series's last value silently
      disagree with the published equity total for any multi-day-held
      position. This is a sample/demo project, not a real account holding
      real long-lived swing positions, so rather than reconciling the two
      conventions, a long-held-and-still-open position is simply left out
      of the day-change series -- it only ever shows up in the overall
      equity curve, not in "today"'s own move.
    - opened during `session_date`: contributes 0 before its own open
      time, then marked from its own entry_price from there on -- not the
      previous close, which it never actually held a position at. Basis
      and prior_close_equity() agree here (it didn't exist before today),
      so no such ambiguity applies to a same-day open.
    - closed before `session_date`: excluded. Its P&L is already inside
      prior_close_equity() above, not part of this session's own change.

    Realized dollar P&L (SizedTrade.pnl_dollars) is exact by construction
    (from the feed's own entry/exit prices), independent of whatever price
    intraday_prices happens to show right at the close timestamp -- using
    it instead of a price-derived figure avoids baking bid/ask-spread
    noise into a number that should be exact.
    """
    out = pd.Series(0.0, index=session_index)
    if not len(session_index):
        return out
    session_start = session_index[0]

    for s in sized_trades:
        if not s.admitted:
            continue
        t = s.trade
        open_date = t.open_dt.tz_convert(tz).date()
        if open_date > session_date:
            continue  # opens after this session; not yet relevant

        opened_today = open_date == session_date

        close_date = t.close_dt.tz_convert(tz).date() if not t.is_open else None
        if close_date is not None and close_date < session_date:
            continue  # closed before this session -- already realized in prior_close_equity

        closed_today = close_date == session_date

        if opened_today:
            basis = t.entry_price
            active_from = t.open_dt
        else:
            if not closed_today:
                continue  # held-overnight-and-still-open -- see docstring
            basis = prev_close_by_ticker.get(t.ticker)
            active_from = session_start
            if basis is None:
                continue
        close_at = t.close_dt if closed_today else None
        sign = -1.0 if t.position == "short" else 1.0
        prices = intraday_prices[t.ticker] if t.ticker in intraday_prices.columns else None

        contrib = []
        for ts in session_index:
            if ts < active_from:
                contrib.append(0.0)
            elif close_at is not None and ts >= close_at:
                contrib.append(s.pnl_dollars)
            else:
                price = None
                if prices is not None:
                    available = prices.loc[:ts].dropna()
                    if len(available):
                        price = float(available.iloc[-1])
                if price is None:
                    contrib.append(0.0)
                else:
                    move_pct = sign * 100 * (price - basis) / basis
                    contrib.append(s.notional * move_pct / 100)
        out = out.add(pd.Series(contrib, index=session_index), fill_value=0.0)

    return out


def max_drawdown(equity: pd.Series) -> dict:
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    trough_idx = drawdown.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    recovery = equity.loc[trough_idx:]
    recovery_idx = recovery[recovery >= running_peak[trough_idx]].index
    return {
        "max_drawdown_pct": float(drawdown.min() * 100),
        "peak_date": str(peak_idx.date()),
        "trough_date": str(trough_idx.date()),
        "recovery_date": str(recovery_idx[0].date()) if len(recovery_idx) else None,
    }


def sharpe_ratio(returns: pd.Series) -> float:
    # len < 2, not just empty. pandas' .std() (ddof=1) returns NaN, not 0,
    # on a single-element series. NaN would pass the "== 0" check below,
    # then break the frontend's JSON.parse (NaN isn't valid JSON).
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(TRADING_DAYS_PER_YEAR)).dropna()


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict:
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1, join="inner",
    ).dropna()
    if len(aligned) < 2 or aligned["benchmark"].std() == 0:
        return {"alpha_annualized_pct": 0.0, "beta": 0.0, "r_squared": 0.0}
    beta, intercept = np.polyfit(aligned["benchmark"], aligned["strategy"], 1)
    predicted = beta * aligned["benchmark"] + intercept
    ss_res = ((aligned["strategy"] - predicted) ** 2).sum()
    ss_tot = ((aligned["strategy"] - aligned["strategy"].mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "alpha_annualized_pct": float(intercept * TRADING_DAYS_PER_YEAR * 100),
        "beta": float(beta),
        "r_squared": float(r_squared),
    }


# ---------------------------------------------------------------------------
# Trade-level stats, computed on *all* closed trades. Signal quality is a
# property of the trades, not of how much capital sizing admitted.
# ---------------------------------------------------------------------------

def trade_level_stats(trades: list[Trade]) -> dict:
    closed = closed_trades(trades)
    returns = pd.Series([t.pct_change for t in closed])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    hold_hours = pd.Series([
        (t.close_dt - t.open_dt).total_seconds() / 3600 for t in closed
    ])
    concurrency = pd.Series(concurrency_series(trades))

    gross_win = wins.sum()
    gross_loss = -losses.sum()

    return {
        "n_closed_trades": int(len(closed)),
        "win_rate_pct": float(100 * len(wins) / len(returns)) if len(returns) else 0.0,
        "mean_return_pct": float(returns.mean()) if len(returns) else 0.0,
        "median_return_pct": float(returns.median()) if len(returns) else 0.0,
        "std_return_pct": float(returns.std()) if len(returns) >= 2 else 0.0,
        "skew": float(returns.skew()) if len(returns) > 2 else 0.0,
        "kurtosis": float(returns.kurt()) if len(returns) > 3 else 0.0,
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        # None (JSON null), not float("inf"). Infinity isn't valid JSON and
        # breaks the frontend's JSON.parse. None matches how the rest of the
        # codebase represents "not applicable" (e.g. recovery_date).
        "profit_factor": float(gross_win / gross_loss) if gross_loss else None,
        # No separate "expectancy_pct" field: win_rate-weighted
        # avg_win/avg_loss collapses algebraically to the plain mean once
        # wins and losses partition every closed trade, so it would always
        # equal mean_return_pct above under a different name.
        "median_hold_hours": float(hold_hours.median()) if len(hold_hours) else 0.0,
        "mean_hold_hours": float(hold_hours.mean()) if len(hold_hours) else 0.0,
        "unique_tickers": int(len(set(t.ticker for t in trades))),
        "n_long": int(sum(1 for t in trades if t.position == "long")),
        "n_short": int(sum(1 for t in trades if t.position == "short")),
        "trades_per_day": float(len(trades) / max(
            1, (max(t.close_dt for t in trades) - min(t.open_dt for t in trades)).days
        )),
        "concurrency": {
            "mean": float(concurrency.mean()) if len(concurrency) else 0.0,
            "median": float(concurrency.median()) if len(concurrency) else 0.0,
            "p90": float(concurrency.quantile(0.90)) if len(concurrency) else 0.0,
            "p95": float(concurrency.quantile(0.95)) if len(concurrency) else 0.0,
            "p99": float(concurrency.quantile(0.99)) if len(concurrency) else 0.0,
            "max": float(concurrency.max()) if len(concurrency) else 0.0,
        },
    }


def ticker_concentration(sized_trades: list[SizedTrade], top_n: int = 5) -> list[dict]:
    rows = [
        (s.trade.ticker, s.pnl_dollars)
        for s in sized_trades if s.admitted and not s.trade.is_open
    ]
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["ticker", "pnl"])
    by_ticker = df.groupby("ticker")["pnl"].sum().sort_values(ascending=False)
    total = by_ticker.sum()
    top = by_ticker.head(top_n)
    return [
        {"ticker": t, "pnl_dollars": float(v), "share_of_total_pct": float(100 * v / total) if total else 0.0}
        for t, v in top.items()
    ]


# ---------------------------------------------------------------------------
# Evaluation-style compliance panel
# ---------------------------------------------------------------------------

def compliance_panel(
    cumulative_return_pct: float,
    max_drawdown_pct: float,
    sized_trades: list[SizedTrade],
    pct_skipped: float,
    market_snapshot: dict[str, dict],
    compliance_cfg: dict,
) -> dict:
    profit_target = compliance_cfg["profit_target_pct"]
    max_loss = compliance_cfg["max_loss_pct"]
    band_low, band_high = compliance_cfg["concentration_band_pct"]
    min_price = compliance_cfg["min_share_price"]
    min_volume = compliance_cfg["min_avg_volume"]

    concentration = ticker_concentration(sized_trades, top_n=1)
    top_concentration_pct = concentration[0]["share_of_total_pct"] if concentration else 0.0

    floor_flags = [
        {
            "ticker": t,
            "last_price": snap["last_price"],
            "avg_volume": snap["avg_volume"],
            "below_price_floor": snap["last_price"] is not None and snap["last_price"] < min_price,
            "below_volume_floor": snap["avg_volume"] is not None and snap["avg_volume"] < min_volume,
        }
        for t, snap in market_snapshot.items()
        if (snap["last_price"] is not None and snap["last_price"] < min_price)
        or (snap["avg_volume"] is not None and snap["avg_volume"] < min_volume)
    ]

    return {
        "profit_target_pct": profit_target,
        "progress_to_target_pct": float(100 * cumulative_return_pct / profit_target) if profit_target else 0.0,
        "max_loss_pct": max_loss,
        # max_drawdown_pct is peak-to-trough (already <= 0). Funded-account
        # max-loss rules use trailing drawdown, not the starting balance.
        "distance_to_max_loss_pct": float(max_loss + max_drawdown_pct),  # positive = safe margin
        "concentration_band_pct": [band_low, band_high],
        "top_position_concentration_pct": top_concentration_pct,
        # "30-50% max" is an upper bound, not a target band. Under band_low
        # is compliant. Over band_high is a violation. Between the two is a
        # caution zone.
        "concentration_status": (
            "ok" if top_concentration_pct <= band_low
            else "caution" if top_concentration_pct <= band_high
            else "violation"
        ),
        "pct_trades_skipped_capital": pct_skipped,
        "min_share_price": min_price,
        "min_avg_volume": min_volume,
        "floor_flags": floor_flags,
    }
