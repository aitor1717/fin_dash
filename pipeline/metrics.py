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
