"""Generates a fully synthetic feed CSV from scratch -- no real trade history
involved -- so a public/portfolio copy of this repo can demonstrate the full
pipeline without exposing any real trading data. Uses real, liquid stock
tickers priced at their ACTUAL historical close on each trade's own
randomly-drawn date (see fetch_historical_prices/price_on) -- so entry
prices are genuinely accurate for when they claim to have happened, not a
guess. What's invented is the trading activity itself: open time, hold
duration, side (long/short), and return are all drawn from a
plausible-but-synthetic distribution, not resampled from or otherwise
derived from any actual account.

A trade is "still open" purely as a consequence of its own randomly-drawn
open time + hold duration landing after --end (i.e. it would still be open
if "today" were --end) -- not chosen separately, so which trades end up open
falls out naturally the same way it would from a real account.

Usage:
  python pipeline/generate_sample_feed.py --n-trades 850 \
      --start 2025-01-01 --end 2026-08-01 \
      --out-feed feed/sample_feed.csv --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random

import pandas as pd
import yfinance as yf

# Real, liquid, large/mid-cap tickers -- plausible for a swing-trading
# strategy, and real enough that yfinance resolves benchmark/compliance data
# for them normally (unlike simulate_scenario.py's SIM-prefixed synthetic
# tickers, which build.py explicitly skips before any yfinance call since
# they aren't real symbols).
TICKER_POOL = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX",
    "CRM", "ADBE", "ORCL", "AVGO", "QCOM", "INTC", "CSCO", "IBM", "TXN",
    "NOW", "INTU", "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "PANW",
    "CRWD", "SNOW", "PLTR", "SHOP", "UBER", "ABNB", "COIN", "SQ", "PYPL",
    "V", "MA", "JPM", "GS", "BAC", "DIS", "NKE", "SBUX", "COST", "WMT",
    "HD", "LOW", "UNH", "JNJ", "PFE", "XOM", "CVX", "CAT", "BA", "GE",
]

# Rough fallback ballpark prices, only used per-ticker if the historical
# fetch below has no data for that symbol on that date -- everything else
# uses ACTUAL historical closes (see fetch_historical_prices/price_on),
# since open_positions' live_price/unrealized_pct compares a trade's own
# open_ask against the ticker's real current yfinance price. A stale or
# ticker-agnostic guess reads as a wildly wrong "unrealized gain" the moment
# it's compared against the real price.
TICKER_PRICE_FALLBACK = {
    "AAPL": 230, "MSFT": 520, "GOOGL": 195, "AMZN": 230, "NVDA": 180,
    "META": 715, "TSLA": 340, "AMD": 165, "NFLX": 1200, "CRM": 330,
    "ADBE": 480, "ORCL": 230, "AVGO": 280, "QCOM": 170, "INTC": 35,
    "CSCO": 65, "IBM": 250, "TXN": 200, "NOW": 1050, "INTU": 700,
    "AMAT": 220, "MU": 130, "LRCX": 90, "KLAC": 1000, "SNPS": 550,
    "CDNS": 340, "PANW": 200, "CRWD": 470, "SNOW": 180, "PLTR": 150,
    "SHOP": 110, "UBER": 90, "ABNB": 140, "COIN": 330, "SQ": 75,
    "PYPL": 75, "V": 340, "MA": 570, "JPM": 290, "GS": 700,
    "BAC": 48, "DIS": 115, "NKE": 75, "SBUX": 95, "COST": 950,
    "WMT": 95, "HD": 410, "LOW": 250, "UNH": 330, "JNJ": 155,
    "PFE": 25, "XOM": 115, "CVX": 155, "CAT": 380, "BA": 200, "GE": 220,
}


def fetch_spy_weekly_returns(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Real SPY weekly % returns, keyed the same way as week_factors
    (isocalendar (year, week)) -- gives the synthetic book a genuine,
    non-zero correlation with the real market (a beta the pipeline's own
    regression can actually pick up), instead of every week's drift being
    independent noise with no relationship to what the market actually did.
    Empty dict on failure -- callers fall back to pure noise for the drift."""
    try:
        data = yf.download("SPY", start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                            interval="1d", progress=False, auto_adjust=True, threads=False)
        # Even for a single ticker, this yfinance version returns a
        # MultiIndex-columned DataFrame (Price, Ticker) -- data["Close"] is
        # a one-column DataFrame, not a Series. .squeeze() flattens it;
        # without that, DataFrame.items() iterates COLUMNS, not rows, and
        # every "date" below would actually be the string "SPY".
        closes = data["Close"].dropna().squeeze()
        if len(closes) < 2:
            return {}
        weekly = closes.resample("W").last().dropna()
        weekly_ret = weekly.pct_change().dropna() * 100
        return {pd.Timestamp(ts).isocalendar()[:2]: float(ret) for ts, ret in weekly_ret.items()}
    except Exception as e:
        print(f"Live SPY fetch failed ({e}) -- weekly drift will be uncorrelated noise")
        return {}


def fetch_historical_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.Series]:
    """Real daily close prices per ticker across the WHOLE generation window
    -- not just today's price -- so each trade can be priced at what that
    ticker actually traded at on its own randomly-drawn date, not today's
    price with a random fudge factor standing in for "sometime in the past".
    Batched, not per-ticker (see market_data.py's own reasoning for why)."""
    try:
        data = yf.download(tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=2)).date()),
                            interval="1d", progress=False, auto_adjust=True, group_by="ticker", threads=False)
        out = {}
        for t in tickers:
            try:
                closes = data[t]["Close"].dropna()
                if len(closes):
                    out[t] = closes
            except Exception:
                continue  # falls back to TICKER_PRICE_FALLBACK for this one ticker
        return out
    except Exception as e:
        print(f"Historical price fetch failed ({e}) -- using fallback prices for all tickers/dates")
        return {}


def price_on(historical: dict, ticker: str, dt: pd.Timestamp) -> float | None:
    """The real close nearest to (at or before) `dt` for this ticker --
    trading days don't align with the trade's own random weekday timestamp,
    so this is a lookup, not an exact-date match. None if this ticker has no
    historical series at all (caller falls back to TICKER_PRICE_FALLBACK)."""
    series = historical.get(ticker)
    if series is None or not len(series):
        return None
    idx = series.index
    naive = pd.Timestamp(dt.date())
    pos = idx.searchsorted(naive, side="right") - 1
    pos = max(0, min(pos, len(idx) - 1))
    return float(series.iloc[pos])

FIELDNAMES = [
    "open_datetime", "close_datetime", "ticker", "open_bid", "open_ask",
    "close_bid", "close_ask", "change", "position", "status",
]


def soft_cap(pct: float, lo: float, hi: float, rng: random.Random, slip: float) -> float:
    """Caps a trade's return like a stop-loss/profit-target would -- but a
    real stop doesn't fill at exactly the same price every time (slippage),
    so a hard min/max clamp here would pile every capped trade onto the
    exact same value, showing up as an unrealistic single-bar spike in the
    return histogram. Trades beyond the threshold instead land a small
    random amount past it, spreading that pileup into a short, believable
    tail segment instead of one delta spike."""
    if pct < lo:
        return lo - rng.uniform(0, slip)
    if pct > hi:
        return hi + rng.uniform(0, slip)
    return pct


def random_weekday_business_time(rng: random.Random, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    for _ in range(50):  # a handful of retries is plenty; span is many days
        offset_days = rng.uniform(0, max(0.0, (end - start).total_seconds() / 86400))
        dt = (start + pd.Timedelta(days=offset_days)).normalize()
        if dt.dayofweek < 5:  # Mon-Fri
            hour = rng.randint(9, 15)
            minute = rng.randint(0, 59)
            second = rng.randint(0, 59)
            return dt + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
    return start  # pathological fallback, should never hit given the retry budget


def make_trade_row(rng: random.Random, lo: pd.Timestamp, hi: pd.Timestamp, end: pd.Timestamp,
                    win_rate: float, week_factors: dict, historical_prices: dict,
                    spy_weekly: dict, beta_target: float, short_frac: float) -> dict:
    ticker = rng.choice(TICKER_POOL)
    open_dt = random_weekday_business_time(rng, lo, hi)
    # Log-normal hold time: mostly a couple of days, with a fat enough right
    # tail (occasional week-plus holds) to give realistic concurrency --
    # mean hold time (not just the median) is what drives how many positions
    # are open at once for a given trade frequency (Little's Law), and
    # lognormal's mean > median by construction.
    hold_hours = max(0.5, rng.lognormvariate(3.4, 1.3))
    close_dt = open_dt + pd.Timedelta(hours=hold_hours)
    is_open = close_dt > end

    # The ticker's REAL close nearest its own open_dt -- not today's price
    # with a random fudge factor standing in for "sometime in the past".
    real_price = price_on(historical_prices, ticker, open_dt)
    entry_price = round(real_price if real_price is not None else TICKER_PRICE_FALLBACK[ticker], 2)
    open_bid = round(entry_price * (1 - rng.uniform(0.0005, 0.002)), 2)

    is_short = rng.random() < short_frac

    # A shared per-calendar-week MARKET component (beta_target * that week's
    # REAL SPY return), on top of each trade's own idiosyncratic return --
    # without some shared pull, ~15 concurrently-open positions with fully
    # independent returns diversify into an implausibly smooth equity curve
    # (a giveaway that it's synthetic, and a Sharpe no real strategy actually
    # sustains). This is what makes the book genuinely correlated with the
    # market (a beta the pipeline's own regression actually measures as
    # nonzero) instead of the ~0 beta a fully independent drift produces.
    #
    # Only the true market term is cached/shared per week; the noise below
    # is drawn fresh per trade (not cached), so trades sharing a week still
    # feel the same market pull without moving in lockstep as one block.
    week_key = close_dt.isocalendar()[:2]
    if week_key not in week_factors:
        week_factors[week_key] = beta_target * spy_weekly.get(week_key, 0.0)
    market_drift = week_factors[week_key] + rng.gauss(0, 16.0)

    if is_open:
        # "Last-marked" unrealized move -- tighter/closer-to-zero than a
        # closed trade's final realized return, same idea as a live position
        # that hasn't necessarily hit its target or stop yet. Clipped like
        # the closed case below, just to a tighter band (a live position
        # hasn't run as far as one that's already been let ride to a stop or
        # target).
        pct = rng.uniform(-4.0, 4.0) + market_drift
        if is_short:
            pct = -pct  # same underlying move, opposite side's payoff
        pct = soft_cap(pct, -18.0, 18.0, rng, 2.0)
        close_bid = "pending"
        close_ask = "pending"
    else:
        win = rng.random() * 100 < win_rate
        # Right-skewed win/loss magnitude distributions, medians close
        # together -- the win-rate edge (not a big win/loss size asymmetry)
        # is what should drive overall performance, so this doesn't
        # compound into an unrealistically high Sharpe.
        pct = (rng.lognormvariate(1.1, 0.8) if win else -rng.lognormvariate(1.1, 0.75)) + market_drift
        # Computed as if long, then flipped -- a short profits from the same
        # move a long loses on, market-wide component included (a short is
        # also short the market, not just the idiosyncratic move).
        if is_short:
            pct = -pct
        # A real trade doesn't ride an ordinary swing entry to a 50%+ swing
        # in a couple of days without a stop or target executing first --
        # the market-drift noise above exists to de-correlate trades within
        # a week, not to manufacture single-trade blowups. Clipped a little
        # asymmetrically (winners allowed to run slightly further than
        # losses), same idea as a stop-loss/profit-target pair.
        pct = soft_cap(pct, -25.0, 35.0, rng, 4.0)
        # `pct` (recorded as "change") is the TRADE's signed return -- for a
        # short that's the inverse of the underlying price's own move. Entry
        # is a real historical close; exit_price has to reflect where the
        # price itself actually went, or a profitable short would show
        # close_bid ABOVE open_ask (bought back higher than sold), which
        # reads as a loss to anyone inspecting the raw price columns even
        # though `change` correctly says it was a win.
        price_move_pct = -pct if is_short else pct
        exit_price = round(entry_price * (1 + price_move_pct / 100), 2)
        close_bid = f"{exit_price:.2f}"
        close_ask = f"{round(exit_price * (1 + rng.uniform(0.0005, 0.002)), 2):.2f}"

    return {
        "open_datetime": str(open_dt),
        "close_datetime": str(close_dt),
        "ticker": ticker,
        "open_bid": f"{open_bid:.2f}",
        "open_ask": f"{entry_price:.2f}",
        "close_bid": close_bid,
        "close_ask": close_ask,
        "change": f"{pct:.4f}",
        "position": "short" if is_short else "long",
        # The authoritative open/closed signal parse_feed.py reads -- kept
        # to exactly these two values, matching the same is_open this row's
        # close_bid/close_ask ("pending" vs. a real price) was set from above.
        "status": "open" if is_open else "closed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trades", type=int, default=850)
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-08-03")
    parser.add_argument("--win-rate", type=float, default=58.0, help="%% of closed trades that are winners")
    parser.add_argument("--short-frac", type=float, default=0.1,
                         help="fraction of trades taken short instead of long -- same underlying move, inverted payoff")
    parser.add_argument("--beta-target", type=float, default=1.5,
                         help="how strongly weekly drift tracks real SPY weekly returns (roughly the resulting beta)")
    parser.add_argument("--recent-days", type=float, default=13.0,
                         help="width of one ramp zone -- trading pace climbs across 4 of these zones before --end, for a realistic still-open count")
    parser.add_argument("--out-feed", default="feed/sample_feed.csv")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tz = "America/New_York"
    start = pd.Timestamp(args.start, tz=tz)
    end = pd.Timestamp(args.end, tz=tz)

    # A uniform spread across the whole range leaves "still open" almost
    # entirely up to chance -- with a multi-hundred-day range and short
    # median hold times, only trades opened within their own hold time of
    # `end` can land open, a razor-thin sliver. Real accounts don't work
    # that way: trading activity is roughly steady right up to "today", so a
    # meaningful chunk of the MOST RECENT trades are naturally still open.
    # Concentrating trades into a trailing window (rather than picking "is
    # this one open" directly) reproduces that same falling-out-naturally-
    # from-timing effect the real feed has, instead of faking it.
    #
    # A single sharp cutoff -- or even one intermediate "ramp" tier -- still
    # reads as a visible step in capital utilization (which only samples at
    # each new admission, see sizing.py): the tier right before the cutoff
    # needs to end at nearly the same density the cutoff tier starts at, or
    # there's still one big jump right at the boundary. Four zones with
    # trading pace roughly geometrically increasing (not just two) spreads
    # that climb out smoothly instead of concentrating it at one edge.
    n_ramp_zones = 4
    zone_span_days = args.recent_days * 2
    zone_density_weights = [1.5, 2.2, 3.2, 4.6]  # relative to the historical baseline (1.0)
    ramp_span_days = zone_span_days * n_ramp_zones
    ramp_start = end - pd.Timedelta(days=ramp_span_days)
    hist_span_days = max(0.0, (ramp_start - start).total_seconds() / 86400)

    total_weighted_days = hist_span_days + sum(w * zone_span_days for w in zone_density_weights)
    baseline_density = args.n_trades / total_weighted_days  # trades/day at density=1.0

    open_windows = []
    if hist_span_days > 0:
        open_windows.append((round(baseline_density * hist_span_days), start, ramp_start))
    for i, w in enumerate(zone_density_weights):
        zone_lo = ramp_start + pd.Timedelta(days=i * zone_span_days)
        zone_hi = ramp_start + pd.Timedelta(days=(i + 1) * zone_span_days)
        open_windows.append((round(baseline_density * zone_span_days * w), zone_lo, zone_hi))
    # Rounding drift goes on the last (most-recent) zone -- it's the one that
    # determines "still open" count, so it's worth keeping exact.
    drift = args.n_trades - sum(c for c, _, _ in open_windows)
    last_n, last_lo, last_hi = open_windows[-1]
    open_windows[-1] = (last_n + drift, last_lo, last_hi)

    print(f"Fetching historical prices for {len(TICKER_POOL)} tickers ({start.date()} to {end.date()})...")
    historical_prices = fetch_historical_prices(TICKER_POOL, start, end)
    print("Fetching SPY weekly returns for market correlation...")
    spy_weekly = fetch_spy_weekly_returns(start, end)

    week_factors: dict = {}
    rows = [make_trade_row(rng, lo, hi, end, args.win_rate, week_factors, historical_prices,
                            spy_weekly, args.beta_target, args.short_frac)
            for count, lo, hi in open_windows for _ in range(count)]
    n_open = sum(1 for r in rows if r["close_bid"] == "pending")

    rows.sort(key=lambda r: r["open_datetime"])
    with open(args.out_feed, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out_feed} ({len(rows)} synthetic trades, {n_open} still open, seed={args.seed})")


if __name__ == "__main__":
    main()
