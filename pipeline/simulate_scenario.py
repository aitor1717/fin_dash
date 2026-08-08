"""What-if scenario generator: layer synthetic long/short trades onto the
real feed to see the effect on portfolio stats IF that much additional,
genuinely uncorrelated edge existed. This does not forecast or validate that
such edge exists -- it shows the consequence of assuming it does, and it
verifies the "uncorrelated" assumption is actually true in the simulation
(reports the realized correlation) rather than just asserting it.

Synthetic trades are bootstrap-resampled from the real closed-trade return
and hold-time distributions, then placed at independent random times across
the same window -- decorrelating them in time from the real day-by-day P&L.
They're written out as ordinary rows in the same feed CSV schema, so the
existing pipeline (parse_feed/sizing/metrics/build) runs on them unchanged.

Usage:
  python pipeline/simulate_scenario.py --config config.yaml \
      --long-multiplier 2.0 --short-pct-of-new-longs 20 \
      --out-feed feed/hypothetical_augmented.csv --seed 7
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np
import pandas as pd
import yaml

from parse_feed import parse_feed, closed_trades
from sizing import size_trades
import metrics as M


def correlation_check(real_trades, synthetic_trades, account_size, weight_pct, tz):
    real_sized = size_trades(real_trades, account_size, weight_pct).sized_trades
    synth_sized = size_trades(synthetic_trades, account_size, weight_pct).sized_trades
    real_pnl = M.daily_pnl(real_sized, tz)
    synth_pnl = M.daily_pnl(synth_sized, tz)
    if real_pnl.empty or synth_pnl.empty:
        return 0.0
    full_index = pd.date_range(
        min(real_pnl.index.min(), synth_pnl.index.min()),
        max(real_pnl.index.max(), synth_pnl.index.max()),
        freq="D",
    )
    a = real_pnl.reindex(full_index, fill_value=0.0)
    b = synth_pnl.reindex(full_index, fill_value=0.0)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def make_synthetic_rows(n: int, position: str, prefix: str, returns_pool, hold_pool,
                         window_start: pd.Timestamp, window_end: pd.Timestamp, rng: random.Random):
    rows = []
    span_hours = (window_end - window_start).total_seconds() / 3600
    for i in range(n):
        pct = rng.choice(returns_pool)
        hold_h = rng.choice(hold_pool)
        open_offset_h = rng.uniform(0, max(1.0, span_hours - hold_h))
        open_dt = window_start + pd.Timedelta(hours=open_offset_h)
        close_dt = open_dt + pd.Timedelta(hours=hold_h)
        entry_price = 100.0
        exit_price = entry_price * (1 + pct / 100)
        rows.append({
            "open_datetime": open_dt.isoformat(),
            "close_datetime": close_dt.isoformat(),
            "ticker": f"{prefix}{i+1:04d}",
            "open_bid": "",
            "open_ask": f"{entry_price:.4f}",
            "close_bid": f"{exit_price:.4f}",
            "close_ask": "",
            "change": f"{pct:.4f}",
            "position": position,
            # Always closed by construction (make_synthetic_rows never
            # produces an open trade) -- "closed", not a third status value,
            # now that parse_feed.py's is_open check is status-driven. The
            # SIM- ticker prefix and this notes field already flag these as
            # synthetic; status doesn't need to carry that too.
            "status": "closed",
            "notes": "synthetic what-if trade, not a real signal",
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--long-multiplier", type=float, default=2.0,
                         help="total long count becomes this multiple of the real long count")
    parser.add_argument("--short-pct-of-new-longs", type=float, default=20.0,
                         help="synthetic short count = this %% of the NEW total long count (real + synthetic)")
    parser.add_argument("--out-feed", default="feed/hypothetical_augmented.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tz = cfg["timezone"]
    account_size = float(cfg["account_size"])
    weight_pct = float(cfg["position_weight_pct"])

    real_trades = parse_feed(cfg["feed_path"])
    real_closed = closed_trades(real_trades)
    returns_pool = [t.pct_change for t in real_closed]
    hold_pool = [(t.close_dt - t.open_dt).total_seconds() / 3600 for t in real_closed]
    window_start = min(t.open_dt for t in real_trades)
    window_end = max(t.close_dt for t in real_trades)

    n_real_long = len(real_trades)
    n_new_long = round(n_real_long * (args.long_multiplier - 1))
    n_new_total_long = n_real_long + n_new_long
    n_new_short = round(n_new_total_long * args.short_pct_of_new_longs / 100)

    rng = random.Random(args.seed)
    synthetic_long_rows = make_synthetic_rows(
        n_new_long, "long", "SIM-L", returns_pool, hold_pool, window_start, window_end, rng)
    synthetic_short_rows = make_synthetic_rows(
        n_new_short, "short", "SIM-S", returns_pool, hold_pool, window_start, window_end, rng)

    print(f"Real long trades: {n_real_long}")
    print(f"Synthetic new longs added: {n_new_long} (total longs: {n_new_total_long})")
    print(f"Synthetic new shorts added: {n_new_short} ({args.short_pct_of_new_longs}% of new total long count)")

    # Verify the "uncorrelated" assumption rather than just asserting it.
    def row_to_trade(row):
        from parse_feed import Trade
        return Trade(
            open_dt=pd.Timestamp(row["open_datetime"]).tz_convert("UTC"),
            close_dt=pd.Timestamp(row["close_datetime"]).tz_convert("UTC"),
            ticker=row["ticker"], entry_price=float(row["open_ask"]), exit_price=float(row["close_bid"]),
            pct_change=float(row["change"]), is_open=False, status=row["status"], position=row["position"],
        )

    synthetic_trades = [row_to_trade(r) for r in synthetic_long_rows + synthetic_short_rows]
    corr = correlation_check(real_trades, synthetic_trades, account_size, weight_pct, tz)
    print(f"Realized correlation between real and synthetic daily P&L series: {corr:.4f} (target ~0)")

    with open(cfg["feed_path"]) as f:
        real_fieldnames = next(csv.reader(f))
    out_rows = []
    with open(cfg["feed_path"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            out_rows.append({k: row.get(k, "") for k in real_fieldnames})
    for row in synthetic_long_rows + synthetic_short_rows:
        out_rows.append({k: row.get(k, "") for k in real_fieldnames})

    with open(args.out_feed, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=real_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out_feed} ({len(out_rows)} total rows)")


if __name__ == "__main__":
    main()
