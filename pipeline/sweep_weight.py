"""Sweep position_weight_pct to find the largest per-trade weight that still
clears the max-loss threshold by a safety margin.

Usage: python pipeline/sweep_weight.py [--config config.yaml] [--margin 1.5]
       [--weights 0.5,1,1.5,2,2.5,3,3.5,4]
"""
from __future__ import annotations

import argparse

import yaml

from parse_feed import parse_feed
from sizing import size_trades
import metrics as M


def evaluate(trades, account_size: float, weight_pct: float, tz: str) -> dict:
    sizing_result = size_trades(trades, account_size, weight_pct)
    pnl = M.daily_pnl(sizing_result.sized_trades, tz)
    equity = M.equity_curve(pnl, account_size)
    cumulative_return_pct = 100 * (equity.iloc[-1] - account_size) / account_size if len(equity) else 0.0
    dd = M.max_drawdown(equity)
    return {
        "weight_pct": weight_pct,
        "total_return_pct": round(cumulative_return_pct, 2),
        "max_drawdown_pct": round(dd["max_drawdown_pct"], 2),
        "pct_skipped": round(sizing_result.pct_skipped, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--margin", type=float, default=1.5, help="required safety margin (points) clear of max_loss_pct")
    parser.add_argument("--max-loss", type=float, default=None, help="override compliance.max_loss_pct from the config (e.g. to test a looser tolerance)")
    parser.add_argument("--weights", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tz = cfg["timezone"]
    account_size = float(cfg["account_size"])
    max_loss_pct = args.max_loss if args.max_loss is not None else cfg["compliance"]["max_loss_pct"]
    weights = [float(w) for w in args.weights.split(",")]

    trades = parse_feed(cfg["feed_path"])

    rows = [evaluate(trades, account_size, w, tz) for w in weights]

    print(f"{'w%':>5} {'return%':>9} {'max_dd%':>9} {'margin_to_maxloss':>18} {'%skipped':>9}")
    safe_max = None
    for r in rows:
        margin = max_loss_pct + r["max_drawdown_pct"]  # max_drawdown_pct is <= 0
        ok = margin >= args.margin
        if ok:
            safe_max = r["weight_pct"]
        flag = "OK" if ok else "BREACH"
        print(f"{r['weight_pct']:>5} {r['total_return_pct']:>9} {r['max_drawdown_pct']:>9} {margin:>17.2f} {r['pct_skipped']:>8}%  {flag}")

    print(f"\nLargest weight keeping max drawdown >= {args.margin} pts clear of -{max_loss_pct}%: {safe_max}%")


if __name__ == "__main__":
    main()
