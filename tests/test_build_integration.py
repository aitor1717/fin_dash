"""End-to-end build.py run against a synthetic feed, with the yfinance-
backed market_data layer stubbed out (no network in tests). Covers the one
thing the unit tests above can't: that the assembled dashboard.json, as a
whole, is strict-JSON-safe (no NaN/Infinity anywhere in it) and carries the
new session_date/is_current_session/account-day-change fields.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import yaml

import build as build_module

FEED_HEADER = "open_datetime,close_datetime,ticker,open_bid,open_ask,close_bid,close_ask,change,position,status,notes\n"
FEED_ROWS = (
    "2024-02-01T10:00:00+00:00,2024-02-01T15:00:00+00:00,TICKA,,100,105,,5.0,long,closed,\n"
    "2024-02-01T09:00:00+00:00,2024-02-01T16:00:00+00:00,TICKC,50,51,45,46,10.0,short,closed,\n"
    "2024-02-01T11:00:00+00:00,2024-02-10T00:00:00+00:00,TICKB,,100,,,2.0,long,open,\n"
)


def fake_fetch_benchmark_history(tickers, start, end):
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame({t: 100.0 for t in tickers}, index=idx)


def fake_fetch_market_snapshot(tickers, asof_date=None):
    return {
        t: {
            "last_price": 100.0,
            "avg_volume": 1_000_000.0,
            "daily_closes": {"2024-02-01": 100.0, "2024-02-02": 100.0},
        }
        for t in tickers
    }


def fake_fetch_intraday_today(tickers, tz, asof_date=None):
    idx = pd.date_range("2024-02-02 09:30", periods=4, freq="1h", tz=tz)
    prices = pd.DataFrame({t: [100.0, 101.0, 102.0, 103.0] for t in tickers}, index=idx)
    return prices, dt.date(2024, 2, 2)


def test_build_produces_strictly_json_safe_dashboard(tmp_path, monkeypatch):
    feed_path = tmp_path / "feed.csv"
    feed_path.write_text(FEED_HEADER + FEED_ROWS)
    output_dir = tmp_path / "out"

    config = {
        "feed_path": str(feed_path),
        "timezone": "UTC",
        "account_size": 10_000.0,
        "position_weight_pct": 50.0,
        "benchmarks": ["QQQ"],
        "compliance": {
            "profit_target_pct": 10.0,
            "max_loss_pct": -5.0,
            "concentration_band_pct": [30.0, 50.0],
            "min_share_price": 5.0,
            "min_avg_volume": 100_000.0,
        },
        "output_dir": str(output_dir),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    monkeypatch.setattr(build_module, "fetch_benchmark_history", fake_fetch_benchmark_history)
    monkeypatch.setattr(build_module, "fetch_market_snapshot", fake_fetch_market_snapshot)
    monkeypatch.setattr(build_module, "fetch_intraday_today", fake_fetch_intraday_today)

    build_module.build(str(config_path))

    with open(output_dir / "dashboard.json") as f:
        dashboard = json.load(f)

    # Round-trips through strict JSON (allow_nan=False rejects a bare NaN/
    # Infinity token) -- the actual regression this guards against, not
    # just "the file parses."
    json.dumps(dashboard, allow_nan=False)

    assert dashboard["today"]["session_date"] == "2024-02-02"
    assert isinstance(dashboard["today"]["is_current_session"], bool)
    assert len(dashboard["today"]["series"]["portfolio"]) == 4
    assert all(v is None or isinstance(v, float) for v in dashboard["today"]["series"]["portfolio"])
