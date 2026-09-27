from __future__ import annotations

import pandas as pd
import pytest

from parse_feed import parse_feed, cap_premature_close_dates
from helpers import ts, make_trade

FEED_HEADER = "open_datetime,close_datetime,ticker,open_bid,open_ask,close_bid,close_ask,change,position,status,notes\n"


def write_feed(tmp_path, rows: str):
    path = tmp_path / "feed.csv"
    path.write_text(FEED_HEADER + rows)
    return str(path)


def test_long_reconciles_on_ask_to_open_bid_to_close(tmp_path):
    # Old-era row: only the long-side columns populated.
    row = "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,AAA,,100,105,,5.0,long,closed,\n"
    trades = parse_feed(write_feed(tmp_path, row))
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 100.0
    assert t.exit_price == 105.0
    assert t.is_open is False
    assert t.position == "long"


def test_short_reconciles_on_bid_to_open_ask_to_close(tmp_path):
    # New-era row: both sides populated. A short profits from a price drop,
    # so it reconciles open_bid (entry) / close_ask (exit) -- the mirror of
    # a long's open_ask/close_bid.
    row = "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,BBB,50,51,45,46,10.0,short,closed,\n"
    trades = parse_feed(write_feed(tmp_path, row))
    t = trades[0]
    assert t.entry_price == 50.0  # open_bid
    assert t.exit_price == 46.0   # close_ask


def test_short_missing_its_own_side_of_the_spread_raises(tmp_path):
    # Old-era row has only the long-side columns -- a short can't reconcile
    # on those, and parse_feed must raise rather than silently substitute
    # the wrong side.
    row = "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,CCC,,100,105,,5.0,short,closed,\n"
    with pytest.raises(ValueError, match="no open_bid"):
        parse_feed(write_feed(tmp_path, row))


def test_unrecognized_status_raises(tmp_path):
    row = "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,DDD,,100,105,,5.0,long,pending,\n"
    with pytest.raises(ValueError, match="Unrecognized status"):
        parse_feed(write_feed(tmp_path, row))


def test_status_is_case_insensitive(tmp_path):
    row = "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,EEE,,100,105,,5.0,long,CLOSED,\n"
    trades = parse_feed(write_feed(tmp_path, row))
    assert trades[0].is_open is False


def test_open_trade_needs_no_exit_price(tmp_path):
    row = "2024-01-01T10:00:00-05:00,2030-01-01T00:00:00-05:00,FFF,,100,,,2.0,long,open,\n"
    trades = parse_feed(write_feed(tmp_path, row))
    t = trades[0]
    assert t.is_open is True
    assert t.exit_price is None


def test_timestamps_normalized_to_utc_across_dst_offsets(tmp_path):
    # The feed mixes fixed -04:00/-05:00 offsets across DST transitions --
    # parse_feed must land both on a common UTC axis, not error building a
    # mixed-offset DatetimeIndex.
    rows = (
        "2024-01-01T10:00:00-05:00,2024-01-01T15:00:00-05:00,GGG,,100,105,,5.0,long,closed,\n"
        "2024-07-01T10:00:00-04:00,2024-07-01T15:00:00-04:00,HHH,,100,105,,5.0,long,closed,\n"
    )
    trades = parse_feed(write_feed(tmp_path, rows))
    assert all(t.open_dt.tzinfo is not None for t in trades)
    assert str(trades[0].open_dt.tz) == "UTC"


def test_cap_premature_close_dates_corrects_only_the_timestamp():
    now = ts("2024-01-09T12:00:00")
    closed_future = make_trade("2024-01-01", "2030-01-01", "T1", pct_change=15.0)
    open_future = make_trade("2024-01-01", "2030-01-01", "T2", pct_change=3.0, is_open=True)

    capped = cap_premature_close_dates([closed_future, open_future], now)

    c1, c2 = capped
    # A closed trade's close_dt is a real, already-happened event -- capped
    # at `now`, and nothing else about the record changes.
    assert c1.close_dt == now
    assert c1.status == "closed"
    assert c1.is_open is False
    assert c1.exit_price == closed_future.exit_price

    # An open trade's close_dt may legitimately be a scheduled/target date
    # far in the future -- left untouched.
    assert c2.close_dt == ts("2030-01-01")
    assert c2.is_open is True


def test_cap_premature_close_dates_leaves_past_dates_alone():
    now = ts("2024-01-09T12:00:00")
    trade = make_trade("2024-01-01", "2024-01-05", "T1", pct_change=5.0)
    capped = cap_premature_close_dates([trade], now)
    assert capped[0].close_dt == trade.close_dt
