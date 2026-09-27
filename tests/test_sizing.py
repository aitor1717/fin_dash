from __future__ import annotations

from sizing import size_trades
from helpers import make_trade

ACCOUNT_SIZE = 10_000.0
WEIGHT_PCT = 50.0  # notional_per_trade = 5000 -- exactly 2 concurrent trades fill the cap


def test_two_concurrent_trades_exactly_fill_the_cap():
    trades = [
        make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0),
        make_trade("2024-01-02", "2024-01-06", "T2", pct_change=-4.0),
    ]
    result = size_trades(trades, ACCOUNT_SIZE, WEIGHT_PCT)
    by_ticker = {s.trade.ticker: s for s in result.sized_trades}
    assert by_ticker["T1"].admitted is True
    assert by_ticker["T2"].admitted is True
    assert result.pct_skipped == 0.0


def test_a_trade_that_would_exceed_the_cap_is_skipped_not_partially_sized():
    trades = [
        make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0),
        make_trade("2024-01-02", "2024-01-06", "T2", pct_change=-4.0),
        # Opens while T1+T2 are both still open (10000 already committed) --
        # admitting even part of it would exceed the account's buying power.
        make_trade("2024-01-03", "2024-01-04", "T3", pct_change=50.0),
    ]
    result = size_trades(trades, ACCOUNT_SIZE, WEIGHT_PCT)
    by_ticker = {s.trade.ticker: s for s in result.sized_trades}
    t3 = by_ticker["T3"]
    assert t3.admitted is False
    # Zero-filled, not a partial fill at some smaller notional.
    assert t3.notional == 0.0
    assert t3.pnl_dollars == 0.0
    assert abs(result.pct_skipped - (100.0 / 3)) < 1e-9


def test_a_trade_is_admitted_once_earlier_positions_have_closed():
    trades = [
        make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0),
        make_trade("2024-01-02", "2024-01-06", "T2", pct_change=-4.0),
        # Opens after T1+T2 have both closed -- exposure is back to 0.
        make_trade("2024-01-07", "2024-01-08", "T4", pct_change=8.0, position="short"),
    ]
    result = size_trades(trades, ACCOUNT_SIZE, WEIGHT_PCT)
    by_ticker = {s.trade.ticker: s for s in result.sized_trades}
    assert by_ticker["T4"].admitted is True
    assert by_ticker["T4"].notional == 5000.0


def test_open_trades_are_admitted_with_zero_realized_pnl():
    trades = [make_trade("2024-01-01", "2030-01-01", "T1", pct_change=3.0, is_open=True)]
    result = size_trades(trades, ACCOUNT_SIZE, WEIGHT_PCT)
    s = result.sized_trades[0]
    assert s.admitted is True
    assert s.pnl_dollars == 0.0  # unrealized while open -- realized-only accounting
