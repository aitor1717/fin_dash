from __future__ import annotations

import datetime as dt

import pandas as pd

import metrics as M
from helpers import make_trade, make_sized

TZ = "UTC"
ACCOUNT_SIZE = 10_000.0
SESSION_DATE = dt.date(2024, 2, 2)


def session_index(*hours: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(f"2024-02-02T{h:02d}:00:00", tz="UTC") for h in hours])


def test_prior_close_equity_is_the_last_value_before_session_date():
    equity_naive = pd.Series([10300.0], index=[pd.Timestamp("2024-02-01")])
    assert M.prior_close_equity(equity_naive, SESSION_DATE, ACCOUNT_SIZE) == 10300.0


def test_prior_close_equity_falls_back_to_account_size_with_no_history():
    assert M.prior_close_equity(pd.Series(dtype=float), SESSION_DATE, ACCOUNT_SIZE) == ACCOUNT_SIZE


def test_held_overnight_and_still_open_position_is_dropped():
    # Opened well before this session and never closes during it -- see the
    # function's own docstring. build.py's separate total_unrealized_dollars
    # overlay marks a still-open position from its entry price (its whole
    # lifetime move), not the previous close, so counting it here too
    # (on a different basis) would make prior_close_equity() + this
    # series's last value silently disagree with the published equity
    # total. This is a sample project's day-change chart, not a real
    # account's P&L ledger, so a long-held, still-open position is simply
    # left out rather than reconciling the two conventions.
    trade = make_trade("2024-01-15", "2030-01-01", "TICKA", pct_change=0.0, is_open=True, entry_price=40.0)
    sized = [make_sized(trade, notional=5000.0)]
    idx = session_index(9, 12)
    prices = pd.DataFrame({"TICKA": [102.0, 104.0]}, index=idx)

    out = M.account_day_change_dollars(sized, idx, prices, {"TICKA": 100.0}, SESSION_DATE, TZ)
    assert (out == 0.0).all()


def test_position_opened_today_contributes_nothing_before_its_own_open_time():
    trade = make_trade("2024-02-02", "2030-01-01", "TICKB", pct_change=0.0, is_open=True, entry_price=100.0)
    trade.open_dt = pd.Timestamp("2024-02-02T15:00:00", tz="UTC")
    sized = [make_sized(trade, notional=5000.0)]
    idx = session_index(9, 15, 16)
    prices = pd.DataFrame({"TICKB": [100.0, 102.0, 104.0]}, index=idx)

    out = M.account_day_change_dollars(sized, idx, prices, {}, SESSION_DATE, TZ)
    assert out.iloc[0] == 0.0  # before its own open time
    assert out.iloc[1] == 5000.0 * (102.0 - 100.0) / 100.0
    assert out.iloc[2] == 5000.0 * (104.0 - 100.0) / 100.0


def test_short_position_gains_on_a_price_drop():
    # Held overnight, closes during the session -- unlike a held-overnight-
    # and-still-open position, this one isn't dropped: it does have a
    # single, unambiguous realized outcome to snap to at close.
    trade = make_trade("2024-01-15", "2024-02-02", "TICKS", pct_change=5.0, position="short", entry_price=100.0)
    trade.close_dt = pd.Timestamp("2024-02-02T11:00:00", tz="UTC")
    sized = [make_sized(trade, notional=5000.0)]
    idx = session_index(10)
    prices = pd.DataFrame({"TICKS": [95.0]}, index=idx)
    out = M.account_day_change_dollars(sized, idx, prices, {"TICKS": 100.0}, SESSION_DATE, TZ)
    assert out.iloc[0] == 5000.0 * (100.0 - 95.0) / 100.0  # positive: short profits on a drop


def test_position_closed_this_session_holds_realized_exit_from_close_time_onward():
    trade = make_trade("2024-01-15", "2024-02-02", "TICKA", pct_change=5.0)
    trade.close_dt = pd.Timestamp("2024-02-02T14:00:00", tz="UTC")
    sized = [make_sized(trade, notional=5000.0)]  # realized pnl_dollars = 250.0
    idx = session_index(9, 12, 14, 16)
    prices = pd.DataFrame({"TICKA": [102.0, 101.0, 105.0, 999.0]}, index=idx)  # post-close price must be ignored

    out = M.account_day_change_dollars(sized, idx, prices, {"TICKA": 100.0}, SESSION_DATE, TZ)
    assert out.iloc[0] == 5000.0 * 0.02
    assert out.iloc[1] == 5000.0 * 0.01
    assert out.iloc[2] == 250.0  # snaps to the exact realized dollar amount at close
    assert out.iloc[3] == 250.0  # stays flat after close, ignoring the (bogus) later price


def test_position_closed_before_this_session_is_excluded():
    trade = make_trade("2024-01-15", "2024-01-20", "TICKC", pct_change=3.0)
    sized = [make_sized(trade, notional=5000.0)]
    idx = session_index(9, 16)
    out = M.account_day_change_dollars(sized, idx, pd.DataFrame({"TICKC": [1.0, 1.0]}, index=idx), {}, SESSION_DATE, TZ)
    assert (out == 0.0).all()


def test_skipped_trades_never_contribute():
    trade = make_trade("2024-01-15", "2030-01-01", "TICKX", pct_change=0.0, is_open=True)
    sized = [make_sized(trade, notional=5000.0, admitted=False)]
    idx = session_index(9)
    out = M.account_day_change_dollars(sized, idx, pd.DataFrame({"TICKX": [999.0]}, index=idx), {"TICKX": 1.0}, SESSION_DATE, TZ)
    assert out.iloc[0] == 0.0


def test_missing_previous_close_skips_the_position_instead_of_guessing():
    # Closes during the session -- so it isn't dropped for being held-
    # overnight-and-still-open (see the dedicated test for that case) --
    # but there's no previous close on record for its ticker to mark it
    # from before that.
    trade = make_trade("2024-01-15", "2024-02-02", "TICKN", pct_change=1.0)
    trade.close_dt = pd.Timestamp("2024-02-02T11:00:00", tz="UTC")
    sized = [make_sized(trade, notional=5000.0)]
    idx = session_index(9)
    out = M.account_day_change_dollars(sized, idx, pd.DataFrame({"TICKN": [110.0]}, index=idx), {"TICKN": None}, SESSION_DATE, TZ)
    assert out.iloc[0] == 0.0


def test_invariant_prior_close_equity_plus_day_change_equals_published_equity():
    """No position here is held-overnight-and-still-open across the session
    boundary -- account_day_change_dollars drops that case entirely (see
    its own docstring), so it never contributes to either side of this
    invariant. Every position that DOES contribute has one unambiguous
    basis (entry price if opened today, prev-close-then-realized-exit if
    closed today), so prior-close equity plus today's change must equal
    the account's published equity total exactly.
    """
    # C: closed entirely before this session -- already realized into
    # prior_close_equity, contributes 0 to today's own change.
    equity_naive = pd.Series([10_000.0 + 300.0], index=[pd.Timestamp("2024-02-01")])  # C's +300 realized yesterday

    # A: opened before the session, closes DURING it -- its full realized
    # dollar move (whatever the previous close was before this session)
    # lands entirely on today, same as equity_curve's own close-date bucketing.
    trade_a = make_trade("2024-01-15", "2024-02-02", "TICKA", pct_change=5.0)  # notional*5% = 250
    trade_a.close_dt = pd.Timestamp("2024-02-02T14:00:00", tz="UTC")
    # B: opened DURING the session, still open at the end -- entry price
    # and "previous close" coincide trivially, since it never existed
    # before today.
    trade_b = make_trade("2024-02-02", "2030-01-01", "TICKB", pct_change=0.0, is_open=True, entry_price=100.0)
    trade_b.open_dt = pd.Timestamp("2024-02-02T15:00:00", tz="UTC")

    sized = [make_sized(trade_a, notional=5000.0), make_sized(trade_b, notional=5000.0)]
    idx = session_index(9, 14, 15, 16)
    prices = pd.DataFrame({
        "TICKA": [102.0, 105.0, 105.0, 105.0],
        "TICKB": [None, None, 102.0, 104.0],
    }, index=idx)
    prev_close = {"TICKA": 100.0}

    prior_equity = M.prior_close_equity(equity_naive, SESSION_DATE, ACCOUNT_SIZE)
    day_change = M.account_day_change_dollars(sized, idx, prices, prev_close, SESSION_DATE, TZ)

    assert prior_equity == 10_300.0
    assert day_change.iloc[-1] == 250.0 + 5000.0 * 0.04  # A's realized 250 + B's unrealized 200

    expected_published_equity = ACCOUNT_SIZE + 300.0 + 250.0 + 200.0
    assert prior_equity + day_change.iloc[-1] == expected_published_equity
