from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import metrics as M
from helpers import make_trade, make_sized

TZ = "UTC"
ACCOUNT_SIZE = 10_000.0


def test_daily_pnl_groups_by_close_date_and_ignores_open_or_skipped():
    admitted_closed = make_sized(make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0))
    admitted_open = make_sized(make_trade("2024-01-01", "2030-01-01", "T2", pct_change=1.0, is_open=True))
    skipped_closed = make_sized(make_trade("2024-01-01", "2024-01-05", "T3", pct_change=99.0), admitted=False)

    pnl = M.daily_pnl([admitted_closed, admitted_open, skipped_closed], TZ)
    assert list(pnl.index.date) == [pd.Timestamp("2024-01-05").date()]
    assert pnl.iloc[0] == 500.0  # 5000 * 10%


def test_equity_curve_cumulates_from_account_size():
    sized = [
        make_sized(make_trade("2024-01-01", "2024-01-05", "T1", pct_change=10.0)),  # +500
        make_sized(make_trade("2024-01-02", "2024-01-06", "T2", pct_change=-4.0)),  # -200
    ]
    pnl = M.daily_pnl(sized, TZ)
    equity = M.equity_curve(pnl, ACCOUNT_SIZE)
    assert float(equity.iloc[-1]) == ACCOUNT_SIZE + 500.0 - 200.0
    # Every calendar day in between is present (ffill'd via cumsum), not just
    # the two days with an actual close.
    assert len(equity) == 2


def test_max_drawdown_finds_peak_before_trough():
    equity = pd.Series(
        [10000.0, 10500.0, 10300.0, 10200.0, 10600.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )
    dd = M.max_drawdown(equity)
    assert abs(dd["max_drawdown_pct"] - (-100 * 300 / 10500)) < 1e-9
    assert dd["peak_date"] == "2024-01-02"
    assert dd["trough_date"] == "2024-01-04"
    assert dd["recovery_date"] == "2024-01-05"


def test_max_drawdown_recovery_date_is_none_if_never_recovered():
    equity = pd.Series([10000.0, 10500.0, 9000.0], index=pd.date_range("2024-01-01", periods=3, freq="D"))
    dd = M.max_drawdown(equity)
    assert dd["recovery_date"] is None


@pytest.mark.parametrize("returns", [
    pd.Series(dtype=float),
    pd.Series([0.01]),
    pd.Series([0.0, 0.0, 0.0]),
])
def test_sharpe_and_sortino_never_nan_on_degenerate_input(returns):
    # len<2 or zero variance would divide 0/0 -> NaN under a naive formula.
    # NaN passes an "== 0" check but breaks the frontend's JSON.parse, so
    # these must return a plain 0.0 instead.
    sharpe = M.sharpe_ratio(returns)
    sortino = M.sortino_ratio(returns)
    assert sharpe == 0.0
    assert sortino == 0.0
    assert not math.isnan(sharpe)
    assert not math.isnan(sortino)


def test_alpha_beta_degenerate_benchmark_returns_zeros_not_nan():
    strategy = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    flat_benchmark = pd.Series([0.0, 0.0, 0.0], index=pd.date_range("2024-01-01", periods=3))
    result = M.alpha_beta(strategy, flat_benchmark)
    assert result == {"alpha_annualized_pct": 0.0, "beta": 0.0, "r_squared": 0.0}


def test_profit_factor_is_none_not_infinity_when_there_are_no_losses():
    trades = [make_trade("2024-01-01", "2024-01-02", "T1", pct_change=5.0)]
    stats = M.trade_level_stats(trades)
    assert stats["profit_factor"] is None
    # None (JSON null) must round-trip through strict JSON, unlike float("inf").
    json.dumps(stats, allow_nan=False)


def test_trade_level_stats_and_ticker_concentration_are_json_safe():
    trades = [
        make_trade("2024-01-01", "2024-01-02", "T1", pct_change=5.0),
        make_trade("2024-01-02", "2024-01-03", "T1", pct_change=-3.0),
        make_trade("2024-01-03", "2030-01-01", "T2", pct_change=1.0, is_open=True),
    ]
    sized = [make_sized(t) for t in trades]
    stats = M.trade_level_stats(trades)
    conc = M.ticker_concentration(sized)
    json.dumps({"stats": stats, "conc": conc}, allow_nan=False)


def test_compliance_panel_is_json_safe_with_no_market_data():
    sized = [make_sized(make_trade("2024-01-01", "2024-01-02", "T1", pct_change=5.0))]
    panel = M.compliance_panel(
        cumulative_return_pct=5.0, max_drawdown_pct=-1.0, sized_trades=sized, pct_skipped=0.0,
        market_snapshot={"T1": {"last_price": None, "avg_volume": None}},
        compliance_cfg={
            "profit_target_pct": 10.0, "max_loss_pct": -5.0,
            "concentration_band_pct": [30.0, 50.0], "min_share_price": 5.0, "min_avg_volume": 100000.0,
        },
    )
    json.dumps(panel, allow_nan=False)
