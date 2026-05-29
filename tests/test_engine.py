import numpy as np
import pandas as pd
import pytest

from equity_backtester.costs import CostModel
from equity_backtester.engine import run_backtest
from equity_backtester.strategy import Strategy


class AlwaysLong(Strategy):
    def generate_signals(self, prices):
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)


class NeverLong(Strategy):
    def generate_signals(self, prices):
        return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


def _flat(n=30, value=100.0):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"A": [value] * n}, index=dates)


def _no_fees() -> CostModel:
    return CostModel(
        commission_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        finra_taf_cap=0.0,
        slippage_bps=0.0,
    )


def test_no_trade_strategy_keeps_cash_flat():
    res = run_backtest(_flat(), _flat(), NeverLong(), _no_fees(), starting_cash=10_000.0)
    assert (res.equity_curve == 10_000.0).all()
    assert (res.trades == 0.0).all().all()
    assert (res.positions == 0.0).all().all()


def test_always_long_tracks_price_in_uptrend():
    n = 50
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=dates)

    res = run_backtest(prices, prices, AlwaysLong(), _no_fees(), starting_cash=10_000.0)

    # Day 0: signal=1 but shifted -> 0, so no trade. First buy on day 1 at price[1].
    # End at price[-1]. Final equity = 10_000 * price[-1] / price[1].
    expected = 10_000 * (prices["A"].iloc[-1] / prices["A"].iloc[1])
    assert float(res.equity_curve.iloc[-1]) == pytest.approx(expected, rel=1e-6)


def test_execution_lags_signal_by_one_day():
    """The trade for an EOD signal must execute on the NEXT open, not the same day."""
    n = 30
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n}, index=dates)

    class StepOn(Strategy):
        def generate_signals(self, prices):
            sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            sig.iloc[10:] = 1.0
            return sig

    res = run_backtest(prices, prices, StepOn(), _no_fees())

    # Signal flips at index 10 (EOD) -> trade executes at open of index 11.
    assert (res.trades.iloc[:11] == 0.0).all().all()
    assert float(res.trades.iloc[11, 0]) > 0


def test_equal_weight_across_signaled_set():
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n, "B": [50.0] * n}, index=dates)

    res = run_backtest(prices, prices, AlwaysLong(), _no_fees(), starting_cash=10_000.0)

    # After the shift, the first trading day is index 1; both names get 50% of PV.
    # Each leg gets $5000: 50 shares of A (@ $100), 100 shares of B (@ $50).
    pos = res.positions.iloc[-1]
    assert float(pos["A"]) == pytest.approx(50.0, rel=1e-6)
    assert float(pos["B"]) == pytest.approx(100.0, rel=1e-6)
    # Flat prices -> equity stays at starting cash.
    assert float(res.equity_curve.iloc[-1]) == pytest.approx(10_000.0, rel=1e-6)


def test_costs_reduce_pnl():
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n}, index=dates)

    with_costs = run_backtest(prices, prices, AlwaysLong(),
                              CostModel(slippage_bps=10.0), starting_cash=10_000.0)
    no_costs = run_backtest(prices, prices, AlwaysLong(), _no_fees(), starting_cash=10_000.0)

    assert float(with_costs.equity_curve.iloc[-1]) < float(no_costs.equity_curve.iloc[-1])
    assert float(with_costs.costs.to_numpy().sum()) > 0


def test_returns_series_aligns_with_equity_curve():
    res = run_backtest(_flat(), _flat(), AlwaysLong(), _no_fees())
    assert len(res.returns) == len(res.equity_curve)
    assert res.returns.index.equals(res.equity_curve.index)


def test_membership_mask_excludes_non_members():
    """A ticker outside the mask never accumulates positions or trades."""
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        {"A": [100.0] * n, "B": [50.0] * n, "C": [25.0] * n}, index=dates,
    )

    mask = pd.DataFrame(
        {"A": [True] * n, "B": [True] * n, "C": [False] * n}, index=dates,
    )

    res = run_backtest(prices, prices, AlwaysLong(), _no_fees(),
                      starting_cash=10_000.0, membership_mask=mask)

    assert (res.positions["C"] == 0.0).all()
    assert (res.trades["C"] == 0.0).all()


def test_membership_mask_admits_new_members_with_one_day_lag():
    """A new member is traded one day after it first joins the mask."""
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n, "B": [50.0] * n}, index=dates)
    mask = pd.DataFrame(
        {"A": [True] * n, "B": [False] * 10 + [True] * 10}, index=dates,
    )

    res = run_backtest(prices, prices, AlwaysLong(), _no_fees(),
                      starting_cash=10_000.0, membership_mask=mask)

    # No position in B until the day after it first becomes a member (signal lag).
    assert (res.positions["B"].iloc[:11] == 0.0).all()
    assert res.positions["B"].iloc[11] > 0


def test_membership_mask_forces_exit_when_member_removed():
    """When a name leaves the mask, its position drops to zero that same day."""
    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n, "B": [50.0] * n}, index=dates)
    mask = pd.DataFrame(
        {"A": [True] * n, "B": [True] * 10 + [False] * 10}, index=dates,
    )

    res = run_backtest(prices, prices, AlwaysLong(), _no_fees(),
                      starting_cash=10_000.0, membership_mask=mask)

    # B is held during the member period (after the one-day signal lag).
    assert (res.positions["B"].iloc[1:10] > 0).all()
    # Post-shift mask drops B to zero the day membership flips, and it stays out.
    assert (res.positions["B"].iloc[10:] == 0.0).all()


def test_missing_open_carries_position_instead_of_liquidating_at_zero():
    """A held name with no open/close print (halt or data gap) must be carried at
    its last known price, not force-sold at $0 (regression: a NaN open once zeroed
    target shares, destroying the position permanently)."""
    n = 10
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame({"A": [100.0] * n}, index=dates)
    gapped = prices.copy()
    gapped.iloc[5, 0] = np.nan  # day 5: no print at all (open == close == NaN)

    res = run_backtest(gapped, gapped, AlwaysLong(), _no_fees(), starting_cash=10_000.0)

    # The gap day is a no-op: the holding is valued at its last known price, so
    # equity neither vaporizes on day 5 nor afterward.
    assert float(res.equity_curve.iloc[5]) == pytest.approx(10_000.0, rel=1e-9)
    assert float(res.equity_curve.iloc[-1]) == pytest.approx(10_000.0, rel=1e-9)
    assert float(res.positions["A"].iloc[5]) == pytest.approx(100.0, rel=1e-9)
    # Only the initial buy trades; the gap never triggers a forced sell/rebuy.
    assert (res.trades["A"].iloc[2:] == 0.0).all()
