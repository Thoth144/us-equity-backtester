import numpy as np
import pandas as pd
import pytest

from equity_backtester.costs import CostModel, corwin_schultz_spread


def _zero_extras() -> dict:
    """All cost components disabled; toggle one at a time per test."""
    return dict(
        commission_per_share=0.0,
        sec_fee_rate=0.0,
        finra_taf_per_share=0.0,
        finra_taf_cap=0.0,
        slippage_bps=0.0,
    )


def test_no_trades_means_no_cost():
    cm = CostModel()
    trades = pd.Series([0.0, 0.0], index=["A", "B"])
    prices = pd.Series([100.0, 50.0], index=["A", "B"])
    assert float(cm.apply(trades, prices).sum()) == pytest.approx(0.0)


def test_buy_pays_commission_and_slippage_only():
    # SEC + FINRA are sell-side: a buy should not incur them.
    cm = CostModel(**(_zero_extras() | {"commission_per_share": 0.005, "slippage_bps": 1.0}))
    trades = pd.Series([100.0], index=["A"])
    prices = pd.Series([50.0], index=["A"])
    cost = float(cm.apply(trades, prices).iloc[0])
    # commission: 100 * 0.005 = 0.50; slippage: 100*50*0.0001 = 0.50
    assert cost == pytest.approx(1.00)


def test_sell_pays_sec_and_finra_fees():
    cm = CostModel(**(_zero_extras() | {
        "sec_fee_rate": 27.80 / 1_000_000,
        "finra_taf_per_share": 0.000166,
        "finra_taf_cap": 9.27,
    }))
    trades = pd.Series([-100.0], index=["A"])
    prices = pd.Series([50.0], index=["A"])
    cost = float(cm.apply(trades, prices).iloc[0])
    expected = 100 * 50 * (27.80 / 1_000_000) + 100 * 0.000166
    assert cost == pytest.approx(expected)


def test_finra_taf_caps_at_max_per_trade():
    cm = CostModel(**(_zero_extras() | {
        "finra_taf_per_share": 0.000166,
        "finra_taf_cap": 9.27,
    }))
    # 100k shares * 0.000166 = 16.6, capped at 9.27.
    trades = pd.Series([-100_000.0], index=["A"])
    prices = pd.Series([50.0], index=["A"])
    assert float(cm.apply(trades, prices).iloc[0]) == pytest.approx(9.27)


def test_slippage_charged_on_both_sides():
    cm = CostModel(**(_zero_extras() | {"slippage_bps": 10.0}))  # 10 bp = 0.1%
    prices = pd.Series([50.0], index=["A"])
    buy_cost = float(cm.apply(pd.Series([100.0], index=["A"]), prices).iloc[0])
    sell_cost = float(cm.apply(pd.Series([-100.0], index=["A"]), prices).iloc[0])
    # 100 * 50 * 0.001 = 5.0 each side
    assert buy_cost == pytest.approx(5.0)
    assert sell_cost == pytest.approx(5.0)


def test_works_on_numpy_arrays():
    cm = CostModel(**(_zero_extras() | {"slippage_bps": 1.0}))
    trades = np.array([100.0, -50.0])
    prices = np.array([10.0, 20.0])
    out = cm.apply(trades, prices)
    # slippage: |t|*p*1bp = [0.1, 0.1]
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.1, 0.1])


def test_works_on_dataframes():
    cm = CostModel(**(_zero_extras() | {"commission_per_share": 0.01}))
    trades = pd.DataFrame({"A": [10.0, -5.0], "B": [0.0, 20.0]})
    prices = pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0]})
    out = cm.apply(trades, prices)
    expected = pd.DataFrame({"A": [0.10, 0.05], "B": [0.0, 0.20]})
    pd.testing.assert_frame_equal(out, expected)


# --- per-name spread override -----------------------------------------------

def test_apply_per_name_spread_overrides_flat_slippage():
    cm = CostModel(**(_zero_extras() | {"slippage_bps": 1.0}))
    trades = np.array([100.0, 100.0])
    prices = np.array([50.0, 50.0])
    # notional 5000 each; name 0 pays 10 bps, name 1 pays 2 bps -> 5.0 and 1.0.
    out = cm.apply(trades, prices, spread_bps=np.array([10.0, 2.0]))
    np.testing.assert_allclose(out, [5.0, 1.0])


def test_apply_spread_none_keeps_flat_behavior():
    cm = CostModel(**(_zero_extras() | {"slippage_bps": 3.0}))
    trades = np.array([100.0])
    prices = np.array([50.0])
    np.testing.assert_allclose(cm.apply(trades, prices),
                               cm.apply(trades, prices, spread_bps=None))


# --- Corwin-Schultz spread estimator ----------------------------------------

def test_corwin_schultz_nonnegative_and_point_in_time():
    idx = pd.bdate_range("2020-01-01", periods=3)
    high = pd.DataFrame({"X": [10.0, 10.0, 10.0]}, index=idx)
    low = pd.DataFrame({"X": [9.0, 9.0, 9.0]}, index=idx)
    s = corwin_schultz_spread(high, low)
    assert pd.isna(s["X"].iloc[0])             # needs day t-1 -> first row undefined
    assert (s["X"].iloc[1:] >= 0).all()
    assert s["X"].iloc[1] == pytest.approx(0.1053, abs=3e-3)  # known value for this band


def test_corwin_schultz_bounce_costs_more_than_trend():
    idx = pd.bdate_range("2020-01-01", periods=2)
    # BOUNCE oscillates in a fixed band (2-day range == 1-day range -> wide spread);
    # TREND marches up (2-day range >> 1-day range -> spread clamps to the floor).
    high = pd.DataFrame({"BOUNCE": [10.0, 10.0], "TREND": [10.0, 11.0]}, index=idx)
    low = pd.DataFrame({"BOUNCE": [9.0, 9.0], "TREND": [9.0, 10.0]}, index=idx)
    s = corwin_schultz_spread(high, low)
    assert s["BOUNCE"].iloc[1] > s["TREND"].iloc[1]
    assert s["TREND"].iloc[1] == pytest.approx(0.0)
