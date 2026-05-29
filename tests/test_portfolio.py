"""Portfolio-construction tests — synthetic data, no network.

The load-bearing test is the cost-aware one: on a high-turnover target, trading
only partway to the aim must beat full rebalancing once costs bite.
"""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.costs import CostModel
from equity_backtester.portfolio import backtest_portfolio, scores_to_weights


def _free() -> CostModel:
    return CostModel(commission_per_share=0, sec_fee_rate=0,
                     finra_taf_per_share=0, finra_taf_cap=0, slippage_bps=0)


def _alternating(n_days=13, rebal_step=2):
    """Flat prices + a target that flips long/short legs every rebalance."""
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    closes = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    rb = dates[rebal_step::rebal_step]
    rows = [[0.5, -0.5] if k % 2 == 0 else [-0.5, 0.5] for k in range(len(rb))]
    tw = pd.DataFrame(rows, index=rb, columns=["A", "B"])
    return closes, tw


# --- scores_to_weights ------------------------------------------------------

def test_long_short_is_dollar_neutral_and_gross_one():
    dates = pd.bdate_range("2020-01-01", periods=2)
    tickers = [f"T{i}" for i in range(10)]
    scores = pd.DataFrame([np.arange(10.0), np.arange(10.0)[::-1]],
                          index=dates, columns=tickers)
    w = scores_to_weights(scores, long_short=True, quantile=0.2)
    assert np.allclose(w.sum(axis=1), 0.0)        # dollar neutral
    assert np.allclose(w.abs().sum(axis=1), 1.0)  # gross exposure 1.0
    assert (w.loc[dates[0], ["T8", "T9"]] > 0).all()   # top scorers long
    assert (w.loc[dates[0], ["T0", "T1"]] < 0).all()   # bottom scorers short


def test_long_only_sums_to_one_and_nonnegative():
    dates = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"T{i}" for i in range(10)]
    scores = pd.DataFrame([np.arange(10.0)], index=dates, columns=tickers)
    w = scores_to_weights(scores, long_short=False, quantile=0.2)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert (w.to_numpy() >= 0).all()
    assert (w.loc[dates[0]] > 0).sum() == 2


def test_nan_scores_are_excluded_from_ranking():
    dates = pd.bdate_range("2020-01-01", periods=1)
    scores = pd.DataFrame([[np.nan, 1.0, 2.0, 3.0]], index=dates,
                          columns=["A", "B", "C", "D"])
    w = scores_to_weights(scores, long_short=False, quantile=0.5)
    assert w.loc[dates[0], "A"] == 0.0
    assert np.allclose(w.sum(axis=1), 1.0)


def test_scores_to_weights_rejects_bad_quantile():
    scores = pd.DataFrame([[1.0, 2.0]], columns=["A", "B"])
    with pytest.raises(ValueError):
        scores_to_weights(scores, quantile=0.0)
    with pytest.raises(ValueError):
        scores_to_weights(scores, quantile=0.6)


# --- backtest_portfolio -----------------------------------------------------

def test_tracks_single_name_return_with_no_costs():
    dates = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41], "B": 100.0}, index=dates)
    tw = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[0]])
    res = backtest_portfolio(tw, closes, cost_model=_free(), starting_cash=1000.0)
    expected = 1000.0 * closes["A"] / closes["A"].iloc[0]
    assert np.allclose(res.equity_curve.to_numpy(), expected.to_numpy())
    assert np.allclose(res.gross_equity_curve.to_numpy(), expected.to_numpy())


def test_costs_pull_net_below_gross():
    closes, tw = _alternating()
    res = backtest_portfolio(tw, closes, cost_model=CostModel(slippage_bps=50))
    assert res.equity_curve.iloc[-1] < res.gross_equity_curve.iloc[-1]
    assert (res.costs > 0).all()
    assert np.isclose(res.gross_equity_curve.iloc[-1], res.gross_equity_curve.iloc[0])


def test_zero_cost_net_equals_gross():
    closes, tw = _alternating()
    res = backtest_portfolio(tw, closes, cost_model=_free())
    assert np.allclose(res.equity_curve.to_numpy(), res.gross_equity_curve.to_numpy())


def test_lower_adjustment_cuts_turnover():
    closes, tw = _alternating()
    full = backtest_portfolio(tw, closes, cost_model=_free(), adjustment=1.0)
    part = backtest_portfolio(tw, closes, cost_model=_free(), adjustment=0.3)
    assert part.turnover.sum() < full.turnover.sum()


def test_partial_adjustment_beats_full_under_costs():
    """The headline: partway trading wins net of costs on a churny target."""
    closes, tw = _alternating()
    cm = CostModel(slippage_bps=50)
    full = backtest_portfolio(tw, closes, cost_model=cm, adjustment=1.0)
    part = backtest_portfolio(tw, closes, cost_model=cm, adjustment=0.3)
    assert part.equity_curve.iloc[-1] > full.equity_curve.iloc[-1]


def test_backtest_rejects_bad_adjustment():
    closes, tw = _alternating()
    with pytest.raises(ValueError):
        backtest_portfolio(tw, closes, adjustment=1.5)


def test_borrow_fee_drags_long_short_equity():
    closes, tw = _alternating()
    free = backtest_portfolio(tw, closes, cost_model=_free(), borrow_fee_bps=0.0)
    charged = backtest_portfolio(tw, closes, cost_model=_free(), borrow_fee_bps=300.0)
    assert charged.equity_curve.iloc[-1] < free.equity_curve.iloc[-1]
    # Financing hits net only; the cost-free gross path is unchanged.
    assert np.allclose(charged.gross_equity_curve.to_numpy(),
                       free.gross_equity_curve.to_numpy())


def test_borrow_fee_has_no_effect_without_shorts():
    dates = pd.bdate_range("2020-01-01", periods=10)
    closes = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
    tw = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[dates[0]])  # long-only
    a = backtest_portfolio(tw, closes, cost_model=_free(), borrow_fee_bps=0.0)
    b = backtest_portfolio(tw, closes, cost_model=_free(), borrow_fee_bps=500.0)
    assert np.allclose(a.equity_curve.to_numpy(), b.equity_curve.to_numpy())


# --- per-name spread panel --------------------------------------------------

def _no_extras() -> CostModel:
    return CostModel(commission_per_share=0, sec_fee_rate=0,
                     finra_taf_per_share=0, finra_taf_cap=0, slippage_bps=7.0)


def test_spread_panel_charges_more_for_wider_spreads():
    closes, tw = _alternating()
    cm = _no_extras()
    flat = backtest_portfolio(tw, closes, cost_model=cm)
    # 1% proportional spread -> 50 bps/side, far above the 7 bp flat rate.
    wide = pd.DataFrame(0.01, index=closes.index, columns=closes.columns)
    res = backtest_portfolio(tw, closes, cost_model=cm, spread_panel=wide)
    assert res.costs.sum() > flat.costs.sum()
    assert res.equity_curve.iloc[-1] < flat.equity_curve.iloc[-1]


def test_spread_panel_matching_flat_rate_reproduces_flat_costs():
    closes, tw = _alternating()
    cm = _no_extras()
    flat = backtest_portfolio(tw, closes, cost_model=cm)
    # half of 0.0014 proportional == 7 bps/side == the flat rate -> identical costs.
    sp = pd.DataFrame(7.0 * 2 / 1e4, index=closes.index, columns=closes.columns)
    same = backtest_portfolio(tw, closes, cost_model=cm, spread_panel=sp)
    assert np.allclose(same.costs.to_numpy(), flat.costs.to_numpy())


def test_spread_panel_missing_name_falls_back_to_flat_rate():
    closes, tw = _alternating()
    cm = _no_extras()
    flat = backtest_portfolio(tw, closes, cost_model=cm)
    # A is set to the flat-equivalent spread; B is absent -> B must use the flat rate,
    # so the net cost stream matches the all-flat run exactly.
    sp = pd.DataFrame({"A": 7.0 * 2 / 1e4}, index=closes.index)
    res = backtest_portfolio(tw, closes, cost_model=cm, spread_panel=sp)
    assert np.allclose(res.costs.to_numpy(), flat.costs.to_numpy())
