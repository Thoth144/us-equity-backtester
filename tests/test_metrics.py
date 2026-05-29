import numpy as np
import pandas as pd
import pytest

from equity_backtester.metrics import summarize


def _curve(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_flat_curve_has_zero_returns_and_no_drawdown():
    summary = summarize(_curve([100.0] * 252))
    assert summary.total_return == pytest.approx(0.0)
    assert summary.sharpe == 0.0
    assert summary.max_drawdown == pytest.approx(0.0)
    assert summary.annual_volatility == pytest.approx(0.0)


def test_monotonic_growth_no_drawdown_positive_cagr():
    summary = summarize(_curve(np.linspace(100, 200, 252)))
    assert summary.total_return == pytest.approx(1.0)
    assert summary.max_drawdown == pytest.approx(0.0, abs=1e-12)
    assert summary.cagr > 0


def test_drawdown_computed_peak_to_trough():
    # Peak 120, trough 90 -> -25%.
    summary = summarize(_curve([100.0, 120.0, 110.0, 90.0, 100.0]))
    assert summary.max_drawdown == pytest.approx(-0.25)


def test_cagr_matches_known_doubling():
    # Double in exactly one trading year => CAGR ~= 100%.
    summary = summarize(_curve(np.linspace(100, 200, 252 + 1)))
    # n_days = 252 (returns drops first row), years = 1.0
    assert summary.n_days == 252
    assert summary.cagr == pytest.approx(1.0, rel=1e-3)


def test_empty_curve_raises():
    with pytest.raises(ValueError):
        summarize(pd.Series(dtype=float))


def test_single_point_curve_raises():
    with pytest.raises(ValueError):
        summarize(_curve([100.0]))
