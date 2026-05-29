"""Volatility-targeting overlay tests — synthetic data, no network."""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.risk import volatility_target_weights


def _alternating_returns(magnitudes, start="2019-01-01"):
    """Daily returns that alternate +m/-m so trailing std (ddof=0) == m exactly."""
    vals = [m if i % 2 == 0 else -m for i, m in enumerate(magnitudes)]
    dates = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=dates)


def test_constant_vol_gives_analytic_constant_leverage():
    a = 0.01
    rets = _alternating_returns([a] * 400)
    rb = rets.index[100::20]
    weights = pd.DataFrame(1.0, index=rb, columns=["A"])
    out = volatility_target_weights(weights, rets, target_vol=0.10,
                                    window=60, max_leverage=10.0)
    expected = 0.10 / (a * np.sqrt(252.0))  # leverage = target / realized vol
    assert np.allclose(out["A"].to_numpy(), expected)


def test_levers_up_calm_down_wild():
    rets = _alternating_returns([0.004] * 200 + [0.02] * 200)
    rb = rets.index[[120, 360]]  # one date in the calm half, one in the wild half
    weights = pd.DataFrame(1.0, index=rb, columns=["A"])
    out = volatility_target_weights(weights, rets, target_vol=0.10,
                                    window=60, max_leverage=10.0)
    assert out.loc[rb[0], "A"] > out.loc[rb[1], "A"]


def test_leverage_is_capped():
    rets = _alternating_returns([0.0] * 300)  # zero realized vol
    rb = rets.index[100::20]
    weights = pd.DataFrame(1.0, index=rb, columns=["A"])
    out = volatility_target_weights(weights, rets, target_vol=0.10,
                                    window=60, max_leverage=3.0)
    assert np.allclose(out["A"].to_numpy(), 3.0)


def test_warmup_dates_pass_through_unscaled():
    rets = _alternating_returns([0.01] * 100)
    rb = rets.index[[2]]  # before min_periods -> no vol estimate yet
    weights = pd.DataFrame(1.0, index=rb, columns=["A"])
    out = volatility_target_weights(weights, rets, target_vol=0.10, window=60)
    assert out.loc[rb[0], "A"] == 1.0


def test_leverage_is_causal():
    rets = _alternating_returns([0.01] * 300)
    rb = rets.index[100::20]
    weights = pd.DataFrame(1.0, index=rb, columns=["A"])
    base = volatility_target_weights(weights, rets, window=60)
    shocked = rets.copy()
    shocked.iloc[-1] = 5.0  # a spike strictly after every rebalance date
    after = volatility_target_weights(weights, shocked, window=60)
    assert np.allclose(base.to_numpy(), after.to_numpy())


def test_scaling_preserves_dollar_neutrality():
    rets = _alternating_returns([0.01] * 300)
    rb = rets.index[100::20]
    weights = pd.DataFrame({"A": 0.5, "B": -0.5}, index=rb)
    out = volatility_target_weights(weights, rets, window=60)
    assert np.allclose(out.sum(axis=1).to_numpy(), 0.0)


def test_rejects_nonpositive_target():
    weights = pd.DataFrame({"A": [1.0]}, index=pd.bdate_range("2020-01-01", periods=1))
    with pytest.raises(ValueError):
        volatility_target_weights(weights, pd.Series(dtype=float), target_vol=0.0)
