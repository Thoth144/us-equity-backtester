"""Tests for factor attribution — synthetic factors, no network."""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.factors import FACTOR_NAMES, factor_attribution


def _make_factors(n, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    data = {f: rng.normal(0, 0.006, n) for f in FACTOR_NAMES}
    data["Mkt-RF"] = rng.normal(0.0003, 0.01, n)
    data["RF"] = np.full(n, 0.00008)
    return pd.DataFrame(data, index=dates)


def test_recovers_known_loadings_and_alpha():
    f = _make_factors(1500, seed=1)
    noise = np.random.default_rng(2).normal(0, 0.0005, len(f))
    alpha_d = 0.0002
    excess = alpha_d + 1.0 * f["Mkt-RF"] + 0.5 * f["SMB"] - 0.3 * f["HML"] + noise
    returns = excess + f["RF"]
    res = factor_attribution(returns, factors=f)
    assert res.betas["Mkt-RF"] == pytest.approx(1.0, abs=0.05)
    assert res.betas["SMB"] == pytest.approx(0.5, abs=0.06)
    assert res.betas["HML"] == pytest.approx(-0.3, abs=0.06)
    assert res.alpha_annual == pytest.approx(alpha_d * 252, abs=0.02)


def test_pure_beta_has_insignificant_alpha():
    f = _make_factors(1500, seed=3)
    noise = np.random.default_rng(4).normal(0, 0.002, len(f))
    returns = f["RF"] + 1.0 * f["Mkt-RF"] + noise  # no alpha injected
    res = factor_attribution(returns, factors=f)
    assert abs(res.alpha_tstat) < 2.5
    assert res.betas["Mkt-RF"] == pytest.approx(1.0, abs=0.05)


def test_strong_alpha_is_significant():
    f = _make_factors(1500, seed=5)
    noise = np.random.default_rng(6).normal(0, 0.002, len(f))
    returns = f["RF"] + 0.0005 + 1.0 * f["Mkt-RF"] + noise
    res = factor_attribution(returns, factors=f)
    assert res.alpha_tstat > 3
    assert res.alpha_annual > 0
    assert res.info_ratio > 0


def test_r_squared_high_when_returns_are_mostly_factors():
    f = _make_factors(1500, seed=7)
    noise = np.random.default_rng(8).normal(0, 0.0005, len(f))  # tiny noise
    returns = f["RF"] + 1.0 * f["Mkt-RF"] + 0.5 * f["SMB"] + noise
    res = factor_attribution(returns, factors=f)
    assert res.r_squared > 0.9


def test_requires_minimum_overlap():
    f = _make_factors(20, seed=9)
    returns = f["RF"] + f["Mkt-RF"]
    with pytest.raises(ValueError):
        factor_attribution(returns, factors=f)


def test_betas_cover_all_factors():
    f = _make_factors(500, seed=10)
    returns = f["RF"] + f["Mkt-RF"]
    res = factor_attribution(returns, factors=f)
    assert set(res.betas) == set(FACTOR_NAMES)
    assert res.n_obs == 500
