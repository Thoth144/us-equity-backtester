"""Tests for Probabilistic and Deflated Sharpe Ratios."""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from equity_backtester.metrics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def _normal_returns(n, mean, std, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n))


def test_psr_exactly_half_for_zero_mean():
    # Alternating returns -> sample mean exactly 0 -> Sharpe 0 -> PSR = 0.5.
    r = pd.Series([0.01, -0.01] * 500)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(0.5, abs=1e-9)


def test_psr_high_for_strong_long_track_record():
    r = _normal_returns(2000, mean=0.001, std=0.01, seed=1)
    assert probabilistic_sharpe_ratio(r, sr_benchmark=0.0) > 0.95


def test_psr_increases_with_sample_length():
    short = _normal_returns(100, 0.0008, 0.01, seed=3)
    long = _normal_returns(2000, 0.0008, 0.01, seed=3)
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_psr_matches_closed_form_for_the_implementations_moments():
    r = _normal_returns(500, 0.0008, 0.01, seed=4)
    arr = r.to_numpy()
    mu, sigma = arr.mean(), arr.std(ddof=0)
    sr_pp = mu / sigma
    z = (arr - mu) / sigma
    skew, kurt = (z**3).mean(), (z**4).mean()
    denom = (1 - skew * sr_pp + (kurt - 1) / 4 * sr_pp**2) ** 0.5
    expected = NormalDist().cdf(sr_pp * np.sqrt(len(arr) - 1) / denom)
    assert probabilistic_sharpe_ratio(r) == pytest.approx(expected, rel=1e-9)


def test_dsr_below_psr_when_trials_disperse():
    r = _normal_returns(1500, 0.0009, 0.01, seed=5)
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    trials = [0.2, 0.5, 0.8, 1.1, 1.4, -0.3, 0.0, 0.9]  # annualized, dispersed
    assert deflated_sharpe_ratio(r, trials) < psr


def test_dsr_equals_psr_vs_zero_when_no_trial_dispersion():
    r = _normal_returns(1500, 0.0009, 0.01, seed=7)
    # Zero variance across trials -> benchmark stays at 0 -> DSR == PSR(0).
    dsr = deflated_sharpe_ratio(r, [0.7, 0.7, 0.7])
    psr0 = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert dsr == pytest.approx(psr0, rel=1e-9)


def test_dsr_requires_two_trials():
    r = _normal_returns(500, 0.001, 0.01, seed=6)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r, [1.0])


def test_significance_rejects_too_few_observations():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(pd.Series([0.01, 0.02]))
