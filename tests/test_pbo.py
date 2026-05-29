"""Probability of Backtest Overfitting (CSCV) tests — synthetic, no network.

The load-bearing tests: PBO is ~0 when one strategy genuinely dominates, ~0.5
under pure noise (in-sample ranking carries no out-of-sample information), and
~1 when the cross-sectional ranking deterministically reverses out of sample.
"""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.metrics import _column_sharpe, probability_of_backtest_overfitting


def _noise(n_obs, n_strat, *, seed=0, scale=0.01):
    """A (n_obs x n_strat) frame of i.i.d. N(0, scale) returns on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_obs)
    data = rng.standard_normal((n_obs, n_strat)) * scale
    return pd.DataFrame(data, index=idx, columns=[f"s{i}" for i in range(n_strat)])


# --- the three regimes ------------------------------------------------------

def test_pbo_low_when_one_strategy_truly_dominates():
    """A column with a real, large Sharpe wins in- and out-of-sample -> PBO ~ 0."""
    df = _noise(160, 10, seed=0)
    rng = np.random.default_rng(99)
    df["s0"] = 0.02 + 0.002 * rng.standard_normal(160)  # Sharpe ~ 10/period
    res = probability_of_backtest_overfitting(df, n_partitions=8)
    assert res.pbo < 0.05


def test_pbo_near_half_under_pure_noise():
    """No strategy has an edge -> the in-sample winner is OOS-random -> PBO ~ 0.5.

    A single realization is high-variance (the CSCV splits share blocks, so they
    are far from independent); the *estimator* is what's centered at 0.5, so we
    average over many noise draws rather than betting on one lucky path.
    """
    pbos = [
        probability_of_backtest_overfitting(_noise(160, 10, seed=s), n_partitions=8).pbo
        for s in range(50)
    ]
    assert 0.4 < float(np.mean(pbos)) < 0.6


def test_pbo_high_when_ranking_reverses_out_of_sample():
    """Means ramp up across columns in block 0 and down in block 1: the in-sample
    winner is the out-of-sample loser on both splits -> PBO ~ 1."""
    n_obs, n_strat = 80, 6
    rng = np.random.default_rng(2)
    half = n_obs // 2
    data = np.empty((n_obs, n_strat))
    for i in range(n_strat):
        data[:half, i] = (i + 1) * 0.01 + rng.standard_normal(half) * 1e-4
        data[half:, i] = (n_strat - i) * 0.01 + rng.standard_normal(n_obs - half) * 1e-4
    df = pd.DataFrame(data, index=pd.bdate_range("2015-01-01", periods=n_obs))
    res = probability_of_backtest_overfitting(df, n_partitions=2)
    assert res.pbo > 0.9


# --- prob_oos_loss ----------------------------------------------------------

def test_prob_oos_loss_low_under_skill():
    df = _noise(160, 10, seed=0)
    rng = np.random.default_rng(99)
    df["s0"] = 0.02 + 0.002 * rng.standard_normal(160)
    res = probability_of_backtest_overfitting(df, n_partitions=8)
    assert res.prob_oos_loss < 0.05


def test_prob_oos_loss_near_half_under_noise():
    # Same single-realization variance as PBO -> average over noise draws.
    losses = [
        probability_of_backtest_overfitting(_noise(160, 10, seed=s), n_partitions=8).prob_oos_loss
        for s in range(50)
    ]
    assert 0.4 < float(np.mean(losses)) < 0.6


# --- bookkeeping & validation ----------------------------------------------

def test_reports_split_and_strategy_counts():
    df = _noise(120, 7, seed=3)
    res = probability_of_backtest_overfitting(df, n_partitions=8)
    assert res.n_splits == 70  # C(8, 4)
    assert len(res.logits) == 70
    assert res.n_strategies == 7


def test_rejects_odd_n_partitions():
    df = _noise(120, 5)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(df, n_partitions=7)


def test_rejects_n_partitions_below_two():
    df = _noise(120, 5)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(df, n_partitions=1)


def test_rejects_single_strategy():
    df = _noise(120, 1)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(df, n_partitions=8)


def test_rejects_too_few_observations():
    df = _noise(6, 4)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(df, n_partitions=8)


def test_custom_metric_changes_selection():
    """Selecting on negated Sharpe picks the in-sample loser, which no longer
    generalizes -> PBO climbs well above the skill case."""
    df = _noise(160, 10, seed=0)
    rng = np.random.default_rng(99)
    df["s0"] = 0.02 + 0.002 * rng.standard_normal(160)
    default = probability_of_backtest_overfitting(df, n_partitions=8)
    flipped = probability_of_backtest_overfitting(
        df, n_partitions=8, metric=lambda b: -_column_sharpe(b)
    )
    assert default.pbo < 0.05
    assert flipped.pbo > default.pbo + 0.1
