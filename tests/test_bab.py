"""Betting-Against-Beta tests — synthetic CAPM data, no network.

The load-bearing tests: rolling_beta recovers known betas causally, and the
factor is *exactly* zero under pure CAPM with no alpha (the beta-neutralization
works) yet earns a positive premium when low-beta names carry positive alpha.
"""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.bab import _bab_weights, bab_factor, rolling_beta


def _capm_closes(betas, *, n_days=400, alpha=None, noise=0.0, seed=0, start="2018-01-01"):
    """Daily closes for names with given betas: r_i = alpha_i + beta_i*r_m + noise.

    Returns (closes, market_returns) on a shared business-day index.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rm = pd.Series(rng.standard_normal(n_days) * 0.01, index=dates)
    alpha = alpha or {}
    cols = {}
    for ticker, b in betas.items():
        eps = rng.standard_normal(n_days) * noise
        r = alpha.get(ticker, 0.0) + b * rm.to_numpy() + eps
        cols[ticker] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(cols, index=dates), rm


# --- rolling_beta -----------------------------------------------------------

def test_rolling_beta_recovers_known_betas():
    betas = {"A": 0.5, "B": 1.0, "C": 1.7}
    closes, rm = _capm_closes(betas, noise=0.0)
    rb = closes.index[[300]]
    out = rolling_beta(closes, rm, rb, window=200, shrink=1.0)
    assert np.allclose(out.loc[rb[0]].to_numpy(), [0.5, 1.0, 1.7], atol=1e-6)


def test_rolling_beta_shrinks_toward_one():
    betas = {"A": 0.0, "B": 2.0}  # one below 1, one above
    closes, rm = _capm_closes(betas, noise=0.0)
    rb = closes.index[[300]]
    out = rolling_beta(closes, rm, rb, window=200, shrink=0.6)
    assert np.allclose(out.loc[rb[0]].to_numpy(), [0.4, 1.6], atol=1e-6)


def test_rolling_beta_is_causal():
    betas = {"A": 0.6, "B": 1.5}
    closes, rm = _capm_closes(betas, noise=0.01, seed=1)
    rb = closes.index[200:380:20]  # all strictly before the spiked last row
    base = rolling_beta(closes, rm, rb, window=150, shrink=1.0)
    closes2, rm2 = closes.copy(), rm.copy()
    closes2.iloc[-1] *= 5.0      # a price/return shock strictly after every rb date
    rm2.iloc[-1] = 5.0
    after = rolling_beta(closes2, rm2, rb, window=150, shrink=1.0)
    assert np.allclose(base.to_numpy(), after.to_numpy())


def test_rolling_beta_warmup_is_nan():
    betas = {"A": 0.6, "B": 1.5}
    closes, rm = _capm_closes(betas, noise=0.0)
    rb = closes.index[[5]]  # far fewer than min_periods observations available
    out = rolling_beta(closes, rm, rb, window=200)
    assert out.loc[rb[0]].isna().all()


def test_rolling_beta_rejects_bad_shrink():
    closes, rm = _capm_closes({"A": 1.0})
    with pytest.raises(ValueError):
        rolling_beta(closes, rm, closes.index[[300]], shrink=1.5)


# --- _bab_weights -----------------------------------------------------------

def test_bab_weights_each_leg_sums_to_one():
    beta = pd.Series({"A": 0.4, "B": 0.9, "C": 1.1, "D": 1.8, "E": 2.3})
    w_low, w_high = _bab_weights(beta)
    assert np.isclose(w_low.sum(), 1.0)
    assert np.isclose(w_high.sum(), 1.0)


def test_bab_weights_low_leg_tilts_to_low_beta():
    beta = pd.Series({"A": 0.4, "B": 0.9, "C": 1.1, "D": 1.8, "E": 2.3})
    w_low, w_high = _bab_weights(beta)
    assert w_low.idxmax() == "A" and w_low["E"] == 0.0   # lowest beta dominates long leg
    assert w_high.idxmax() == "E" and w_high["A"] == 0.0  # highest beta dominates short leg


def test_bab_weights_are_rank_based_not_level():
    beta = pd.Series({"A": 0.4, "B": 0.9, "C": 1.1, "D": 1.8, "E": 2.3})
    monotone = np.exp(beta)  # order-preserving -> identical ranks -> identical weights
    lo1, hi1 = _bab_weights(beta)
    lo2, hi2 = _bab_weights(monotone)
    assert np.allclose(lo1.to_numpy(), lo2.to_numpy())
    assert np.allclose(hi1.to_numpy(), hi2.to_numpy())


# --- bab_factor -------------------------------------------------------------

def test_bab_factor_zero_under_pure_capm():
    """No alpha, no noise, betas known exactly -> the neutralized factor is ~0."""
    betas = {"A": 0.5, "B": 0.8, "C": 1.2, "D": 1.6, "E": 2.0}
    closes, rm = _capm_closes(betas, noise=0.0, alpha=None)
    rb = closes.index[250:399]  # daily rebalances after warmup; last has no fwd return
    res = bab_factor(closes, rm, rb, window=200, shrink=1.0)
    assert len(res.factor_returns) >= 140
    assert np.allclose(res.factor_returns.to_numpy(), 0.0, atol=1e-9)


def test_bab_factor_earns_low_beta_alpha():
    betas = {"A": 0.4, "B": 0.7, "C": 1.0, "D": 1.5, "E": 2.0}
    alpha = {t: 0.0008 * (1.0 - b) for t, b in betas.items()}  # low beta -> +alpha
    closes, rm = _capm_closes(betas, noise=0.004, alpha=alpha, seed=3)
    rb = closes.index[250:399]
    res = bab_factor(closes, rm, rb, window=200, shrink=0.6)
    assert res.factor_returns.mean() > 0.0


def test_bab_factor_beta_low_below_high():
    betas = {"A": 0.4, "B": 0.7, "C": 1.0, "D": 1.5, "E": 2.0}
    closes, rm = _capm_closes(betas, noise=0.004, seed=4)
    rb = closes.index[250:399]
    res = bab_factor(closes, rm, rb, window=200, shrink=0.6)
    assert (res.beta_low < res.beta_high).all()
