"""Betting-Against-Beta factor (Frazzini-Pedersen 2014).

The empirical security market line is too flat: low-beta assets earn positive
CAPM alpha and high-beta assets negative, because leverage-constrained investors
over-pay for high-beta exposure. The BAB factor harvests that spread — long the
low-beta names levered up to beta 1, short the high-beta names de-levered to
beta 1 — so the factor is market-neutral by construction and bets purely on the
low-minus-high alpha.

Three pieces:

- `rolling_beta` estimates each name's market beta over a trailing daily window
  (Cov(r_i, r_m) / Var(r_m)), then shrinks it toward 1 the way Frazzini-Pedersen
  do (`shrink * beta + (1 - shrink)`, with shrink = 0.6). It is causal — only
  data up to each rebalance date is used. Negate it for a plain low-beta signal.

- `_bab_weights` turns a cross-section of betas into the factor's rank weights:
  names are ranked on beta and weighted by rank distance from the median, with
  each leg (low and high) summing to 1. This is a continuous tilt across every
  name, not an equal-weight top/bottom quantile — which is why BAB needs its own
  construction rather than `scores_to_weights`.

- `bab_factor` assembles the leg returns, levers each leg to beta 1
  (1/beta_low on the long leg, 1/beta_high on the short leg), and returns the
  per-period factor return plus the leg betas as diagnostics.

Simplifications vs. the paper, by design: a single 1-year OLS beta (not their
separate multi-horizon correlation/volatility estimator) — it preserves the
cross-sectional *ordering*, which is all the rank weighting consumes — a caller-
supplied market proxy, and a zero risk-free rate in the leverage formula (the
financing layer lives in `portfolio.py` if BAB is later traded).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import forward_returns


@dataclass
class BabResult:
    factor_returns: pd.Series  # per-period beta-neutral BAB return
    beta_low: pd.Series        # portfolio beta of the long (low-beta) leg
    beta_high: pd.Series       # portfolio beta of the short (high-beta) leg


def rolling_beta(
    closes: pd.DataFrame,
    market_returns: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    window: int = 252,
    shrink: float = 0.6,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Trailing-window market betas per name, shrunk toward 1 (Frazzini-Pedersen).

    For each rebalance date, beta_i = Cov(r_i, r_m) / Var(r_m) over the trailing
    `window` daily returns up to that date, then shrunk: `shrink*beta + (1-shrink)`.
    `market_returns` is a daily market-proxy return series aligned to `closes`.
    Returns a (dates x tickers) panel; names with fewer than `min_periods`
    observations in the window are NaN. Negate the result for a low-beta signal.
    """
    if not 0.0 <= shrink <= 1.0:
        raise ValueError("shrink must be in [0, 1]")
    if min_periods is None:
        min_periods = max(20, window // 2)
    rets = closes.pct_change(fill_method=None)
    m = market_returns.reindex(rets.index)
    out = pd.DataFrame(index=dates, columns=closes.columns, dtype=float)
    for d in dates:
        r_win = rets.loc[:d].tail(window)
        m_win = m.loc[:d].tail(window)
        if len(m_win) < min_periods:
            continue
        mc = m_win - m_win.mean()
        var_m = float((mc * mc).mean())
        if var_m == 0 or not np.isfinite(var_m):
            continue
        cov = r_win.subtract(r_win.mean()).multiply(mc, axis=0).mean()
        beta = cov / var_m
        beta[r_win.notna().sum() < min_periods] = np.nan
        out.loc[d] = shrink * beta + (1.0 - shrink)
    return out


def _bab_weights(beta: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Frazzini-Pedersen rank weights: (w_low, w_high), each leg summing to 1.

    Names are ranked on beta (ascending) and weighted by rank distance from the
    mean rank: below-median ranks form the low-beta leg, above-median the
    high-beta leg. With k = 2 / sum|rank - mean_rank| each leg sums to 1.
    """
    z = beta.rank()
    dev = z - z.mean()
    denom = float(dev.abs().sum())
    if denom == 0.0:  # single name (or all tied): no spread to bet on
        zeros = pd.Series(0.0, index=beta.index)
        return zeros, zeros
    k = 2.0 / denom
    w_high = k * dev.clip(lower=0.0)
    w_low = k * (-dev).clip(lower=0.0)
    return w_low, w_high


def bab_factor(
    closes: pd.DataFrame,
    market_returns: pd.Series,
    dates: pd.DatetimeIndex,
    *,
    window: int = 252,
    shrink: float = 0.6,
) -> BabResult:
    """Frazzini-Pedersen Betting-Against-Beta factor on `closes`.

    Estimates trailing betas (`rolling_beta`), forms a rank-weighted low-beta
    long leg and high-beta short leg (each summing to 1), then levers each leg to
    beta 1 — long 1/beta_low of the low-beta leg, short 1/beta_high of the
    high-beta leg — so the factor is market-neutral by construction and earns the
    low-minus-high alpha spread. `market_returns` is a daily market proxy.
    Returns per-period factor returns plus the leg betas as diagnostics.
    """
    betas = rolling_beta(closes, market_returns, dates, window=window, shrink=shrink)
    fwd = forward_returns(closes, dates)
    fac: dict = {}
    bl: dict = {}
    bh: dict = {}
    for d in dates:
        if d not in fwd.index:
            continue
        beta = betas.loc[d].dropna()
        r = fwd.loc[d].reindex(beta.index)
        beta = beta[r.notna()]
        r = r.dropna()
        if len(beta) < 2:
            continue
        w_low, w_high = _bab_weights(beta)
        beta_low = float((w_low * beta).sum())
        beta_high = float((w_high * beta).sum())
        if beta_low <= 0.0 or beta_high <= 0.0:
            continue
        ret_low = float((w_low * r).sum())
        ret_high = float((w_high * r).sum())
        fac[d] = ret_low / beta_low - ret_high / beta_high
        bl[d] = beta_low
        bh[d] = beta_high
    return BabResult(
        factor_returns=pd.Series(fac, name="bab", dtype=float),
        beta_low=pd.Series(bl, name="beta_low", dtype=float),
        beta_high=pd.Series(bh, name="beta_high", dtype=float),
    )
