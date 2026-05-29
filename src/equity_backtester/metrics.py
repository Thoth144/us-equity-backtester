"""Performance metrics for backtest results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from math import comb, e, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
_EULER_MASCHERONI = 0.5772156649015329


@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    n_days: int


@dataclass
class PBOResult:
    pbo: float                 # P(in-sample-best strategy ranks below the OOS median)
    prob_oos_loss: float       # P(the in-sample-best strategy earns <= 0 out-of-sample)
    logits: np.ndarray         # logit of the OOS relative rank, one per CSCV split
    n_splits: int              # number of CSCV splits = C(n_partitions, n_partitions / 2)
    n_strategies: int


def summarize(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> PerformanceSummary:
    """Summary statistics for a daily equity curve.

    `risk_free_rate` is an annualized rate (e.g. 0.04 = 4%/yr).
    """
    if equity_curve.empty:
        raise ValueError("equity_curve is empty")

    returns = equity_curve.pct_change().dropna()
    n_days = len(returns)
    if n_days == 0:
        raise ValueError("not enough data to compute returns")

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    years = n_days / TRADING_DAYS_PER_YEAR
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    sigma = returns.std(ddof=0)
    annual_vol = float(sigma * np.sqrt(TRADING_DAYS_PER_YEAR))

    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    sharpe = float(excess.mean() / sigma * np.sqrt(TRADING_DAYS_PER_YEAR)) if sigma > 0 else 0.0

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_dd = float(drawdown.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

    return PerformanceSummary(
        total_return=total_return,
        cagr=float(cagr),
        annual_volatility=annual_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        calmar=calmar,
        n_days=int(n_days),
    )


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    sr_benchmark: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """P(true Sharpe > sr_benchmark), per Bailey & Lopez de Prado (2014).

    Accounts for sample length, skew, and kurtosis of the return distribution.
    `sr_benchmark` is an annualized Sharpe. Returns a probability in [0, 1].
    """
    sr, n, skew, kurt = _sharpe_and_moments(returns)
    sr_star = sr_benchmark / sqrt(periods_per_year)
    return _psr_core(sr, n, skew, kurt, sr_star)


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    trial_sharpes: list[float] | np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Deflated Sharpe Ratio, per Bailey & Lopez de Prado (2014).

    Raises the PSR benchmark to the expected maximum Sharpe under the null of
    zero true Sharpe across N independent trials, then returns the PSR against
    that threshold. `trial_sharpes` are the annualized Sharpes of every config
    tried (N = len). Requires N >= 2.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n_trials = len(trials)
    if n_trials < 2:
        raise ValueError(
            "deflated_sharpe_ratio needs >= 2 trials; "
            "use probabilistic_sharpe_ratio for a single strategy"
        )

    sr, n, skew, kurt = _sharpe_and_moments(returns)
    var_trials = float(np.var(trials, ddof=1)) / periods_per_year  # per-period units
    nd = NormalDist()
    expected_max = (
        (1.0 - _EULER_MASCHERONI) * nd.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_MASCHERONI * nd.inv_cdf(1.0 - 1.0 / (n_trials * e))
    )
    sr_star = sqrt(var_trials) * expected_max
    return _psr_core(sr, n, skew, kurt, sr_star)


def _sharpe_and_moments(returns: pd.Series | np.ndarray) -> tuple[float, int, float, float]:
    """Per-observation Sharpe, count, skewness, and (non-excess) kurtosis."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        raise ValueError("need at least 3 return observations")
    mu = r.mean()
    sigma = r.std(ddof=0)
    if sigma == 0:
        raise ValueError("returns have zero variance; Sharpe is undefined")
    z = (r - mu) / sigma
    return mu / sigma, n, float((z**3).mean()), float((z**4).mean())


def _psr_core(sr: float, n: int, skew: float, kurt: float, sr_star: float) -> float:
    """PSR in per-observation units (sr, sr_star per-observation; kurt normal = 3)."""
    denom = sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr)
    return float(NormalDist().cdf((sr - sr_star) * sqrt(n - 1) / denom))


def probability_of_backtest_overfitting(
    returns: pd.DataFrame | np.ndarray,
    *,
    n_partitions: int = 16,
    metric: Callable[[np.ndarray], np.ndarray] | None = None,
) -> PBOResult:
    """Probability of Backtest Overfitting via CSCV, per Bailey et al. (2017).

    `returns` is a (T observations x N strategies) matrix — one column per config
    tried. Combinatorially Symmetric Cross-Validation splits the T rows into
    `n_partitions` even blocks, then for every way of assigning half the blocks to
    in-sample (IS) and the complementary half to out-of-sample (OOS) — C(S, S/2)
    splits — it picks the IS-best strategy and records where that strategy's OOS
    performance ranks among all N. PBO is the fraction of splits where the IS
    winner lands in the *bottom* half OOS (logit of its relative rank <= 0): the
    rate at which selecting on in-sample performance buys you a below-median
    out-of-sample strategy. PBO near 0.5 means in-sample ranking carries no OOS
    information (pure overfitting); PBO near 0 means the selection generalizes.

    `metric` maps a (rows x N) block to an N-vector of per-strategy scores; it
    defaults to the per-column Sharpe (scale-free, so columns on different return
    scales are comparable). Requires N >= 2 and T >= `n_partitions`.
    """
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 2")
    if metric is None:
        metric = _column_sharpe

    matrix = np.asarray(returns, dtype=float)
    n_obs, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError("need at least 2 strategies to rank")
    if n_obs < n_partitions:
        raise ValueError("need at least n_partitions observations")

    blocks = np.array_split(np.arange(n_obs), n_partitions)
    half = n_partitions // 2
    logits: list[float] = []
    oos_selected: list[float] = []
    for combo in combinations(range(n_partitions), half):
        chosen = set(combo)
        in_rows = np.concatenate([blocks[b] for b in combo])
        out_rows = np.concatenate([blocks[b] for b in range(n_partitions) if b not in chosen])
        r_is = metric(matrix[in_rows])
        r_oos = metric(matrix[out_rows])
        best = int(np.argmax(r_is))
        omega = _relative_rank(r_oos, best) / (n_strategies + 1)
        logits.append(float(np.log(omega / (1.0 - omega))))
        oos_selected.append(float(r_oos[best]))

    logit_arr = np.asarray(logits, dtype=float)
    oos_arr = np.asarray(oos_selected, dtype=float)
    return PBOResult(
        pbo=float(np.mean(logit_arr <= 0.0)),
        prob_oos_loss=float(np.mean(oos_arr <= 0.0)),
        logits=logit_arr,
        n_splits=comb(n_partitions, half),
        n_strategies=n_strategies,
    )


def _column_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (rows x N) block; zero where a column has no spread."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0.0, mu / sd, 0.0)


def _relative_rank(scores: np.ndarray, idx: int) -> float:
    """Midrank of scores[idx] within scores, in [1, len] (ties share the average)."""
    value = scores[idx]
    return float(np.sum(scores < value) + (np.sum(scores == value) + 1) / 2.0)
