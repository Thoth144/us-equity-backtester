"""Volatility-targeting risk overlay.

Scales a strategy's exposure each rebalance so its forward volatility tracks a
target, using the strategy's OWN trailing realized volatility (lagged, so the
rule is causal). Moreira-Muir (2017), "Volatility-Managed Portfolios", show
this raises the Sharpe of nearly every equity factor; Barroso-Santa-Clara
(2015) show it specifically defuses momentum's crash risk — volatility spikes
in crashes, so the overlay de-levers automatically. That subsumes the useful
part of drawdown control without the sell-at-the-bottom pathology of explicit
stops.

Use it as a second pass on top of `portfolio.backtest_portfolio`: backtest the
raw weights once to get their unlevered return stream, scale the weights with
this function, then backtest again so transaction and financing costs scale
with the levered positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_ANNUALIZE = np.sqrt(252.0)


def volatility_target_weights(
    weights: pd.DataFrame,
    strategy_returns: pd.Series,
    *,
    target_vol: float = 0.10,
    window: int = 63,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Scale each rebalance's weights toward a target annualized volatility.

    weights: target weights at rebalance dates (e.g. from `scores_to_weights`).
    strategy_returns: the UNLEVERED daily return stream of those weights (run
      `backtest_portfolio` once to get it). Realized vol is the trailing
      `window`-day std (annualized) as of each rebalance date — strictly
      backward-looking, so no look-ahead. Leverage = target_vol / realized_vol,
      clipped to [0, max_leverage]; dates without enough history pass through
      unscaled (leverage 1.0).
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    realized = (
        strategy_returns.rolling(window, min_periods=max(2, window // 3)).std(ddof=0)
        * _ANNUALIZE
    )
    scaled = weights.copy()
    for date in weights.index:
        vol = realized.asof(date)
        if not np.isfinite(vol):
            lev = 1.0          # warmup: not enough history to estimate vol
        elif vol <= 0:
            lev = max_leverage  # zero realized vol -> lever to the cap
        else:
            lev = min(target_vol / vol, max_leverage)
        scaled.loc[date] = weights.loc[date] * lev
    return scaled
