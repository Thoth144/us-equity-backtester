"""US-equities transaction cost model.

Default rates (sell-side regulatory fees, current 2025/2026 schedule):
  - SEC Section 31 fee:        $27.80 per $1M of sale proceeds (0.00278%).
  - FINRA Trading Activity Fee: $0.000166 per share sold, capped at $9.27 per trade.
  - Commission:                 $0 per share (e.g. Alpaca / Schwab / Robinhood).
  - Slippage:                   1 bp of notional, applied to both buys and sells.

Override any of these via the constructor to model a different broker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_per_share: float = 0.0
    sec_fee_rate: float = 27.80 / 1_000_000  # fraction of notional, sells only
    finra_taf_per_share: float = 0.000166    # per share, sells only
    finra_taf_cap: float = 9.27              # per trade
    slippage_bps: float = 1.0                # one-way, applied to both buys and sells

    def apply(self, trades, prices, spread_bps=None):
        """Compute total per-(date, ticker) cost given share trades and execution prices.

        `trades` and `prices` may be numpy arrays, pandas Series, or DataFrames
        of the same shape. Positive trade values are buys, negative are sells.
        Returns the same container type as the inputs, holding positive costs.

        `spread_bps`: optional per-name one-way slippage in bps (e.g. half the
        bid-ask spread from `corwin_schultz_spread`), aligned with `trades`. When
        given it replaces the flat `slippage_bps`, so thin names cost more than
        liquid ones; when None, every name pays the flat rate (prior behavior).
        """
        abs_trades = np.abs(trades)
        notional = abs_trades * prices
        bps = self.slippage_bps if spread_bps is None else spread_bps
        slippage = notional * (bps / 10_000.0)
        commission = abs_trades * self.commission_per_share

        sells = np.maximum(-trades, 0)
        sec_fee = (sells * prices) * self.sec_fee_rate
        finra_fee = np.minimum(sells * self.finra_taf_per_share, self.finra_taf_cap)

        return slippage + commission + sec_fee + finra_fee


def corwin_schultz_spread(
    high: pd.DataFrame,
    low: pd.DataFrame,
    *,
    window: int | None = None,
) -> pd.DataFrame:
    """Corwin-Schultz (2012) bid-ask spread estimator from daily high/low.

    Returns a per-(date, ticker) *proportional* spread (e.g. 0.004 = 40 bps),
    backed out of how much wider the two-day high-low range is than a single
    day's: a wide overnight range that doesn't persist is the bid-ask bounce.
    Each estimate uses days t-1 and t only, so it is point-in-time -- the value
    on date t never peeks at t+1. Negative estimates (the model's noise floor)
    clamp to 0; pass `window` to smooth with a trailing mean.

    A trade pays roughly *half* this spread per side, so feed `spread / 2 *
    10_000` (bps) to `CostModel.apply(spread_bps=)` or, panel-wise, hand the
    proportional spread to `backtest_portfolio(spread_panel=)`.
    """
    high = high.where(high > 0)
    low = low.where(low > 0)
    k = 3.0 - 2.0 * np.sqrt(2.0)

    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)                      # two single-day ranges (days t-1, t)
    hi2 = np.maximum(high, high.shift(1))        # high over the two-day window
    lo2 = np.minimum(low, low.shift(1))          # low over the two-day window
    gamma = np.log(hi2 / lo2) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.clip(lower=0.0)
    if window:
        spread = spread.rolling(window, min_periods=1).mean()
    return spread
