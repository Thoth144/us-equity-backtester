"""Vectorized backtesting engine.

Execution model
---------------
- Signals are computed using close-of-day data (no look-ahead).
- Target positions for bar T are shifted to execute at the OPEN of bar T+1.
- The portfolio equal-weights across the currently signaled tickers,
  rebalancing whenever the signal set changes.
- Fractional shares are permitted; this is a single-account backtest with
  no margin, no shorts, and no borrow costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .strategy import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series      # portfolio value at each close
    returns: pd.Series           # daily portfolio returns
    positions: pd.DataFrame      # share positions held at each close
    trades: pd.DataFrame         # signed share trades executed at each open
    costs: pd.DataFrame          # per-(date, ticker) transaction costs


def run_backtest(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    strategy: Strategy,
    cost_model: CostModel | None = None,
    starting_cash: float = 100_000.0,
    membership_mask: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run a vectorized long-only backtest.

    closes, opens: dates x tickers, adjusted prices aligned on the NYSE calendar.
    membership_mask: optional date x ticker bool panel. When provided, signals
      are restricted to tickers that are members on each signal date, and
      positions are force-exited the day a ticker leaves the membership set.
      Use this to eliminate survivorship bias.
    """
    if cost_model is None:
        cost_model = CostModel()

    closes, opens = closes.align(opens, join="inner")
    tickers = closes.columns
    dates = closes.index

    signals = strategy.generate_signals(closes).reindex_like(closes).fillna(0.0)

    mask = None
    if membership_mask is not None:
        mask = (membership_mask.reindex(index=dates, columns=tickers)
                .fillna(False).astype(float))
        signals = signals * mask

    # Equal-weight across the active signaled set; shift by 1 so the EOD
    # signal on day T sizes the position taken at the OPEN of day T+1.
    active = signals.sum(axis=1)
    weights = signals.div(active.where(active > 0, np.nan), axis=0).fillna(0.0)
    weights = weights.shift(1).fillna(0.0)

    if mask is not None:
        # Force exit the day a name leaves the membership set.
        weights = weights * mask

    opens_arr = opens.to_numpy(dtype=float)
    closes_arr = closes.to_numpy(dtype=float)
    weights_arr = weights.to_numpy(dtype=float)
    n_days, n_tickers = opens_arr.shape

    shares = np.zeros((n_days, n_tickers))
    trades = np.zeros((n_days, n_tickers))
    costs = np.zeros((n_days, n_tickers))
    equity = np.zeros(n_days)

    cash = float(starting_cash)
    prev_shares = np.zeros(n_tickers)

    for i in range(n_days):
        open_px = opens_arr[i]
        close_px = closes_arr[i]
        tgt_w = weights_arr[i]

        open_safe = np.nan_to_num(open_px, nan=0.0)
        close_safe = np.nan_to_num(close_px, nan=0.0)

        # Mark-to-market at the open before trading.
        pv = cash + float(np.dot(prev_shares, open_safe))

        target_dollars = tgt_w * pv
        with np.errstate(divide="ignore", invalid="ignore"):
            target_shares = np.where(open_safe > 0, target_dollars / open_safe, 0.0)

        day_trades = target_shares - prev_shares
        day_costs = cost_model.apply(day_trades, open_safe)

        cash -= float(np.dot(day_trades, open_safe)) + float(day_costs.sum())

        shares[i] = target_shares
        trades[i] = day_trades
        costs[i] = day_costs
        equity[i] = cash + float(np.dot(target_shares, close_safe))
        prev_shares = target_shares

    equity_curve = pd.Series(equity, index=dates, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=pd.DataFrame(shares, index=dates, columns=tickers),
        trades=pd.DataFrame(trades, index=dates, columns=tickers),
        costs=pd.DataFrame(costs, index=dates, columns=tickers),
    )
