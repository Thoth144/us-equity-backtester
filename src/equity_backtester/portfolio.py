"""Cost-aware portfolio construction.

Bridges cross-sectional scores (T4/T5) to an executable, net-of-cost backtest.
Two pieces:

- `scores_to_weights` turns a per-date score panel into target weights:
  equal-weighted quantile portfolios, dollar-neutral long-short (gross 1.0) or
  long-only. Equal-weight quantiles are the Fama-French standard — robust to the
  scale of the score and far less concentrated than score-proportional sizing.

- `backtest_portfolio` holds those weights between monthly rebalances (weights
  drift with close-to-close returns) and, on each rebalance, trades only a
  fraction `adjustment` of the way to the new target. Trading partway to the
  "aim" is the Garleanu-Pedersen (2013) result for linear costs: it trims
  turnover — and thus the cost drag that kills thin-alpha strategies (cf. the T5
  verdict) — while giving up little signal. Costs use the repo `CostModel` (SEC
  Section 31 + FINRA TAF + slippage), so net vs. gross shows the drag directly.

Out of scope by design: a mean-variance optimizer (needs a covariance estimate
and forecasts in return units, not ranks), quadratic market-impact costs, and
short borrow/financing fees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel


@dataclass
class PortfolioResult:
    equity_curve: pd.Series        # net-of-cost portfolio value at each close
    gross_equity_curve: pd.Series  # same trading path with costs switched off
    returns: pd.Series             # net daily returns
    turnover: pd.Series            # one-way fraction of NAV traded per rebalance
    costs: pd.Series               # dollar cost charged per rebalance


def scores_to_weights(
    scores: pd.DataFrame,
    *,
    long_short: bool = True,
    quantile: float = 0.2,
) -> pd.DataFrame:
    """Equal-weight quantile target weights from a per-date score panel.

    long_short: long the top `quantile`, short the bottom `quantile`,
      dollar-neutral with gross exposure 1.0 (each side sums to 0.5). Otherwise
      long the top `quantile` only, fully invested (weights sum to 1.0). Names
      with a NaN score on a date are excluded from that date's ranking.
    """
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date, row in scores.iterrows():
        s = row.dropna()
        n = len(s)
        k = int(n * quantile)
        if long_short:
            k = min(k, n // 2)  # keep the long and short legs disjoint
        if k < 1:
            continue
        ranked = s.sort_values()
        if long_short:
            weights.loc[date, ranked.index[-k:]] = 0.5 / k
            weights.loc[date, ranked.index[:k]] = -0.5 / k
        else:
            weights.loc[date, ranked.index[-k:]] = 1.0 / k
    return weights


def backtest_portfolio(
    target_weights: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    cost_model: CostModel | None = None,
    adjustment: float = 1.0,
    starting_cash: float = 100_000.0,
    borrow_fee_bps: float = 0.0,
    spread_panel: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Simulate a drifting weight portfolio with partial-adjustment rebalancing.

    target_weights: rows at rebalance dates (a subset of `closes` dates), columns
      a subset of `closes` columns. Between rebalances weights drift with
      close-to-close returns; on each rebalance date the book trades a fraction
      `adjustment` of the gap to the new target, paying `cost_model` on the
      traded shares. Weights set on a rebalance date earn from the next day.
    borrow_fee_bps: annual short-borrow fee, accrued daily on the short-leg
      notional (0 disables it). Reduces net equity only; the gross curve stays
      cost-free.
    spread_panel: optional per-(date, ticker) *proportional* spread (e.g. from
      `corwin_schultz_spread`). When given, each name pays half its own spread
      per side instead of the flat `cost_model.slippage_bps`; missing names fall
      back to the flat rate. None reproduces the prior flat-cost behavior.
    """
    if not 0.0 <= adjustment <= 1.0:
        raise ValueError("adjustment must be in [0, 1]")
    if cost_model is None:
        cost_model = CostModel()
    borrow_daily = borrow_fee_bps / 10_000.0 / 252.0

    tickers = closes.columns
    dates = closes.index
    tw = target_weights.reindex(columns=tickers)
    targets = {d: tw.loc[d].fillna(0.0).to_numpy(dtype=float) for d in tw.index if d in dates}

    # Per-name one-way slippage in bps (half the proportional spread), by date.
    spread_by_date: dict = {}
    if spread_panel is not None:
        sp = spread_panel.reindex(columns=tickers)
        for d in tw.index:
            if d in sp.index:
                row = sp.loc[d].to_numpy(dtype=float) / 2.0 * 10_000.0
                spread_by_date[d] = np.where(np.isfinite(row), row, cost_model.slippage_bps)

    rets = closes.pct_change(fill_method=None).fillna(0.0).to_numpy(dtype=float)
    closes_arr = np.nan_to_num(closes.to_numpy(dtype=float), nan=0.0)
    n_days, n_tickers = closes_arr.shape

    w = np.zeros(n_tickers)
    pv = float(starting_cash)
    pv_gross = float(starting_cash)
    equity = np.zeros(n_days)
    gross = np.zeros(n_days)
    turnover_log: dict = {}
    cost_log: dict = {}

    for i in range(n_days):
        r = rets[i]
        port_ret = float(np.dot(w, r))
        short_frac = float(np.maximum(-w, 0.0).sum())  # short notional held today
        denom = 1.0 + port_ret
        if denom != 0.0:
            w = w * (1.0 + r) / denom  # drift weights with the day's returns
        pv *= 1.0 + port_ret
        pv_gross *= 1.0 + port_ret
        pv -= pv * short_frac * borrow_daily  # short-borrow financing (net only)

        date = dates[i]
        if date in targets:
            delta = adjustment * (targets[date] - w)
            px = closes_arr[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                shares = np.where(px > 0, delta * pv / px, 0.0)
            cost = float(cost_model.apply(shares, px, spread_bps=spread_by_date.get(date)).sum())
            pv -= cost
            turnover_log[date] = float(np.abs(delta).sum())
            cost_log[date] = cost
            w = w + delta

        equity[i] = pv
        gross[i] = pv_gross

    equity_curve = pd.Series(equity, index=dates, name="equity")
    return PortfolioResult(
        equity_curve=equity_curve,
        gross_equity_curve=pd.Series(gross, index=dates, name="gross_equity"),
        returns=equity_curve.pct_change(fill_method=None).fillna(0.0),
        turnover=pd.Series(turnover_log, name="turnover", dtype=float),
        costs=pd.Series(cost_log, name="cost", dtype=float),
    )
