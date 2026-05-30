# us-equity-backtester

A research platform for US-equity cross-sectional strategies, built around a
single obsession: **not lying to yourself**. Every stage — universe
construction, fundamentals, signals, model validation, portfolio simulation —
is designed to be point-in-time and leakage-safe, and the validation layer is
built to *detect* the overfitting that makes most backtests worthless.

It started as a vectorized SMA-crossover backtester (still here, still the
`backtest` CLI) and grew into a full pipeline in the spirit of Gu-Kelly-Xiu
(2020) and López de Prado (2018): factor signals → ML return forecasts →
cost-aware portfolios → honest out-of-sample evaluation.

> **The honest result first.** Run on a survivor-only S&P 500 universe, the
> price + fundamental signals here produce an information coefficient around
> **0.03 (t ≈ 1.6)** — not distinguishable from noise — and the SMA(50, 200)
> rule (~15.9% CAGR, 0.99 Sharpe) **underperforms** equal-weight buy-and-hold
> (~18.2% CAGR, 1.02 Sharpe). There is no robust tradeable alpha in this data.
> That's the point: the tooling is built to reach that conclusion instead of
> manufacturing a false one. See [`dashboard.html`](dashboard.html) for the
> full research write-up.

## What makes it "US-specialized" and honest

- **Point-in-time S&P 500 / 600 membership** reconstructed from the Wikipedia
  changes table — no survivorship bias from using today's index on history.
- **Filing-date fundamentals** from SEC EDGAR: every datapoint is gated on the
  date it was actually filed, never the fiscal-period end (the #1 silent
  lookahead bug in fundamental backtests).
- **NYSE trading calendar** (`pandas_market_calendars`, `XNYS`) and **adjusted
  prices** (yfinance `auto_adjust`) — splits and dividends back-adjusted.
- **Realistic US transaction costs** — SEC Section 31 fee, FINRA Trading
  Activity Fee, configurable commission/slippage, and a Corwin-Schultz bid-ask
  spread estimator for per-name costs.
- **Leakage-safe validation** — purged walk-forward and combinatorial purged
  cross-validation (CPCV), plus the overfitting statistics (PBO, Deflated and
  Probabilistic Sharpe) that tell you whether a backtest is real.
- **Next-open execution, no look-ahead** — signals use close-of-day data and
  trade at the following session's open.

## Install

```bash
pip install -e ".[dev]"
```

Python ≥ 3.10. Core deps: pandas, numpy, yfinance, pandas-market-calendars,
pyarrow, lxml, requests, statsmodels, scikit-learn.

## Two ways in

### 1. The CLI: an SMA-crossover backtest

```bash
# Default: current S&P 500, 2015-2025, SMA(50, 200), $100k
backtest

# Survivorship-bias-free: reconstruct historical membership
backtest --point-in-time --start 2010-01-01

# A custom basket, exporting the equity curve
backtest --tickers AAPL MSFT NVDA GOOGL --start 2018-01-01 --output equity.csv
```

### 2. The research API: signals → forecast → portfolio → validation

```python
from equity_backtester import (
    sp500_membership_panel, momentum_signal, low_vol_signal, value_signal,
    zscore_cross_section, forward_returns, monthly_rebalance_dates,
    build_design_matrix, fit_cross_sectional_forecast,
    scores_to_weights, backtest_portfolio, CostModel,
    probability_of_backtest_overfitting, deflated_sharpe_ratio,
)
from equity_backtester.data import load_ohlc

# 1. Survivorship-free universe + prices
panel = sp500_membership_panel("2010-01-01", "2024-12-31")
ohlc = load_ohlc(list(panel.columns), "2010-01-01", "2024-12-31", cache_dir=".cache")
closes = ohlc["Close"].dropna(axis=1, how="all").ffill()

# 2. Cross-sectional signals, z-scored and combined
dates = monthly_rebalance_dates(closes)
signals = {
    "momentum": zscore_cross_section(momentum_signal(closes, dates)),
    "low_vol":  zscore_cross_section(low_vol_signal(closes, dates)),
}
fwd = forward_returns(closes, dates)

# 3. Leakage-safe ML forecast: rank IC + t-stat vs a Ridge baseline
X, y = build_design_matrix(signals, fwd)
result = fit_cross_sectional_forecast(X, y, train_size=36, test_size=1, purge=1)
print(f"IC {result.mean_ic:+.4f}  t {result.ic_tstat:+.2f}  (Ridge {result.baseline_mean_ic:+.4f})")

# 4. Scores -> cost-aware long-short portfolio
scores = result.predictions.unstack()              # (date x ticker)
weights = scores_to_weights(scores, long_short=True, quantile=0.2)
pf = backtest_portfolio(weights, closes, cost_model=CostModel(), adjustment=0.5)
print(pf.equity_curve.iloc[-1], "net vs", pf.gross_equity_curve.iloc[-1], "gross")
```

## Point-in-time membership (avoid survivorship bias)

By default `backtest` uses today's S&P 500 list — implicitly assuming you knew
the future winners. `--point-in-time` reconstructs historical membership from
the Wikipedia changes table, builds a date×ticker mask, constrains signals to
names actually in the index on each date, and force-exits positions when a name
leaves.

Caveats specific to this mode:
- Wikipedia's changes table is reliable from ~2000 onward; older history is patchy.
- yfinance doesn't carry most delisted tickers; missing names are silently dropped
  (use `splice_delistings` with a delisting-inclusive source to model terminal moves).
- The mask snaps to the change's effective date — S&P's ~5-day pre-announcement
  window is not modeled.

## Execution & cost model

- Signals computed at close; trades execute at the **next session's open**.
- Equal-weight across the active signaled set (`engine`), or quantile / drifting
  weights with partial-adjustment rebalancing (`portfolio`).
- Fractional shares; the simple engine is long-only, the portfolio layer supports
  dollar-neutral long-short with optional short-borrow financing.
- Default costs (2025/2026 schedule): SEC Section 31 $27.80 per $1M of sales,
  FINRA TAF $0.000166/share capped at $9.27/trade, $0 commission, 1 bp slippage
  per side. Override via `CostModel(...)`, or pass a `corwin_schultz_spread`
  panel for per-name spreads.

## Module map

```
src/equity_backtester/
  universe.py     — point-in-time S&P 500/600 membership + current tickers
  data.py         — yfinance loader, NYSE calendar, parquet cache, delisting splice
  fundamentals.py — SEC EDGAR filing-date FactStore (gross profitability, SUE, ...)
  factors.py      — Fama-French factor data + HAC (Newey-West) factor attribution
  signals.py      — cross-sectional signal library (momentum/reversal/low-vol/value/...)
  bab.py          — Betting-Against-Beta factor (Frazzini-Pedersen)
  forecast.py     — cross-sectional ML forecast (GBM vs Ridge), purged walk-forward IC
  meta.py         — meta-labeling: a classifier that sizes the primary model's bets
  portfolio.py    — scores -> quantile weights -> cost-aware drifting backtest
  risk.py         — volatility-targeting overlay (Moreira-Muir)
  engine.py       — vectorized long-only next-open backtest loop
  costs.py        — US regulatory cost model + Corwin-Schultz spread estimator
  metrics.py      — Sharpe/CAGR/Calmar + PBO, Deflated & Probabilistic Sharpe
  walkforward.py  — walk-forward and combinatorial purged CV (CPCV)
  strategy.py     — Strategy ABC and SMACrossover
  cli.py          — `backtest` entry point
tests/            — 174 tests, no network (HTTP and yfinance are mocked)
.github/          — CI: ruff lint + pytest on Python 3.10 / 3.11 / 3.12
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how these fit together and the
correctness decision behind each stage.

## Caveats

- **Data quality.** yfinance is research-grade, not production: survivor-only,
  occasional missing bars, no delisting returns. Fundamental coverage on EDGAR
  is partial (value ~56%, profitability ~34% of name-months in testing).
- **Survivorship still bites the default path.** `--point-in-time` removes index
  survivorship, but yfinance's missing delistings reintroduce some.
- **No tax, margin, or true market-impact modeling.** Costs are linear; the
  open-fill assumption ignores liquidity and the official-open-vs-tradeable gap.
- **Statistical significance is approximate.** IC t-stats assume IID-across-dates;
  real factor returns are autocorrelated, so treat thin t-stats as optimistic.

## Extending

- **New strategy (engine):** subclass `Strategy` and implement
  `generate_signals(prices) -> DataFrame` of 0/1 signals.
- **New signal (research):** add a function returning a (dates × tickers) panel,
  z-score it with `zscore_cross_section`, and sanity-check its sign with
  `quantile_spread` before trusting it.
- **New broker:** construct `CostModel(commission_per_share=..., slippage_bps=...)`.
- **Validate honestly:** never report a single Sharpe from a parameter search —
  run `combinatorial_walk_forward` for a *distribution* of OOS paths and
  `probability_of_backtest_overfitting` / `deflated_sharpe_ratio` to discount it.
