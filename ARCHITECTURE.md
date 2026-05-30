# Architecture

This document explains how the pieces fit together and *why* each is built the
way it is. The README covers what the project does and how to run it; this is
the map for someone reading or extending the code.

## Design principles

Every module is built against the same four constraints. When a design choice
looks more complicated than necessary, it is almost always one of these:

1. **Point-in-time, always.** A computation on date `t` may only use data that
   was knowable on date `t`. This is enforced at the source — filing dates for
   fundamentals, effective dates for index membership, lagged windows for
   estimates — not patched up later.
2. **Leakage-safe validation.** Forward-return labels overlap in time, so naive
   k-fold CV leaks the future. Cross-validation here *purges* and *embargoes*
   the boundary so a label spanning train/test can't bleed across it.
3. **Cost realism.** A thin signal that looks profitable gross is usually dead
   net. Costs (regulatory fees, slippage, bid-ask spread, borrow) are first-class,
   and every portfolio result carries both a net and a gross curve.
4. **Overfitting is the default hypothesis.** Searching parameters and reporting
   the best Sharpe is how backtests lie. The validation layer estimates the
   probability of overfitting (PBO) and deflates Sharpe ratios for the number of
   trials, so a single flattering number never stands unchallenged.

## The pipeline

Data flows left to right. Each stage is one module (or a small group), and each
consumes the stage before it.

```
 ┌────────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────┐
 │ universe   │──>│ data     │──>│ signals      │──>│ forecast  │
 │ (PIT       │   │ (prices, │   │ fundamentals │   │ meta      │
 │  members)  │   │  calendar│   │ factors, bab │   │ (ML, IC)  │
 └────────────┘   │  cache)  │   └──────────────┘   └─────┬─────┘
                  └──────────┘                            │
                                                          v
 ┌──────────────────────────┐   ┌──────────┐   ┌────────────────────┐
 │ metrics + walkforward    │<──│ engine   │<──│ portfolio + risk   │
 │ (PBO, DSR, CPCV) —        │   │ (SMA,    │   │ (scores -> weights,│
 │  the honesty layer        │   │ next-open│   │  cost-aware, vol-  │
 │                           │   │  fills)  │   │  targeted)         │
 └──────────────────────────┘   └──────────┘   └────────────────────┘
```

There are two backtest paths that share the data and cost layers:

- **`engine.py`** — the simple long-only signal backtest the `backtest` CLI runs
  (e.g. SMA crossover). One weight per signaled name, next-open execution.
- **`portfolio.py`** — the research path: turn a cross-sectional *score* into
  quantile long/short weights and simulate them with drift and partial-adjustment
  rebalancing.

## Stage by stage

### `universe.py` — survivorship-bias-free membership

The trap: using today's S&P 500 as the universe for a 2010 backtest silently
deletes every company that has since been removed (usually for failing), which
flatters returns. `sp500_membership_panel(start, end)` reconstructs the index
day by day by replaying the Wikipedia changes table backward from today's
members. The output is a boolean date×ticker mask consumed by both backtest
paths. `sp600_*` mirrors this for small-caps (where the bias is worse).

### `data.py` — prices, calendar, cache

yfinance adjusted OHLCV (`auto_adjust=True`, so splits/dividends are
back-adjusted), aligned to the NYSE calendar (`XNYS`). Downloads are cached as
parquet keyed by a hash of the **ticker set** (not just its length) so two
different baskets of the same size can't collide. `splice_delistings` is the
mechanism for booking terminal returns of delisted names — it can't invent the
data yfinance lacks, but it lets a CRSP/Sharadar-quality delisting feed be applied
correctly.

### `fundamentals.py` — filing-date discipline

The single most important correctness property in the repo. SEC EDGAR's
companyfacts API stamps every datapoint with its **filing date**, and
`FactStore.annual_history(concepts, asof)` returns only facts with
`filed <= asof`. Using the fiscal-period-end date instead (which is public weeks
later) is the canonical lookahead bug. It also handles two real-world messes:
restatements (keep the latest-filed value per period) and XBRL tag drift (filers
switch concept names over time, so candidates are merged by priority). Derived
signals: gross profitability (Novy-Marx), asset growth, and standardized
unexpected earnings (SUE, the PEAD signal).

### `factors.py` — is the alpha just known factors?

Downloads the Ken French daily factors (Fama-French 5 + Momentum +
Short-term-Reversal) and regresses a strategy's excess returns on them.
Significance uses **Newey-West (HAC)** standard errors — financial residuals are
autocorrelated and heteroskedastic, and plain OLS errors understate uncertainty,
manufacturing false alpha. The output separates factor exposure (betas) from
residual alpha and its t-stat.

### `signals.py` + `bab.py` — the cross-sectional signal library

Each signal maps the universe to a per-name score at each monthly rebalance
(higher = more attractive): momentum (12-1), short-term reversal, low-volatility,
value (point-in-time book-to-market), profitability, asset growth, and PEAD.
`zscore_cross_section` standardizes a raw panel for combining; `quantile_spread`
is the sign check — the top-minus-bottom-quantile forward return tells you
whether a signal points the documented direction before you trust it. `bab.py`
is a standalone Betting-Against-Beta factor (Frazzini-Pedersen): rank-weighted,
levered to beta-neutral, with its own construction because it tilts continuously
across every name rather than slicing quantiles.

### `forecast.py` + `meta.py` — ML with leakage-safe CV

`forecast.py` stacks the signal panels into a `(date, ticker)` design matrix and
learns a non-linear map to next-period returns with a gradient-boosted tree,
against a Ridge baseline (so you can see whether the non-linearity earns its
keep). Two correctness details: labels are **cross-sectionally demeaned** per
date (the model learns relative ranking, not market timing), and folds are
**expanding walk-forward with a purge gap** so a forward-return label can't leak
across the train/test boundary. Skill is the **rank Information Coefficient** and
its t-stat across dates — a no-edge model scores IC ≈ 0.

`meta.py` is meta-labeling (López de Prado): the primary model picks the *side*,
and a secondary classifier learns *whether each bet will pay off*, conditioned
only on the names the primary actually bets. Its probability sizes the bet. It
reports precision/recall/F1/AUC against the base rate, on the same purged folds.

### `portfolio.py` + `risk.py` — scores to an executable book

`scores_to_weights` turns a score panel into equal-weight quantile portfolios —
dollar-neutral long-short (gross 1.0) or long-only. `backtest_portfolio` holds
those weights between monthly rebalances (letting them drift with returns) and,
on each rebalance, trades only a fraction of the way to the new target. This
**partial adjustment** (Garleanu-Pedersen) trims turnover — and thus the cost
drag that kills thin-alpha strategies — for little signal loss. It charges the
real `CostModel`, optionally per-name spreads and short-borrow financing, and
returns net *and* gross curves so the cost drag is visible. `risk.py` is a
volatility-targeting overlay (Moreira-Muir): scale exposure each rebalance by
trailing realized vol so forward vol tracks a target — which, per
Barroso-Santa-Clara, automatically de-levers into crashes.

### `engine.py` — the simple backtest loop

The CLI path. Equal-weights across the currently signaled set; the EOD signal on
day `T` sizes the position taken at the **open of day T+1** (the `shift(1)`).
Mark-to-market handles halts/gaps without vaporizing value or injecting spurious
returns, and the membership mask force-exits a name the day it leaves the index.

### `costs.py` — what trading actually costs

The default `CostModel` encodes the current US sell-side regulatory schedule
(SEC Section 31, FINRA TAF) plus a configurable commission and flat slippage.
`corwin_schultz_spread` estimates the bid-ask spread from daily high/low ranges
(point-in-time, days `t-1` and `t` only) so thin names can be charged more than
liquid ones.

### `metrics.py` + `walkforward.py` — the honesty layer

This is where a result is allowed to be believed or not.

- `summarize` — the usual CAGR / Sharpe / max-drawdown / Calmar.
- `walk_forward` — pick the best parameter set on each train window, evaluate on
  the next unseen window, stitch the out-of-sample slices into the only return
  stream untouched by selection. Anchored (expanding) or rolling.
- `combinatorial_purged_splits` / `combinatorial_walk_forward` — CPCV (López de
  Prado, ch. 12): instead of one OOS path, reconstruct *many* full-timeline paths
  with purge + embargo, giving a **distribution** of OOS Sharpe rather than a
  single point estimate.
- `probability_of_backtest_overfitting` — CSCV (Bailey et al.): the rate at which
  the in-sample-best strategy lands in the bottom half out-of-sample. ≈ 0.5 means
  in-sample ranking carries no OOS information (pure overfitting).
- `probabilistic_sharpe_ratio` / `deflated_sharpe_ratio` — P(true Sharpe > 0)
  adjusted for sample length, skew, and kurtosis; the deflated version raises the
  bar to the expected maximum Sharpe under the null across all trials searched.

## Cross-cutting conventions

- **Panels are `dates × tickers` DataFrames**; stacked design matrices use a
  `(date, ticker)` MultiIndex named exactly that.
- **Rebalances are monthly** (`monthly_rebalance_dates`); the engine path runs
  daily but only trades when the signal set changes.
- **Causality via `shift`/`asof`/trailing windows** — anything that looks at a
  rolling estimate lags it so date `t` never reads `t+1`.
- **No network in tests.** All HTTP (yfinance, EDGAR, Wikipedia, Ken French) is
  mocked; the 174-test suite runs offline in CI on Python 3.10–3.12.

## Extending

- **New signal:** return a `dates × tickers` panel, `zscore_cross_section` it,
  and confirm its sign with `quantile_spread` before adding it to a design matrix.
- **New strategy (engine path):** subclass `Strategy`, implement
  `generate_signals(prices) -> DataFrame` of 0/1 signals.
- **New cost regime:** construct `CostModel(...)` or pass a spread panel.
- **Before believing any result:** run it through `combinatorial_walk_forward`
  and `probability_of_backtest_overfitting`. A single Sharpe from a parameter
  sweep is not evidence.
