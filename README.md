# us-equity-backtester

A small, vectorized backtester for US-equity strategies. Daily bars, long-only,
adjusted prices, NYSE trading calendar, and realistic regulatory fees.

Ships with one strategy out of the box: **SMA(50, 200) crossover, long-only**.

## What's "US-specialized" about it

- **NYSE calendar** via `pandas_market_calendars` — no weekends, no holidays.
- **Adjusted prices** from yfinance (`auto_adjust=True`) — splits and dividends
  are back-adjusted.
- **Sell-side regulatory fees** modeled by default:
  - SEC Section 31 fee: $27.80 per $1M of sale proceeds.
  - FINRA Trading Activity Fee: $0.000166 per share, capped at $9.27 per trade.
- **$0 commission** by default (Alpaca/Schwab/Robinhood retail). Configurable.
- **Universe loader** for the current S&P 500 constituents.

## Install

```bash
pip install -e ".[dev]"
```

## Use

```bash
# Default: S&P 500, 2015-2025, SMA(50, 200), $100k starting capital
backtest

# Single ticker
backtest --tickers SPY --start 2010-01-01 --end 2025-01-01

# Custom basket
backtest --tickers AAPL MSFT NVDA GOOGL --start 2018-01-01

# Export equity curve
backtest --tickers SPY --output equity.csv
```

From Python:

```python
from equity_backtester import SMACrossover, CostModel, run_backtest, summarize
from equity_backtester.data import load_ohlc

ohlc = load_ohlc(["SPY"], "2015-01-01", "2025-01-01", cache_dir=".cache")
result = run_backtest(
    closes=ohlc["Close"],
    opens=ohlc["Open"],
    strategy=SMACrossover(fast=50, slow=200),
    cost_model=CostModel(),
)
print(summarize(result.equity_curve))
```

## Point-in-time membership (avoid survivorship bias)

By default, `backtest` uses today's S&P 500 list as the universe — implicitly
assuming you knew the future winners. Pass `--point-in-time` to reconstruct
historical membership instead:

```bash
backtest --point-in-time --start 2010-01-01
```

This fetches the Wikipedia changes table, builds a date×ticker membership
panel, and constrains the strategy to only signal names that were actually
in the index on each date. Positions are force-exited when a name leaves
the index.

Programmatic use:

```python
from equity_backtester import run_backtest, sp500_membership_panel
from equity_backtester.data import load_ohlc

panel = sp500_membership_panel("2010-01-01", "2025-01-01")
ohlc = load_ohlc(list(panel.columns), "2010-01-01", "2025-01-01", cache_dir=".cache")
result = run_backtest(
    closes=ohlc["Close"].dropna(axis=1, how="all").ffill(),
    opens=ohlc["Open"].dropna(axis=1, how="all").ffill(),
    strategy=...,
    membership_mask=panel,
)
```

Caveats specific to this mode:
- Wikipedia's changes table is reliable from ~2000 onward; older history is patchy.
- yfinance doesn't always carry delisted tickers; missing names are silently dropped.
- The mask snaps to the change's "effective date" — we don't model S&P's
  pre-announcement window (typically ~5 business days).

## Execution model

- Signals are computed using close-of-day data.
- Trades execute at the **next session's open** — no look-ahead bias.
- Portfolio is equal-weighted across the active signaled set, rebalanced when
  the signal set changes.
- Fractional shares allowed; no margin, no shorts.

## Caveats

- **Survivorship bias.** Default mode uses today's S&P 500 list; pass
  `--point-in-time` to reconstruct historical membership (see the section above).
- **yfinance data quality.** Fine for research-grade backtests; not adequate
  for production. Watch for occasional missing bars.
- **No margin / short / borrow modeling.** Long-only by design.
- **No tax modeling.** Wash sales, short-term vs long-term capital gains,
  Section 1256 — all ignored.
- **No realistic execution.** Open-fill assumption ignores liquidity, market
  impact, and the difference between official-open and your tradeable price.

## Layout

```
src/equity_backtester/
  universe.py   — S&P 500 ticker scrape
  data.py       — yfinance loader + NYSE calendar, parquet-cached
  strategy.py   — Strategy ABC and SMACrossover
  costs.py      — US-equity transaction-cost model
  engine.py     — vectorized backtest loop
  metrics.py    — Sharpe, drawdown, CAGR, Calmar
  cli.py        — `backtest` entry point
tests/          — pytest suite (no network calls)
.github/        — CI: lint + tests on 3.10/3.11/3.12
```

## Extending

Add a new strategy by subclassing `Strategy` and implementing
`generate_signals(prices) -> DataFrame` returning 0/1 (or fractional weights
if you want non-equal allocation; the engine will renormalize across the
signaled set).

To swap brokers, construct `CostModel(commission_per_share=..., slippage_bps=...)`
and pass it into `run_backtest`.
