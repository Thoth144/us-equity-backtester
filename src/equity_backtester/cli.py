"""Command-line entry point: download data, run the SMA backtest, print metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .costs import CostModel
from .data import load_ohlc
from .engine import run_backtest
from .metrics import summarize
from .strategy import SMACrossover
from .universe import sp500_membership_panel, sp500_tickers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="US equities SMA-crossover backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2025-01-01", help="End date (YYYY-MM-DD, exclusive).")
    parser.add_argument("--fast", type=int, default=50, help="Fast SMA window.")
    parser.add_argument("--slow", type=int, default=200, help="Slow SMA window.")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Tickers to backtest. Defaults to the current S&P 500 (scraped from Wikipedia).",
    )
    parser.add_argument(
        "--point-in-time",
        action="store_true",
        help="Use point-in-time S&P 500 membership (survivorship-bias-free). "
             "Ignored when --tickers is given.",
    )
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting capital.")
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory for caching downloaded price data. Pass an empty string to disable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the equity curve as CSV.",
    )
    args = parser.parse_args()

    tickers, membership_mask = _resolve_universe(args)
    cache = Path(args.cache_dir) if args.cache_dir else None

    print(f"Loading prices for {len(tickers)} tickers from {args.start} to {args.end}...")
    ohlc = load_ohlc(tickers, args.start, args.end, cache_dir=cache)
    opens = _clean(ohlc["Open"])
    closes = _clean(ohlc["Close"])
    common = opens.columns.intersection(closes.columns)
    opens, closes = opens[common], closes[common]

    print(f"Running SMA({args.fast}, {args.slow}) crossover on {len(common)} tickers...")
    result = run_backtest(
        closes=closes,
        opens=opens,
        strategy=SMACrossover(fast=args.fast, slow=args.slow),
        cost_model=CostModel(),
        starting_cash=args.cash,
        membership_mask=membership_mask,
    )

    summary = summarize(result.equity_curve)
    _print_report(args, len(common), result, summary)

    if args.output:
        pd.DataFrame({"equity": result.equity_curve}).to_csv(args.output, index_label="date")
        print(f"\nEquity curve written to {args.output}")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-NaN columns; forward-fill brief gaps (rare suspensions)."""
    return df.dropna(axis=1, how="all").ffill()


def _resolve_universe(args) -> tuple[list[str], pd.DataFrame | None]:
    """Return (tickers, membership_mask). Mask is None unless --point-in-time."""
    if args.tickers:
        return args.tickers, None
    if args.point_in_time:
        print("Fetching historical S&P 500 membership from Wikipedia...")
        panel = sp500_membership_panel(args.start, args.end)
        return list(panel.columns), panel
    return sp500_tickers(), None


def _print_report(args, n_tickers, result, summary) -> None:
    total_cost = float(result.costs.to_numpy().sum())
    final_equity = float(result.equity_curve.iloc[-1])

    print()
    print(f"Strategy:        SMA({args.fast}, {args.slow}) crossover, long-only")
    print(f"Universe:        {n_tickers} tickers")
    print(f"Period:          {args.start} to {args.end} ({summary.n_days} trading days)")
    print()
    print(f"Final equity:    ${final_equity:,.2f}")
    print(f"Total return:    {summary.total_return * 100:7.2f}%")
    print(f"CAGR:            {summary.cagr * 100:7.2f}%")
    print(f"Annual vol:      {summary.annual_volatility * 100:7.2f}%")
    print(f"Sharpe ratio:    {summary.sharpe:7.2f}")
    print(f"Max drawdown:    {summary.max_drawdown * 100:7.2f}%")
    print(f"Calmar ratio:    {summary.calmar:7.2f}")
    print()
    print(f"Total cost paid: ${total_cost:,.2f}")


if __name__ == "__main__":
    main()
