"""Price-data loader and NYSE trading calendar."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

NYSE = mcal.get_calendar("XNYS")

_OHLC_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def trading_days(start: str | date, end: str | date) -> pd.DatetimeIndex:
    """Valid NYSE trading days between start and end (inclusive)."""
    return NYSE.schedule(start_date=start, end_date=end).index


def load_ohlc(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download adjusted OHLCV data for `tickers` between `start` and `end`.

    With auto_adjust=True, all prices are back-adjusted for splits and dividends.

    Returns a dict mapping field name (Open/High/Low/Close/Volume) to a
    DataFrame indexed by date with tickers as columns. Tickers that yfinance
    returned no data for are dropped.
    """
    raw = _download_cached(tickers, start, end, cache_dir)

    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker download: columns are (field, ticker).
        fields = {f: raw[f] for f in raw.columns.get_level_values(0).unique() if f in _OHLC_FIELDS}
    else:
        # Single-ticker download: columns are just field names.
        ticker = tickers[0]
        fields = {
            f: raw[[f]].rename(columns={f: ticker})
            for f in raw.columns
            if f in _OHLC_FIELDS
        }

    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in fields.items():
        df = df.dropna(axis=1, how="all").sort_index()
        cleaned[name] = df
    return cleaned


def _download_cached(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | str | None,
) -> pd.DataFrame:
    if cache_dir is None:
        return _download(tickers, start, end)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Key on ticker identity, not just count: two distinct lists of equal length
    # over the same window must not collide on the same cache file.
    key = hashlib.sha1(",".join(sorted(set(tickers))).encode()).hexdigest()[:10]
    cache_path = cache_dir / f"ohlc_{start}_{end}_{len(tickers)}_{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = _download(tickers, start, end)
    raw.to_parquet(cache_path)
    return raw


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def splice_delistings(
    closes: pd.DataFrame,
    delistings: dict[str, tuple[str | date | pd.Timestamp, float]],
) -> pd.DataFrame:
    """Apply delisting returns so a price panel books terminal moves, not silence.

    yfinance returns only names that still trade today; the ones that went
    bankrupt or were acquired simply never appear -- survivorship bias that
    inflates every backtest. Given `delistings` mapping
    ``ticker -> (last_trading_date, delisting_return)`` (e.g. ``-1.0`` for a
    wipeout, or the cash-merger return), this realizes each terminal move on the
    first trading day *after* ``last_trading_date`` and NaNs the name thereafter,
    so a held position captures the delisting P&L and then exits. Names absent
    from `delistings` are untouched.

    This is the mechanism, not the data: populating `delistings` faithfully
    requires a delisting-inclusive source (CRSP, Sharadar). yfinance cannot
    provide it, so a survivor-only panel run through this function is unchanged.
    """
    out = closes.copy()
    for ticker, (last_date, delisting_return) in delistings.items():
        if ticker not in out.columns:
            continue
        last_date = pd.Timestamp(last_date)
        valid = out[ticker].loc[:last_date].dropna()
        after = out.index[out.index > last_date]
        if valid.empty:
            out[ticker] = np.nan          # the name's life ended before this window
            continue
        out.loc[after, ticker] = np.nan
        if len(after):
            out.loc[after[0], ticker] = float(valid.iloc[-1]) * (1.0 + delisting_return)
    return out
