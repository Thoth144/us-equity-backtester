"""Cross-sectional signal library.

Each signal maps the universe to a per-name score at each rebalance date
(higher = more attractive). Price signals need only a close-price panel;
fundamental signals query point-in-time FactStores (T3). Signals return RAW
panels (dates x tickers); `zscore_cross_section` standardizes them for
combining, and `quantile_spread` computes the top-minus-bottom-quantile forward
return used to sanity-check a signal's sign.

References: Jegadeesh-Titman (momentum), Nagel 2012 (short-term reversal),
Frazzini-Pedersen 2014 (low risk), Novy-Marx 2013 (gross profitability),
Fama-French (value/HML and CMA/asset growth), Bernard-Thomas 1989 (PEAD).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import (
    asset_growth,
    gross_profitability,
    shares_outstanding,
    standardized_unexpected_earnings,
)

_BOOK_EQUITY = [
    "StockholdersEquity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]


# --- framework --------------------------------------------------------------

def monthly_rebalance_dates(closes: pd.DataFrame) -> pd.DatetimeIndex:
    """Last trading day of each month in the price index."""
    idx = closes.index
    return pd.DatetimeIndex(idx.to_series().groupby(idx.to_period("M")).last().values)


def zscore_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    """Standardize each row (date) across tickers to mean 0, std 1 (NaNs ignored)."""
    mu = panel.mean(axis=1)
    sigma = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(mu, axis=0).div(sigma, axis=0)


def forward_returns(closes: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return from each rebalance date to the next, per ticker (last row NaN)."""
    px = closes.reindex(dates, method="ffill")
    return px.shift(-1) / px - 1.0


def quantile_spread(signal: pd.DataFrame, fwd: pd.DataFrame, quantiles: int = 10) -> pd.Series:
    """Top-minus-bottom-quantile forward return at each date.

    A positive mean means the signal ranks names in the documented direction.
    """
    out = {}
    for date in signal.index:
        if date not in fwd.index:
            continue
        s = signal.loc[date].dropna()
        f = fwd.loc[date].dropna()
        common = s.index.intersection(f.index)
        if len(common) < quantiles:
            continue
        s, f = s[common], f[common]
        try:
            buckets = pd.qcut(s, quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        out[date] = f[buckets == buckets.max()].mean() - f[buckets == buckets.min()].mean()
    return pd.Series(out, name="spread")


# --- price signals ----------------------------------------------------------

def momentum_signal(closes, dates, lookback=252, skip=21):
    """12-1 momentum: trailing ~12-month return excluding the most recent month."""
    mom = closes.shift(skip) / closes.shift(lookback) - 1.0
    return mom.reindex(dates, method="ffill")


def reversal_signal(closes, dates, window=21):
    """Short-term reversal: negative of the most recent ~1-month return."""
    rev = -(closes / closes.shift(window) - 1.0)
    return rev.reindex(dates, method="ffill")


def low_vol_signal(closes, dates, window=126):
    """Low volatility: negative trailing realized vol (calm names score high)."""
    vol = closes.pct_change(fill_method=None).rolling(window).std()
    return (-vol).reindex(dates, method="ffill")


# --- fundamental signals ----------------------------------------------------

def profitability_signal(stores, dates):
    """Novy-Marx gross profitability, point-in-time."""
    return _fundamental_panel(stores, dates, gross_profitability)


def asset_growth_signal(stores, dates):
    """Asset growth, negated (low investment predicts higher returns)."""
    return _fundamental_panel(stores, dates, lambda s, d: _neg(asset_growth(s, d)))


def value_signal(stores, closes, dates):
    """Book-to-market: point-in-time book equity / market cap (shares x price)."""
    panel = pd.DataFrame(index=dates, columns=list(stores), dtype=float)
    for ticker, store in stores.items():
        if ticker not in closes.columns:
            continue
        prices = closes[ticker]
        for date in dates:
            book = store.annual_history(_BOOK_EQUITY, date)
            sh = shares_outstanding(store, date)
            if book.empty or not sh:
                continue
            price = prices.asof(date)
            if pd.isna(price) or price <= 0:
                continue
            mcap = sh * price
            if mcap > 0:
                panel.loc[date, ticker] = float(book.iloc[-1]) / mcap
    return panel


def earnings_surprise_signal(stores, dates):
    """Post-earnings-announcement drift (PEAD): standardized unexpected earnings,
    point-in-time. Positive surprises drift up, so higher SUE scores higher
    (Bernard-Thomas 1989)."""
    return _fundamental_panel(stores, dates, standardized_unexpected_earnings)


def _fundamental_panel(stores, dates, fn):
    panel = pd.DataFrame(index=dates, columns=list(stores), dtype=float)
    for ticker, store in stores.items():
        for date in dates:
            value = fn(store, date)
            if value is not None:
                panel.loc[date, ticker] = value
    return panel


def _neg(x):
    return None if x is None else -x
