"""S&P 500 universe loader — current constituents and point-in-time membership.

`sp500_tickers()` returns today's S&P 500. Using it as the universe for a
historical backtest introduces survivorship bias: companies removed from the
index over time are missing, biasing returns upward.

For honest historical work, use `sp500_membership_panel(start, end)`, which
reconstructs membership on each trading day from the Wikipedia changes table.
Pass the result as the `membership_mask` argument to `run_backtest`.
"""

from __future__ import annotations

from datetime import date, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .data import load_ohlc, trading_days

_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_SP600_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
_USER_AGENT = "us-equity-backtester/0.1 (educational backtesting tool)"


def sp500_tickers() -> list[str]:
    """Scrape the current S&P 500 ticker list from Wikipedia.

    Yahoo Finance uses '-' instead of '.' for class shares (e.g. BRK.B -> BRK-B),
    so we normalize symbols to the Yahoo convention.
    """
    tables = _fetch_wiki_tables()
    symbols = tables[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-") for s in symbols]


def sp500_ohlc(
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Adjusted OHLCV for the *current* S&P 500 constituents over [start, end].

    Convenience wrapper: today's members from `sp500_tickers()` handed to
    `load_ohlc`. Applying current membership to history is survivorship-biased —
    names that have since left the index are absent, which flatters returns. For
    bias-free historical work build `sp500_membership_panel` instead and pass it
    as `membership_mask` to `run_backtest`. Returns the same field -> panel dict
    as `load_ohlc`.
    """
    return load_ohlc(sp500_tickers(), start, end, cache_dir=cache_dir)


def sp600_tickers() -> list[str]:
    """Scrape the current S&P SmallCap 600 ticker list from Wikipedia.

    Symbols are normalized to the Yahoo convention ('.' -> '-'), as in
    `sp500_tickers`. Note: the S&P 600 applies an earnings/liquidity inclusion
    screen, so it is *quality-filtered* small-cap, not the raw small-cap
    universe — which is part of why it has historically beaten the Russell 2000.
    """
    tables = _fetch_wiki_tables(_SP600_WIKI_URL)
    symbols = tables[0]["Symbol"].astype(str).tolist()
    return [s.replace(".", "-") for s in symbols]


def sp600_ohlc(
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Adjusted OHLCV for the *current* S&P SmallCap 600 constituents.

    Same survivorship caveat as `sp500_ohlc`, but stronger: small-caps delist
    far more often (buyouts, bankruptcies, falling below listing standards), so
    applying today's members to history omits many failures and flatters
    returns. Returns the same field -> panel dict as `load_ohlc`.
    """
    return load_ohlc(sp600_tickers(), start, end, cache_dir=cache_dir)


def sp500_changes() -> pd.DataFrame:
    """Historical S&P 500 membership changes from Wikipedia.

    Returns a DataFrame with columns:
      date              — effective date of the change.
      added_ticker      — ticker added on that date (or NaN).
      added_security    — company name added (or NaN).
      removed_ticker    — ticker removed on that date (or NaN).
      removed_security  — company name removed (or NaN).

    Tickers use the Yahoo convention ('.' -> '-').
    """
    tables = _fetch_wiki_tables()
    return _parse_changes_table(tables[1])


def _fetch_wiki_tables(url: str = _WIKI_URL) -> list[pd.DataFrame]:
    """Fetch a Wikipedia page with a real User-Agent and parse all its tables.

    Wikipedia returns 403 for the default urllib User-Agent that pd.read_html
    uses when given a URL, so we fetch the HTML ourselves first.
    """
    response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def sp500_members_at(
    target_date: str | date | datetime,
    current_members: list[str] | None = None,
    changes: pd.DataFrame | None = None,
) -> set[str]:
    """Reconstruct the S&P 500 membership in effect on `target_date`.

    Starts from today's members and undoes every change with an effective date
    strictly after `target_date`. Pass `current_members` and `changes`
    explicitly to avoid re-fetching from Wikipedia.
    """
    if current_members is None:
        current_members = sp500_tickers()
    if changes is None:
        changes = sp500_changes()

    target = pd.to_datetime(target_date)
    members = set(current_members)
    future = changes[changes["date"] > target]
    for row in future.itertuples(index=False):
        if pd.notna(row.added_ticker):
            members.discard(row.added_ticker)
        if pd.notna(row.removed_ticker):
            members.add(row.removed_ticker)
    return members


def sp500_membership_panel(
    start: str | date,
    end: str | date,
    current_members: list[str] | None = None,
    changes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Boolean membership panel for each NYSE trading day in [start, end].

    Index: NYSE trading days.
    Columns: union of every ticker that was an index member at any point in
      the period (sorted).
    Cells: True iff the ticker was an S&P 500 member on that day.

    Pass the columns as the universe for `load_ohlc`, and pass the panel as
    `membership_mask` to `run_backtest`.
    """
    if current_members is None:
        current_members = sp500_tickers()
    if changes is None:
        changes = sp500_changes()

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    dates = trading_days(start, end)

    current = sp500_members_at(start_ts, current_members, changes)
    in_range = (changes[(changes["date"] > start_ts) & (changes["date"] <= end_ts)]
                .sort_values("date"))
    change_records = list(in_range.itertuples(index=False))

    all_tickers: set[str] = set(current)
    for c in change_records:
        if pd.notna(c.added_ticker):
            all_tickers.add(c.added_ticker)
        if pd.notna(c.removed_ticker):
            all_tickers.add(c.removed_ticker)
    tickers = sorted(all_tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    panel = np.zeros((len(dates), len(tickers)), dtype=bool)
    ci = 0
    for di, day in enumerate(dates):
        while ci < len(change_records) and change_records[ci].date <= day:
            c = change_records[ci]
            if pd.notna(c.added_ticker):
                current.add(c.added_ticker)
            if pd.notna(c.removed_ticker):
                current.discard(c.removed_ticker)
            ci += 1
        for t in current:
            panel[di, ticker_idx[t]] = True

    return pd.DataFrame(panel, index=dates, columns=tickers)


def _parse_changes_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Wikipedia changes table to a flat 5-column schema."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(a) if a == b else f"{a}_{b}"
            for a, b in df.columns
        ]
    df = df.iloc[:, :5]
    df.columns = ["date", "added_ticker", "added_security",
                  "removed_ticker", "removed_security"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    for col in ("added_ticker", "removed_ticker"):
        df[col] = (df[col].astype(str)
                   .str.replace(".", "-", regex=False)
                   .replace({"nan": None, "NaN": None, "": None, "<NA>": None}))
    return df
