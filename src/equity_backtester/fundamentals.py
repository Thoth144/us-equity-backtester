"""Point-in-time fundamentals from SEC EDGAR.

The whole point of this module is the filing-date discipline. EDGAR's
companyfacts API stamps every datapoint with the date it was actually filed,
so `FactStore.annual_history(concepts, asof)` returns only what a trader could
have known on `asof` — never the fiscal-period-end value that wasn't public
until weeks later. Using period-end dates instead of filing dates is the #1
silent lookahead bug in fundamental backtests, and it inflates returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
# SEC requires a descriptive User-Agent with contact info, else it returns 403.
_SEC_UA = "us-equity-backtester/0.1 research (contact: research@example.com)"

# XBRL concept names vary across filers/years; try candidates in priority order.
# Prefer the modern ASC 606 contract-revenue tag: many filers ALSO carry a legacy
# `Revenues`/`SalesRevenueNet` tag holding a partial or stale figure (e.g. Apple's
# `Revenues` stops at FY2018 with a segment-level value), so those must rank last.
_REVENUE = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
_COGS = ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"]
_ASSETS = ["Assets"]
_NET_INCOME = ["NetIncomeLoss"]


@dataclass
class FactStore:
    ticker: str
    cik: str
    facts: pd.DataFrame  # columns: concept, fp, start, end, filed, val, form
    shares: pd.DataFrame = field(  # columns: end, filed, val (common shares outstanding)
        default_factory=lambda: pd.DataFrame(columns=["end", "filed", "val"])
    )

    def annual_history(self, concepts: list[str], asof) -> pd.Series:
        """Annual (fp=='FY') values, merged across candidate concepts, point-in-time.

        Only facts with filed <= asof are visible. For each fiscal-year-end we
        keep the latest-filed value (so restatements resolve to what was known
        on `asof`). Candidate concepts are merged per fiscal year, preferring
        higher-priority ones: this matters because filers switch XBRL tags over
        time (e.g. MSFT moved CostOfRevenue -> CostOfGoodsAndServicesSold after
        FY2017), so picking a single "first available" concept would silently
        return a stale tag's history. Returns a Series indexed by fiscal-year-end,
        sorted ascending; empty if no candidate has data as of that date.
        """
        asof = pd.Timestamp(asof)
        merged = pd.Series(dtype=float)
        for concept in concepts:
            sub = self.facts[
                (self.facts["concept"] == concept)
                & (self.facts["fp"] == "FY")
                & (self.facts["filed"] <= asof)
            ]
            if sub.empty:
                continue
            sub = sub.sort_values("filed").drop_duplicates("end", keep="last")
            series = sub.set_index("end")["val"].astype(float)
            merged = merged.combine_first(series)  # keep higher-priority years, fill gaps
        return merged.sort_index()

    def quarterly_history(self, concepts: list[str], asof) -> pd.Series:
        """Discrete fiscal-quarter values (~3-month duration), point-in-time.

        EDGAR reports income-statement items in several durations per period — a
        Q2 NetIncomeLoss row can be the 3-month figure OR the 6-month year-to-date
        figure, sharing the same period-end — so the fiscal-period label alone is
        ambiguous. We keep only ~quarter-length durations (the discrete quarter),
        take the latest-filed value per quarter-end (restatement discipline, as in
        `annual_history`), and merge candidate concepts by priority. Q4 is usually
        not filed as a standalone quarter (only the FY 10-K 12-month figure), so it
        is absent here. Returns a Series indexed by quarter-end, sorted ascending.
        """
        asof = pd.Timestamp(asof)
        if "start" not in self.facts.columns:
            return pd.Series(dtype=float)
        duration = (self.facts["end"] - self.facts["start"]).dt.days
        is_quarter = duration.between(80, 100)
        merged = pd.Series(dtype=float)
        for concept in concepts:
            sub = self.facts[
                is_quarter
                & (self.facts["concept"] == concept)
                & (self.facts["filed"] <= asof)
            ]
            if sub.empty:
                continue
            sub = sub.sort_values("filed").drop_duplicates("end", keep="last")
            series = sub.set_index("end")["val"].astype(float)
            merged = merged.combine_first(series)
        return merged.sort_index()


def gross_profitability(store: FactStore, asof) -> float | None:
    """Novy-Marx gross profitability: (Revenue - COGS) / Total Assets.

    Aligns all three inputs on the latest fiscal year for which every input is
    available as of `asof`. Returns None if the data isn't there.
    """
    rev = store.annual_history(_REVENUE, asof)
    cogs = store.annual_history(_COGS, asof)
    assets = store.annual_history(_ASSETS, asof)
    common = rev.index.intersection(cogs.index).intersection(assets.index)
    if len(common) == 0:
        return None
    end = common.max()
    if assets[end] == 0:
        return None
    return float((rev[end] - cogs[end]) / assets[end])


def asset_growth(store: FactStore, asof) -> float | None:
    """Year-over-year growth in Total Assets, point-in-time. None if < 2 years."""
    assets = store.annual_history(_ASSETS, asof)
    if len(assets) < 2 or assets.iloc[-2] == 0:
        return None
    return float(assets.iloc[-1] / assets.iloc[-2] - 1.0)


def standardized_unexpected_earnings(
    store: FactStore, asof, *, min_history: int = 5
) -> float | None:
    """Standardized unexpected earnings (SUE), point-in-time — the PEAD signal.

    Seasonal-random-walk-with-drift surprise (Foster-Olsen-Shevlin 1984; the basis
    of Bernard-Thomas 1989 post-earnings-announcement drift): quarterly net income
    is differenced against the same quarter a year earlier (so seasonality cancels),
    then standardized by the firm's own history of those changes:

        SUE = (latest YoY change - mean(prior YoY changes)) / std(prior YoY changes)

    Uses only data filed on or before `asof`. Returns None if fewer than
    `min_history` seasonal differences are available or the history has no
    dispersion (a constant-growth firm scores neutral, not infinite).
    """
    diffs = _seasonal_diffs(store.quarterly_history(_NET_INCOME, asof))
    if len(diffs) < min_history:
        return None
    prior = diffs.iloc[:-1]
    sd = prior.std(ddof=1)
    if not sd > 0:
        return None
    return float((diffs.iloc[-1] - prior.mean()) / sd)


def _seasonal_diffs(quarterly: pd.Series) -> pd.Series:
    """Year-over-year change in a quarterly series, matching each quarter to the one
    ending ~1 year earlier (±20 days). Indexed by quarter-end; robust to a missing
    quarter (e.g. the absent discrete Q4), which positional lag-4 would not be.
    """
    out = {}
    idx = quarterly.index
    for end in idx:
        target = end - pd.Timedelta(days=365)
        window = idx[(idx >= target - pd.Timedelta(days=20))
                     & (idx <= target + pd.Timedelta(days=20))]
        if len(window) == 0:
            continue
        prev = min(window, key=lambda d: abs((d - target).days))
        out[end] = float(quarterly.loc[end] - quarterly.loc[prev])
    return pd.Series(out).sort_index()


def shares_outstanding(store: FactStore, asof) -> float | None:
    """Most recent common-shares-outstanding count filed on or before `asof`."""
    asof = pd.Timestamp(asof)
    sub = store.shares[store.shares["filed"] <= asof]
    if sub.empty:
        return None
    return float(sub.sort_values(["end", "filed"]).iloc[-1]["val"])


def cik_for_ticker(ticker: str, cache_dir: Path | str | None = None) -> str:
    """Zero-padded 10-digit CIK for a ticker, via the SEC ticker map."""
    data = _http_json(_TICKERS_URL, _cache_path(cache_dir, "company_tickers.json"))
    target = ticker.upper()
    for row in data.values():
        if str(row["ticker"]).upper() == target:
            return f"{int(row['cik_str']):010d}"
    raise ValueError(f"ticker {ticker!r} not found in SEC ticker map")


def load_company_facts(ticker: str, cache_dir: Path | str | None = None) -> FactStore:
    """Fetch and normalize SEC companyfacts for a ticker."""
    cik = cik_for_ticker(ticker, cache_dir)
    payload = _http_json(_FACTS_URL.format(cik=cik), _cache_path(cache_dir, f"CIK{cik}.json"))
    return FactStore(
        ticker=ticker.upper(), cik=cik,
        facts=_normalize_facts(payload), shares=_extract_shares(payload),
    )


def _normalize_facts(payload: dict) -> pd.DataFrame:
    rows = []
    gaap = payload.get("facts", {}).get("us-gaap", {})
    for concept, body in gaap.items():
        for item in body.get("units", {}).get("USD", []):
            if "filed" in item and "end" in item and "val" in item:
                rows.append((concept, item.get("fp"), item.get("start"), item["end"],
                             item["filed"], item["val"], item.get("form")))
    df = pd.DataFrame(rows, columns=["concept", "fp", "start", "end", "filed", "val", "form"])
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    return df


def _extract_shares(payload: dict) -> pd.DataFrame:
    """Common shares outstanding from the dei cover page or us-gaap balance sheet."""
    facts = payload.get("facts", {})
    rows = []
    for ns, concept in (("dei", "EntityCommonStockSharesOutstanding"),
                        ("us-gaap", "CommonStockSharesOutstanding")):
        body = facts.get(ns, {}).get(concept, {})
        for item in body.get("units", {}).get("shares", []):
            if "filed" in item and "end" in item and "val" in item:
                rows.append((item["end"], item["filed"], item["val"]))
    df = pd.DataFrame(rows, columns=["end", "filed", "val"])
    if not df.empty:
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
    return df


def _cache_path(cache_dir: Path | str | None, name: str) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / "edgar" / name


def _http_json(url: str, cache_path: Path | None) -> dict:
    if cache_path is not None and cache_path.exists():
        with open(cache_path) as fh:
            return json.load(fh)
    resp = requests.get(url, headers={"User-Agent": _SEC_UA}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(payload, fh)
    return payload
