"""Cross-sectional signal tests — synthetic data, no network."""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.fundamentals import FactStore, shares_outstanding
from equity_backtester.signals import (
    asset_growth_signal,
    earnings_surprise_signal,
    forward_returns,
    low_vol_signal,
    momentum_signal,
    monthly_rebalance_dates,
    profitability_signal,
    quantile_spread,
    reversal_signal,
    value_signal,
    zscore_cross_section,
)


def _store(rows, shares_rows=None):
    df = pd.DataFrame(rows, columns=["concept", "fp", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["form"] = "10-K"
    sh = pd.DataFrame(shares_rows or [], columns=["end", "filed", "val"])
    if not sh.empty:
        sh["end"] = pd.to_datetime(sh["end"])
        sh["filed"] = pd.to_datetime(sh["filed"])
    return FactStore(ticker="X", cik="0", facts=df, shares=sh)


# --- framework --------------------------------------------------------------

def test_zscore_makes_rows_mean0_std1():
    p = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]},
                     index=pd.bdate_range("2020-01-01", periods=2))
    z = zscore_cross_section(p)
    assert z.mean(axis=1).abs().max() < 1e-9
    assert (z.std(axis=1) - 1.0).abs().max() < 1e-9


def test_zscore_constant_row_is_nan_not_crash():
    p = pd.DataFrame({"A": [5.0], "B": [5.0]}, index=pd.bdate_range("2020-01-01", periods=1))
    assert zscore_cross_section(p).isna().all().all()


def test_forward_returns_to_next_rebalance():
    dates = pd.bdate_range("2020-01-01", periods=4)
    closes = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=dates)
    f = forward_returns(closes, dates)
    assert f["A"].iloc[0] == pytest.approx(0.1)
    assert f["A"].iloc[1] == pytest.approx(0.1)
    assert pd.isna(f["A"].iloc[-1])


def test_quantile_spread_positive_when_signal_predicts():
    dates = pd.bdate_range("2020-01-01", periods=3)
    tickers = [f"T{i}" for i in range(10)]
    rng = np.random.default_rng(0)
    fwd = pd.DataFrame(rng.normal(0, 0.05, (3, 10)), index=dates, columns=tickers)
    assert (quantile_spread(fwd, fwd, quantiles=5) > 0).all()      # signal == outcome
    assert (quantile_spread(-fwd, fwd, quantiles=5) < 0).all()     # anti-correlated


def test_monthly_rebalance_dates_are_month_ends():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")
    closes = pd.DataFrame({"A": range(len(dates))}, index=dates)
    rb = monthly_rebalance_dates(closes)
    assert len(rb) == 3
    assert rb[0] == dates[dates.month == 1][-1]


# --- price signals ----------------------------------------------------------

def test_momentum_ranks_winners_above_losers():
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = pd.DataFrame({"WIN": np.linspace(100, 300, n),
                           "LOSE": np.linspace(300, 100, n)}, index=dates)
    m = momentum_signal(closes, pd.DatetimeIndex([dates[-1]]))
    assert m.loc[dates[-1], "WIN"] > m.loc[dates[-1], "LOSE"]


def test_reversal_favors_recent_losers():
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = pd.DataFrame({"UP": np.linspace(100, 130, n),
                           "DOWN": np.linspace(130, 100, n)}, index=dates)
    r = reversal_signal(closes, pd.DatetimeIndex([dates[-1]]), window=21)
    assert r.loc[dates[-1], "DOWN"] > r.loc[dates[-1], "UP"]


def test_low_vol_favors_calm_names():
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(1)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.005, n))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.03, n))
    closes = pd.DataFrame({"CALM": calm, "WILD": wild}, index=dates)
    lv = low_vol_signal(closes, pd.DatetimeIndex([dates[-1]]), window=126)
    assert lv.loc[dates[-1], "CALM"] > lv.loc[dates[-1], "WILD"]


# --- fundamental signals ----------------------------------------------------

def test_profitability_signal_value_and_point_in_time():
    s = _store([
        ("Revenues", "FY", "2020-12-31", "2021-02-15", 1000),
        ("CostOfRevenue", "FY", "2020-12-31", "2021-02-15", 600),
        ("Assets", "FY", "2020-12-31", "2021-02-15", 2000),
    ])
    dates = pd.DatetimeIndex(["2021-06-01", "2021-01-01"])
    p = profitability_signal({"X": s}, dates)
    assert p.loc["2021-06-01", "X"] == pytest.approx(0.2)
    assert pd.isna(p.loc["2021-01-01", "X"])  # nothing filed yet


def test_asset_growth_signal_is_negated():
    s = _store([
        ("Assets", "FY", "2019-12-31", "2020-02-15", 100),
        ("Assets", "FY", "2020-12-31", "2021-02-15", 130),
    ])
    p = asset_growth_signal({"X": s}, pd.DatetimeIndex(["2021-06-01"]))
    assert p.loc["2021-06-01", "X"] == pytest.approx(-0.30)  # +30% growth -> -0.30 score


def test_value_signal_book_to_market():
    s = _store(
        [("StockholdersEquity", "FY", "2020-12-31", "2021-02-15", 1000)],
        shares_rows=[("2020-12-31", "2021-02-15", 100)],
    )
    closes = pd.DataFrame({"X": [5.0]}, index=pd.DatetimeIndex(["2021-06-01"]))
    p = value_signal({"X": s}, closes, pd.DatetimeIndex(["2021-06-01"]))
    # book 1000 / (100 shares * $5) = 1000/500 = 2.0
    assert p.loc["2021-06-01", "X"] == pytest.approx(2.0)


def test_shares_outstanding_point_in_time():
    s = _store(
        [("Assets", "FY", "2020-12-31", "2021-02-15", 1)],
        shares_rows=[("2020-12-31", "2021-02-15", 100), ("2021-12-31", "2022-02-15", 110)],
    )
    assert shares_outstanding(s, "2021-06-01") == 100
    assert shares_outstanding(s, "2022-06-01") == 110
    assert shares_outstanding(s, "2021-01-01") is None


def _earnings_store(values_by_end):
    """A FactStore of discrete (~90-day) quarterly NetIncomeLoss from {end: value}."""
    rows = []
    for end, val in values_by_end.items():
        e = pd.Timestamp(end)
        rows.append(("NetIncomeLoss", (e - pd.Timedelta(days=90)).date().isoformat(),
                     end, (e + pd.Timedelta(days=40)).date().isoformat(), val))
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "filed", "val"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["fp"], df["form"] = "Q", "10-Q"
    return FactStore(ticker="X", cik="0", facts=df)


def test_earnings_surprise_signal_ranks_upside_above_downside():
    base = {"2019-03-31": 100, "2019-06-30": 100, "2019-09-30": 100, "2019-12-31": 100,
            "2020-03-31": 108, "2020-06-30": 112, "2020-09-30": 109, "2020-12-31": 111}
    hi = _earnings_store({**base, "2021-03-31": 200})  # large upside surprise
    lo = _earnings_store({**base, "2021-03-31": 40})   # large downside surprise
    panel = earnings_surprise_signal({"HI": hi, "LO": lo}, pd.DatetimeIndex(["2021-07-01"]))
    assert panel.loc["2021-07-01", "HI"] > panel.loc["2021-07-01", "LO"]
    # Point-in-time: before the Q1-2021 filing there isn't enough history -> NaN.
    early = earnings_surprise_signal({"HI": hi}, pd.DatetimeIndex(["2021-04-01"]))
    assert pd.isna(early.loc["2021-04-01", "HI"])
