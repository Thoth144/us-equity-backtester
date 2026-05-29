"""Point-in-time fundamentals tests — synthetic FactStores, no network.

These exercise the one thing that matters most: filing-date discipline.
"""

import pandas as pd
import pytest

from equity_backtester.fundamentals import (
    FactStore,
    asset_growth,
    gross_profitability,
    standardized_unexpected_earnings,
)


def _store(rows):
    """rows: list of (concept, fp, end, filed, val)."""
    df = pd.DataFrame(rows, columns=["concept", "fp", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["form"] = "10-K"
    return FactStore(ticker="TEST", cik="0", facts=df)


def test_annual_history_excludes_unfiled_facts():
    s = _store([
        ("Assets", "FY", "2020-12-31", "2021-02-15", 100),
        ("Assets", "FY", "2021-12-31", "2022-02-15", 120),
    ])
    # As of 2021-06-01 only FY2020 has been filed.
    h = s.annual_history(["Assets"], "2021-06-01")
    assert list(h.index.year) == [2020]
    assert h.iloc[-1] == 100
    # As of 2022-06-01 both are filed.
    h2 = s.annual_history(["Assets"], "2022-06-01")
    assert list(h2.index.year) == [2020, 2021]


def test_restatement_uses_latest_filed_available_as_of_date():
    s = _store([
        ("Assets", "FY", "2020-12-31", "2021-02-15", 100),   # original
        ("Assets", "FY", "2020-12-31", "2022-02-15", 110),   # restated later
    ])
    assert s.annual_history(["Assets"], "2021-06-01").iloc[-1] == 100  # only original visible
    assert s.annual_history(["Assets"], "2022-06-01").iloc[-1] == 110  # restatement now visible


def test_concept_fallback_tries_candidates_in_order():
    s = _store([("SalesRevenueNet", "FY", "2020-12-31", "2021-02-15", 50)])
    h = s.annual_history(["Revenues", "SalesRevenueNet"], "2021-06-01")
    assert h.iloc[-1] == 50


def test_annual_history_merges_across_a_concept_switch():
    # A filer that switched COGS tags after FY2017 (the MSFT pattern): the latest
    # year must come from the NEW tag, not the stale legacy one.
    s = _store([
        ("CostOfRevenue", "FY", "2016-12-31", "2017-02-15", 30),
        ("CostOfRevenue", "FY", "2017-12-31", "2018-02-15", 34),
        ("CostOfGoodsAndServicesSold", "FY", "2018-12-31", "2019-02-15", 62),
        ("CostOfGoodsAndServicesSold", "FY", "2019-12-31", "2020-02-15", 65),
    ])
    h = s.annual_history(["CostOfRevenue", "CostOfGoodsAndServicesSold"], "2020-06-01")
    assert list(h.index.year) == [2016, 2017, 2018, 2019]
    assert h.iloc[-1] == 65  # newest year from the current tag, not stale CostOfRevenue


def test_higher_priority_concept_wins_in_overlap_years():
    # Both tags report the same year; the higher-priority (first-listed) one wins.
    s = _store([
        ("RevenueFromContractWithCustomerExcludingAssessedTax",
         "FY", "2020-12-31", "2021-02-15", 1000),
        ("Revenues", "FY", "2020-12-31", "2021-02-15", 250),  # stale/partial
    ])
    h = s.annual_history(
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"], "2021-06-01"
    )
    assert h.iloc[-1] == 1000


def test_gross_profitability_value_and_point_in_time():
    s = _store([
        ("Revenues", "FY", "2020-12-31", "2021-02-15", 1000),
        ("CostOfRevenue", "FY", "2020-12-31", "2021-02-15", 600),
        ("Assets", "FY", "2020-12-31", "2021-02-15", 2000),
    ])
    assert gross_profitability(s, "2021-06-01") == pytest.approx(0.2)  # (1000-600)/2000
    assert gross_profitability(s, "2021-01-01") is None  # nothing filed yet


def test_gross_profitability_aligns_on_common_fiscal_year():
    s = _store([
        ("Revenues", "FY", "2020-12-31", "2021-02-15", 1000),
        ("CostOfRevenue", "FY", "2020-12-31", "2021-02-15", 600),
        ("Assets", "FY", "2020-12-31", "2021-02-15", 2000),
        ("Revenues", "FY", "2021-12-31", "2022-02-15", 1200),
        ("CostOfRevenue", "FY", "2021-12-31", "2022-02-15", 700),
        # 2021 Assets deliberately absent -> must fall back to the 2020 common year.
    ])
    assert gross_profitability(s, "2022-06-01") == pytest.approx(0.2)


def test_asset_growth_value_and_min_years():
    s = _store([
        ("Assets", "FY", "2019-12-31", "2020-02-15", 100),
        ("Assets", "FY", "2020-12-31", "2021-02-15", 130),
    ])
    assert asset_growth(s, "2021-06-01") == pytest.approx(0.30)
    assert asset_growth(s, "2020-06-01") is None  # only one year filed


def test_characteristics_return_none_when_data_missing():
    s = _store([("Assets", "FY", "2020-12-31", "2021-02-15", 100)])
    assert gross_profitability(s, "2021-06-01") is None  # no revenue/cogs


def test_empty_history_for_unknown_concept():
    s = _store([("Assets", "FY", "2020-12-31", "2021-02-15", 100)])
    assert s.annual_history(["Revenues"], "2021-06-01").empty


# --- quarterly_history + standardized unexpected earnings (PEAD) -------------

def _qstore(rows):
    """rows: (concept, start, end, filed, val) — quarterly facts with a `start` date."""
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "filed", "val"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["fp"] = "Q"
    df["form"] = "10-Q"
    return FactStore(ticker="TEST", cik="0", facts=df)


def _quarters(values_by_end):
    """Discrete (~90-day) quarterly NetIncomeLoss rows from {quarter-end: value},
    each filed ~40 days after the quarter-end."""
    rows = []
    for end, val in values_by_end.items():
        e = pd.Timestamp(end)
        rows.append((
            "NetIncomeLoss",
            (e - pd.Timedelta(days=90)).date().isoformat(),
            end,
            (e + pd.Timedelta(days=40)).date().isoformat(),
            val,
        ))
    return rows


def test_quarterly_history_isolates_discrete_quarter_and_is_point_in_time():
    s = _qstore([
        ("NetIncomeLoss", "2020-04-01", "2020-06-30", "2020-08-05", 25),  # discrete Q2 (90d)
        ("NetIncomeLoss", "2020-01-01", "2020-06-30", "2020-08-05", 47),  # H1 YTD (181d) -> excluded
        ("NetIncomeLoss", "2020-07-01", "2020-09-30", "2020-11-05", 30),  # discrete Q3 (91d)
    ])
    # Q3 not yet filed; the 6-month YTD row sharing Q2's end must be excluded by duration.
    h = s.quarterly_history(["NetIncomeLoss"], "2020-09-15")
    assert list(h.values) == [25.0]
    assert h.index[0] == pd.Timestamp("2020-06-30")
    # Once Q3 is filed it appears too, sorted by quarter-end.
    assert list(s.quarterly_history(["NetIncomeLoss"], "2020-12-01").values) == [25.0, 30.0]


def test_quarterly_history_uses_latest_filed_value():
    s = _qstore([
        ("NetIncomeLoss", "2020-01-01", "2020-03-31", "2020-05-05", 20),  # original Q1
        ("NetIncomeLoss", "2020-01-01", "2020-03-31", "2021-05-04", 22),  # restated a year later
    ])
    assert list(s.quarterly_history(["NetIncomeLoss"], "2020-06-01").values) == [20.0]
    assert list(s.quarterly_history(["NetIncomeLoss"], "2021-06-01").values) == [22.0]


def test_quarterly_history_empty_without_start_column():
    # A store built the old way (no `start`) cannot resolve durations -> empty, no crash.
    df = pd.DataFrame([("NetIncomeLoss", "Q1", "2020-03-31", "2020-05-05", 20)],
                      columns=["concept", "fp", "end", "filed", "val"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    s = FactStore(ticker="T", cik="0", facts=df)
    assert s.quarterly_history(["NetIncomeLoss"], "2021-01-01").empty


_BASE_QUARTERS = {  # two prior years with mild dispersion in YoY changes (~+8..+12)
    "2019-03-31": 100, "2019-06-30": 100, "2019-09-30": 100, "2019-12-31": 100,
    "2020-03-31": 108, "2020-06-30": 112, "2020-09-30": 109, "2020-12-31": 111,
}


def test_sue_positive_when_latest_surprise_beats_trend():
    s = _qstore(_quarters({**_BASE_QUARTERS, "2021-03-31": 200}))  # +92 YoY vs ~+10 trend
    sue = standardized_unexpected_earnings(s, "2021-07-01")
    assert sue is not None and sue > 5


def test_sue_negative_when_latest_surprise_below_trend():
    s = _qstore(_quarters({**_BASE_QUARTERS, "2021-03-31": 40}))  # -68 YoY vs ~+10 trend
    sue = standardized_unexpected_earnings(s, "2021-07-01")
    assert sue is not None and sue < -5


def test_sue_none_with_insufficient_history():
    s = _qstore(_quarters(_BASE_QUARTERS))  # only 4 seasonal differences < min_history=5
    assert standardized_unexpected_earnings(s, "2021-07-01") is None


def test_sue_none_when_history_has_no_dispersion():
    flat = {  # every YoY change is identical (+10) -> std(prior)=0
        "2019-03-31": 100, "2019-06-30": 100, "2019-09-30": 100, "2019-12-31": 100,
        "2020-03-31": 110, "2020-06-30": 110, "2020-09-30": 110, "2020-12-31": 110,
        "2021-03-31": 120,
    }
    assert standardized_unexpected_earnings(_qstore(_quarters(flat)), "2021-07-01") is None


def test_sue_is_point_in_time_as_new_quarters_file():
    s = _qstore(_quarters({**_BASE_QUARTERS, "2021-03-31": 130, "2021-06-30": 300}))
    # 2021-06-30 (a blowout) is filed ~2021-08-09, so it is invisible on 2021-07-01.
    before = standardized_unexpected_earnings(s, "2021-07-01")
    after = standardized_unexpected_earnings(s, "2021-09-01")
    assert before is not None and after is not None
    assert after > before  # the newly-public blowout raises the surprise
