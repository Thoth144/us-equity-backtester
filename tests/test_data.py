"""Tests for delisting-bias correction — synthetic data, no network.

`load_ohlc` hits yfinance and is exercised manually; `splice_delistings` is
pure and is the survivorship-bias fix, so it gets the coverage here.
"""

import pandas as pd
import pytest

from equity_backtester.data import _download_cached, splice_delistings


def test_splice_delistings_books_terminal_return_then_exits():
    idx = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame(
        {"DEAD": [100.0, 110.0, 121.0, 130.0, 140.0],
         "LIVE": [50.0, 51.0, 52.0, 53.0, 54.0]},
        index=idx,
    )
    # DEAD last trades on day 2 (121), then takes a -50% delisting return on day 3.
    out = splice_delistings(closes, {"DEAD": (idx[2], -0.5)})
    assert out.loc[idx[2], "DEAD"] == 121.0                       # last normal price kept
    assert out.loc[idx[3], "DEAD"] == pytest.approx(121.0 * 0.5)  # terminal move booked
    assert pd.isna(out.loc[idx[4], "DEAD"])                       # gone thereafter
    assert out["DEAD"].pct_change().loc[idx[3]] == pytest.approx(-0.5)  # return realized
    pd.testing.assert_series_equal(out["LIVE"], closes["LIVE"])   # other names untouched


def test_splice_delistings_total_wipeout_goes_to_zero():
    idx = pd.bdate_range("2020-01-01", periods=4)
    closes = pd.DataFrame({"BK": [10.0, 12.0, 11.0, 13.0]}, index=idx)
    out = splice_delistings(closes, {"BK": (idx[1], -1.0)})
    assert out.loc[idx[2], "BK"] == pytest.approx(0.0)  # bankruptcy -> zero
    assert pd.isna(out.loc[idx[3], "BK"])


def test_splice_delistings_ignores_unknown_ticker():
    idx = pd.bdate_range("2020-01-01", periods=3)
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    pd.testing.assert_frame_equal(splice_delistings(closes, {"ZZZ": (idx[1], -1.0)}), closes)


def test_splice_delistings_noop_when_last_date_is_panel_end():
    idx = pd.bdate_range("2020-01-01", periods=3)
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    # No bar after the last_trading_date to book on -> prices preserved unchanged.
    out = splice_delistings(closes, {"A": (idx[-1], -1.0)})
    pd.testing.assert_series_equal(out["A"], closes["A"])


def test_download_cache_keys_on_ticker_identity_not_count(tmp_path, monkeypatch):
    """Distinct ticker lists of equal length over the same window must not collide.

    Regression: the cache key once used only ``len(tickers)``, so a second request
    for a different equal-length universe got the first one's prices back silently.
    """
    calls = []

    def fake_download(tickers, start, end):
        calls.append(list(tickers))
        return pd.DataFrame({t: [1.0] for t in tickers}, index=pd.to_datetime(["2020-01-02"]))

    monkeypatch.setattr("equity_backtester.data._download", fake_download)

    a = _download_cached(["AAA", "BBB"], "2020-01-01", "2020-12-31", tmp_path)
    b = _download_cached(["CCC", "DDD"], "2020-01-01", "2020-12-31", tmp_path)

    assert list(a.columns) == ["AAA", "BBB"]
    assert list(b.columns) == ["CCC", "DDD"]  # not a stale hit of AAA/BBB
    assert len(calls) == 2                     # both distinct universes were fetched

    # The same universe (any order) reuses the cache -> no extra download.
    again = _download_cached(["BBB", "AAA"], "2020-01-01", "2020-12-31", tmp_path)
    assert len(calls) == 2
    assert set(again.columns) == {"AAA", "BBB"}
