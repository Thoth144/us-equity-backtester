"""Unit tests for the universe module — synthetic data, no Wikipedia calls."""

import pandas as pd
import pytest

from equity_backtester.universe import (
    sp500_members_at,
    sp500_membership_panel,
    sp500_ohlc,
    sp600_ohlc,
    sp600_tickers,
)


@pytest.fixture
def fake_history():
    """A tiny S&P-like history.

    Today's members: {A, B, D}.
    Changes:
      2020-03-01:  +B (added), -C (removed)
      2021-06-15:  +D (added), -E (removed)

    So historical membership is:
      Before 2020-03-01:        {A, C, E}
      2020-03-01 to 2021-06-14: {A, B, E}
      2021-06-15 onward:        {A, B, D}
    """
    current = ["A", "B", "D"]
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-01", "2021-06-15"]),
        "added_ticker": ["B", "D"],
        "added_security": ["BCorp", "DCorp"],
        "removed_ticker": ["C", "E"],
        "removed_security": ["CCorp", "ECorp"],
    })
    return current, changes


def test_members_at_today_matches_current(fake_history):
    current, changes = fake_history
    assert sp500_members_at("2025-01-01", current, changes) == {"A", "B", "D"}


def test_members_at_before_any_change(fake_history):
    current, changes = fake_history
    assert sp500_members_at("2019-01-01", current, changes) == {"A", "C", "E"}


def test_members_at_between_changes(fake_history):
    current, changes = fake_history
    assert sp500_members_at("2020-06-01", current, changes) == {"A", "B", "E"}


def test_members_at_exact_change_date_reflects_the_change(fake_history):
    """A change whose effective date == target_date is in effect on that day."""
    current, changes = fake_history
    assert sp500_members_at("2020-03-01", current, changes) == {"A", "B", "E"}


def test_membership_panel_columns_are_union_of_ever_members(fake_history):
    current, changes = fake_history
    panel = sp500_membership_panel("2020-01-01", "2021-12-31", current, changes)
    assert set(panel.columns) == {"A", "B", "C", "D", "E"}


def test_membership_panel_reflects_changes_over_time(fake_history):
    current, changes = fake_history
    panel = sp500_membership_panel("2020-01-01", "2021-12-31", current, changes)

    before = panel.loc["2020-02-03"]
    assert before["A"] and before["C"] and before["E"]
    assert not before["B"] and not before["D"]

    mid = panel.loc["2020-06-01"]
    assert mid["A"] and mid["B"] and mid["E"]
    assert not mid["C"] and not mid["D"]

    after = panel.loc["2021-08-02"]
    assert after["A"] and after["B"] and after["D"]
    assert not after["C"] and not after["E"]


def test_membership_panel_index_is_trading_days(fake_history):
    current, changes = fake_history
    panel = sp500_membership_panel("2020-01-01", "2020-01-10", current, changes)
    # Jan 1 2020 is New Year's Day (closed); Jan 2 is the first trading day.
    assert pd.Timestamp("2020-01-02") in panel.index
    assert all(d.weekday() < 5 for d in panel.index)


# --- sp500_ohlc -------------------------------------------------------------
# Network-free: monkeypatch the two collaborators (current members + the loader)
# so the wrapper's wiring is tested without hitting Wikipedia or Yahoo.

def test_sp500_ohlc_loads_current_members_and_forwards_args(monkeypatch):
    captured = {}

    def fake_load(tickers, start, end, cache_dir=None):
        captured.update(tickers=tickers, start=start, end=end, cache_dir=cache_dir)
        return {"Close": pd.DataFrame({"AAA": [1.0]})}

    monkeypatch.setattr("equity_backtester.universe.sp500_tickers", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr("equity_backtester.universe.load_ohlc", fake_load)

    out = sp500_ohlc("2010-01-01", "2025-01-01", cache_dir=".cache")
    assert captured["tickers"] == ["AAA", "BBB", "CCC"]  # today's members become the universe
    assert (captured["start"], captured["end"], captured["cache_dir"]) == (
        "2010-01-01", "2025-01-01", ".cache",
    )
    assert "Close" in out  # load_ohlc's dict is passed straight through


def test_sp500_ohlc_defaults_cache_dir_to_none(monkeypatch):
    seen = {}

    def fake_load(tickers, start, end, cache_dir=None):
        seen["cache_dir"] = cache_dir
        return {}

    monkeypatch.setattr("equity_backtester.universe.sp500_tickers", lambda: ["X"])
    monkeypatch.setattr("equity_backtester.universe.load_ohlc", fake_load)

    sp500_ohlc("2020-01-01", "2020-12-31")
    assert seen["cache_dir"] is None


# --- sp600 (small-cap) ------------------------------------------------------

def test_sp600_tickers_scrapes_600_page_and_normalizes(monkeypatch):
    captured = {}

    def fake_fetch(url):
        captured["url"] = url
        return [pd.DataFrame({"Symbol": ["AAA", "BRK.B", "CCC"]})]

    monkeypatch.setattr("equity_backtester.universe._fetch_wiki_tables", fake_fetch)
    out = sp600_tickers()
    assert out == ["AAA", "BRK-B", "CCC"]  # '.' -> '-' Yahoo convention
    assert "600" in captured["url"]  # the S&P 600 page, not the 500 page


def test_sp600_ohlc_loads_current_members_and_forwards_args(monkeypatch):
    captured = {}

    def fake_load(tickers, start, end, cache_dir=None):
        captured.update(tickers=tickers, start=start, end=end, cache_dir=cache_dir)
        return {"Close": pd.DataFrame({"AAA": [1.0]})}

    monkeypatch.setattr("equity_backtester.universe.sp600_tickers", lambda: ["AAA", "BBB"])
    monkeypatch.setattr("equity_backtester.universe.load_ohlc", fake_load)

    out = sp600_ohlc("2010-01-01", "2025-01-01", cache_dir=".cache")
    assert captured["tickers"] == ["AAA", "BBB"]
    assert (captured["start"], captured["end"], captured["cache_dir"]) == (
        "2010-01-01", "2025-01-01", ".cache",
    )
    assert "Close" in out
