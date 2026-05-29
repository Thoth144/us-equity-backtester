"""Tests for the walk-forward harness — synthetic data, no network."""

import math

import numpy as np
import pandas as pd
import pytest

from equity_backtester.costs import CostModel
from equity_backtester.strategy import Strategy
from equity_backtester.walkforward import (
    _generate_folds,
    combinatorial_purged_splits,
    combinatorial_walk_forward,
    walk_forward,
)


def _no_fees() -> CostModel:
    return CostModel(commission_per_share=0, sec_fee_rate=0,
                     finra_taf_per_share=0, finra_taf_cap=0, slippage_bps=0)


class HoldOne(Strategy):
    """Always long a single chosen ticker — a trivial parameterized strategy."""

    def __init__(self, ticker):
        self.ticker = ticker

    def generate_signals(self, prices):
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if self.ticker in sig.columns:
            sig[self.ticker] = 1.0
        return sig


def _ramp(spec: dict, n=200, start="2020-01-01"):
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(spec, index=dates)


def test_generate_folds_rolling_count_and_bounds():
    folds = _generate_folds(n=100, train_size=40, test_size=20, anchored=False)
    assert folds == [(0, 40, 40, 60), (20, 60, 60, 80), (40, 80, 80, 100)]


def test_generate_folds_anchored_expands_train():
    folds = _generate_folds(n=100, train_size=40, test_size=20, anchored=True)
    assert folds == [(0, 40, 40, 60), (0, 60, 60, 80), (0, 80, 80, 100)]


def test_selects_param_that_wins_in_training():
    closes = _ramp({"A": np.linspace(100, 300, 200), "B": np.linspace(300, 100, 200)})
    opens = closes.copy()
    res = walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        train_size=60, test_size=30, cost_model=_no_fees(), starting_cash=10_000.0,
    )
    assert all(f.selected_params == {"ticker": "A"} for f in res.folds)
    assert res.oos_equity.iloc[-1] > res.oos_equity.iloc[0]


def test_selection_uses_only_train_window_not_test():
    """A param great in train but terrible in test must still be selected on train."""
    n = 180
    a = np.concatenate([np.linspace(100, 200, 80), np.linspace(200, 50, n - 80)])
    b = np.concatenate([np.full(80, 100.0), np.linspace(100, 400, n - 80)])
    closes = _ramp({"A": a, "B": b}, n=n)
    opens = closes.copy()
    res = walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        train_size=80, test_size=20, cost_model=_no_fees(),
    )
    # Fold 0 trains on [0,80): A rising, B flat -> picks A, despite B soaring in test.
    assert res.folds[0].selected_params == {"ticker": "A"}


def test_oos_length_matches_test_windows():
    closes = _ramp({"A": np.linspace(100, 200, 200), "B": np.linspace(100, 150, 200)})
    opens = closes.copy()
    res = walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        train_size=60, test_size=30, cost_model=_no_fees(),
    )
    folds = _generate_folds(200, 60, 30, False)
    assert len(res.oos_returns) == sum(d - c for (_, _, c, d) in folds)
    assert len(res.trial_sharpes) == 2


def test_anchored_mode_runs():
    closes = _ramp({"A": np.linspace(100, 200, 200), "B": np.linspace(100, 120, 200)})
    opens = closes.copy()
    res = walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        train_size=60, test_size=30, anchored=True, cost_model=_no_fees(),
    )
    assert len(res.folds) >= 1


def test_raises_on_insufficient_data():
    closes = _ramp({"A": np.linspace(100, 110, 50)}, n=50)
    opens = closes.copy()
    with pytest.raises(ValueError):
        walk_forward(closes, opens, lambda p: HoldOne(p["ticker"]),
                     [{"ticker": "A"}], train_size=60, test_size=20)


def test_raises_on_empty_grid():
    closes = _ramp({"A": np.linspace(100, 110, 100)}, n=100)
    opens = closes.copy()
    with pytest.raises(ValueError):
        walk_forward(closes, opens, lambda p: HoldOne(p["ticker"]), [],
                     train_size=40, test_size=20)


# --- combinatorial purged CV (CPCV) -----------------------------------------

def test_cpcv_split_and_path_counts():
    scheme = combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=2)
    assert len(scheme.splits) == math.comb(6, 2)   # 15 train/test combinations
    assert scheme.n_paths == math.comb(5, 1)       # 5 reconstructable backtest paths
    assert len(scheme.groups) == 6


def test_cpcv_each_group_tested_in_n_paths_splits():
    scheme = combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=2)
    counts = {g: 0 for g in range(6)}
    for split in scheme.splits:
        for g in split.test_groups:
            counts[g] += 1
    assert set(counts.values()) == {scheme.n_paths}  # every group tested equally often


def test_cpcv_train_test_disjoint_and_cover_all_without_purge():
    scheme = combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=2)
    for split in scheme.splits:
        assert not set(split.train_idx.tolist()) & set(split.test_idx.tolist())
        assert len(split.train_idx) + len(split.test_idx) == 60  # purge=0 -> full cover


def test_cpcv_purge_removes_train_positions_before_test_block():
    # 6 groups over 60 obs -> group 3 occupies positions [30, 40).
    scheme = combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=1, purge=3)
    split = next(s for s in scheme.splits if s.test_groups == (3,))
    assert {27, 28, 29}.isdisjoint(split.train_idx.tolist())  # purged
    assert 26 in split.train_idx.tolist()                     # just outside the purge


def test_cpcv_embargo_removes_train_positions_after_test_block():
    # group 2 occupies [20, 30); embargo=3 drops train positions 30, 31, 32.
    scheme = combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=1, embargo=3)
    split = next(s for s in scheme.splits if s.test_groups == (2,))
    assert {30, 31, 32}.isdisjoint(split.train_idx.tolist())
    assert 33 in split.train_idx.tolist()


def test_cpcv_rejects_bad_group_counts():
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_obs=60, n_groups=6, n_test_groups=6)  # k must be < N
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_obs=4, n_groups=6, n_test_groups=2)   # N > n_obs


def test_cpcv_walk_forward_reconstructs_full_timeline_paths():
    closes = _ramp({"A": np.linspace(100, 300, 200), "B": np.linspace(300, 100, 200)})
    opens = closes.copy()
    res = combinatorial_walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        n_groups=5, n_test_groups=2, cost_model=_no_fees(),
    )
    assert res.n_paths == math.comb(4, 1) == 4
    assert len(res.paths) == 4
    for path in res.paths:
        assert len(path) == len(closes)              # each path covers the whole timeline
        assert path.index.is_monotonic_increasing
    assert len(res.path_sharpes) == 4
    assert len(res.trial_sharpes) == 2


def test_cpcv_walk_forward_selects_dominant_param_each_split():
    # A rises everywhere, B falls everywhere -> every split selects A on its train set.
    closes = _ramp({"A": np.linspace(100, 300, 200), "B": np.linspace(300, 100, 200)})
    opens = closes.copy()
    res = combinatorial_walk_forward(
        closes, opens,
        lambda p: HoldOne(p["ticker"]),
        [{"ticker": "A"}, {"ticker": "B"}],
        n_groups=5, n_test_groups=2, cost_model=_no_fees(),
    )
    assert all(sel == {"ticker": "A"} for sel in res.split_selections)
    assert all(s > 0 for s in res.path_sharpes)      # holding the riser -> positive OOS
