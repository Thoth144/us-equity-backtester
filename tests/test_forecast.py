"""ML forecast tests — synthetic data, no network.

The two load-bearing tests are statistical: random features must yield ~zero
out-of-sample IC (no leakage), and a planted signal must be recovered. A third
checks the GBM beats the linear baseline on a pure interaction it alone can see.
"""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.forecast import (
    _ml_folds,
    build_design_matrix,
    fit_cross_sectional_forecast,
)


def _feature_panels(n_dates, n_tickers, n_feats, seed):
    dates = pd.bdate_range("2018-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(seed)
    panels = {
        f"f{j}": pd.DataFrame(
            rng.standard_normal((n_dates, n_tickers)), index=dates, columns=tickers
        )
        for j in range(n_feats)
    }
    return dates, tickers, panels, rng


# --- folds ------------------------------------------------------------------

def test_ml_folds_purge_and_expanding():
    assert _ml_folds(20, 10, 5, 1) == [(0, 9, 10, 15), (0, 14, 15, 20)]
    assert _ml_folds(20, 10, 5, 0) == [(0, 10, 10, 15), (0, 15, 15, 20)]


def test_ml_folds_empty_when_too_short():
    assert _ml_folds(8, 10, 5, 1) == []


# --- design matrix ----------------------------------------------------------

def test_build_design_matrix_stacks_and_drops_incomplete_rows():
    dates = pd.bdate_range("2020-01-01", periods=3)
    f0 = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=dates)
    f1 = pd.DataFrame({"A": [7.0, 8.0, 9.0], "B": [10.0, 11.0, 12.0]}, index=dates)
    fwd = pd.DataFrame({"A": [0.1, 0.2, 0.3], "B": [0.4, 0.5, 0.6]}, index=dates)
    f0.iloc[0, 0] = np.nan  # (date0, A) becomes an incomplete case

    X, y = build_design_matrix({"f0": f0, "f1": f1}, fwd)

    assert list(X.columns) == ["f0", "f1"]
    assert X.index.names == ["date", "ticker"]
    assert X.index.equals(y.index)
    assert len(X) == 5  # 6 cells minus the one with a NaN feature
    assert (dates[0], "A") not in X.index
    assert X.loc[(dates[0], "B"), "f0"] == 4.0
    assert y.loc[(dates[0], "B")] == 0.4


def test_build_design_matrix_raises_on_empty():
    with pytest.raises(ValueError):
        build_design_matrix({}, pd.DataFrame())


def test_build_design_matrix_tolerates_duplicate_columns():
    """yfinance sometimes emits a duplicated ticker column; the duplicate once
    produced a non-unique (date, ticker) index that crashed the join."""
    dates = pd.bdate_range("2020-01-01", periods=2)
    f0 = pd.DataFrame([[1.0, 9.0], [2.0, 8.0]], index=dates, columns=["A", "A"])  # dup col
    fwd = pd.DataFrame({"A": [0.1, 0.2]}, index=dates)

    X, y = build_design_matrix({"f0": f0}, fwd)

    assert list(X.columns) == ["f0"]
    assert X.index.is_unique
    assert X.index.names == ["date", "ticker"]
    assert X.loc[(dates[0], "A"), "f0"] == 1.0  # first occurrence kept


# --- learning behavior ------------------------------------------------------

def test_no_leakage_random_features_give_zero_ic():
    dates, tickers, panels, rng = _feature_panels(60, 40, 2, seed=0)
    label = pd.DataFrame(rng.standard_normal((60, 40)), index=dates, columns=tickers)
    X, y = build_design_matrix(panels, label)
    res = fit_cross_sectional_forecast(X, y, train_size=30, test_size=5)
    assert res.n_folds == 6
    assert abs(res.mean_ic) < 0.1  # no signal -> no out-of-sample skill


def test_recovers_a_linear_signal():
    dates, tickers, panels, rng = _feature_panels(60, 40, 2, seed=1)
    noise = rng.standard_normal((60, 40))
    label = pd.DataFrame(
        0.7 * panels["f0"].to_numpy() + 0.3 * noise, index=dates, columns=tickers
    )
    X, y = build_design_matrix(panels, label)
    res = fit_cross_sectional_forecast(X, y, train_size=30, test_size=5)
    assert res.mean_ic > 0.3
    assert res.feature_importance["f0"] > res.feature_importance["f1"]


def test_gbm_beats_linear_on_interaction():
    dates, tickers, panels, rng = _feature_panels(80, 50, 2, seed=2)
    interaction = panels["f0"].to_numpy() * panels["f1"].to_numpy()
    noise = rng.standard_normal((80, 50))
    label = pd.DataFrame(interaction + 0.05 * noise, index=dates, columns=tickers)
    X, y = build_design_matrix(panels, label)
    res = fit_cross_sectional_forecast(X, y, train_size=40, test_size=5)
    assert res.baseline_mean_ic < 0.1  # a linear model can't see f0*f1
    assert res.mean_ic > res.baseline_mean_ic + 0.05
    assert res.mean_ic > 0.1


def test_fit_raises_when_too_few_dates():
    dates, tickers, panels, rng = _feature_panels(20, 10, 2, seed=3)
    label = pd.DataFrame(rng.standard_normal((20, 10)), index=dates, columns=tickers)
    X, y = build_design_matrix(panels, label)
    with pytest.raises(ValueError):
        fit_cross_sectional_forecast(X, y, train_size=30, test_size=5)
