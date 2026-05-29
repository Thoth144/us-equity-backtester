"""Meta-labeling tests — synthetic data, no network.

The load-bearing tests: meta_labels marks a bet profitable iff the primary's
chosen side agreed with the realized forward return, and fit_meta_model recovers
a learnable edge (AUC) while staying honest (~0.5) on noise.
"""

import numpy as np
import pandas as pd
import pytest

from equity_backtester.meta import fit_meta_model, meta_labels


def _panels(n_dates=40, n_tickers=12, seed=0):
    """Random primary scores and forward returns as (dates x tickers) panels."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    scores = pd.DataFrame(rng.standard_normal((n_dates, n_tickers)),
                          index=dates, columns=tickers)
    fwd = pd.DataFrame(rng.standard_normal((n_dates, n_tickers)) * 0.05,
                       index=dates, columns=tickers)
    return scores, fwd


def _features(panel, n_feats=3, seed=1):
    """Random design matrix over every (date, ticker) in `panel`."""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([panel.index, panel.columns],
                                     names=["date", "ticker"])
    return pd.DataFrame(rng.standard_normal((len(idx), n_feats)),
                        index=idx, columns=[f"f{i}" for i in range(n_feats)])


# --- meta_labels ------------------------------------------------------------

def test_meta_labels_perfect_primary_gives_all_ones():
    scores, _ = _panels()
    n = scores.shape[1]
    fwd = scores.rank(axis=1) - (n + 1) / 2.0  # top ranks > 0, bottom ranks < 0
    _, labels = meta_labels(scores, fwd, quantile=0.2)
    assert (labels == 1.0).all()


def test_meta_labels_anti_primary_gives_all_zeros():
    scores, _ = _panels()
    n = scores.shape[1]
    fwd = -(scores.rank(axis=1) - (n + 1) / 2.0)  # top names fall, bottom rise
    _, labels = meta_labels(scores, fwd, quantile=0.2)
    assert (labels == 0.0).all()


def test_meta_labels_bet_count_and_sides():
    scores, fwd = _panels(n_dates=5, n_tickers=10)
    sides, labels = meta_labels(scores, fwd, quantile=0.2)  # k = min(2, 5) = 2
    assert len(sides) == len(labels) == 4 * 5
    longs = (sides > 0).groupby(level=0).sum()
    assert (longs == 2).all()
    assert set(sides.unique()) == {1.0, -1.0}


def test_meta_labels_excludes_nan_label_bets():
    scores, fwd = _panels(n_dates=5, n_tickers=10)
    fwd.iloc[-1] = np.nan  # last date's forward returns unknown
    sides, _ = meta_labels(scores, fwd, quantile=0.2)
    assert len(sides) == 4 * 4
    assert scores.index[-1] not in sides.index.get_level_values(0)


def test_meta_labels_rejects_bad_quantile():
    scores, fwd = _panels()
    with pytest.raises(ValueError):
        meta_labels(scores, fwd, quantile=0.0)
    with pytest.raises(ValueError):
        meta_labels(scores, fwd, quantile=0.6)


# --- fit_meta_model ---------------------------------------------------------

def test_fit_meta_recovers_feature_skill():
    """Label = (f0 > 0): a clean threshold the classifier should find."""
    scores, _ = _panels()
    feats = _features(scores, seed=2)
    sides = pd.Series(1.0, index=feats.index, name="side")
    labels = (feats["f0"] > 0).astype(float).rename("meta_label")
    res = fit_meta_model(feats, sides, labels, train_size=20, test_size=4)
    assert res.auc > 0.8
    assert res.precision > res.base_rate


def test_fit_meta_no_skill_auc_near_half():
    scores, _ = _panels()
    feats = _features(scores, seed=2)
    rng = np.random.default_rng(7)
    sides = pd.Series(1.0, index=feats.index, name="side")
    labels = pd.Series(rng.integers(0, 2, len(feats)).astype(float),
                       index=feats.index, name="meta_label")
    res = fit_meta_model(feats, sides, labels, train_size=20, test_size=4)
    assert 0.30 < res.auc < 0.70


def test_fit_meta_single_class_fold_no_crash():
    scores, _ = _panels()
    feats = _features(scores, seed=2)
    sides = pd.Series(1.0, index=feats.index, name="side")
    labels = pd.Series(1.0, index=feats.index, name="meta_label")  # all winners
    res = fit_meta_model(feats, sides, labels, train_size=20, test_size=4)
    assert np.allclose(res.meta_prob.to_numpy(), 1.0)
    assert np.isnan(res.auc)
    assert res.base_rate == 1.0


def test_fit_meta_sized_score_is_side_times_prob():
    scores, fwd = _panels()
    feats = _features(scores, seed=3)
    sides, labels = meta_labels(scores, fwd, quantile=0.2)
    res = fit_meta_model(feats, sides, labels, train_size=20, test_size=4)
    aligned_sides = sides.reindex(res.meta_prob.index)
    assert np.allclose(res.sized_score.to_numpy(),
                       (aligned_sides * res.meta_prob).to_numpy())
    assert (res.sized_score[aligned_sides < 0] <= 0).all()  # shorts carry negative conviction


def test_fit_meta_raises_when_too_few_dates():
    scores, fwd = _panels(n_dates=10)
    feats = _features(scores, seed=4)
    sides, labels = meta_labels(scores, fwd, quantile=0.2)
    with pytest.raises(ValueError):
        fit_meta_model(feats, sides, labels, train_size=20, test_size=4)


def test_fit_meta_raises_on_no_overlap():
    scores, fwd = _panels()
    sides, labels = meta_labels(scores, fwd, quantile=0.2)
    bad_idx = pd.MultiIndex.from_product([scores.index, ["ZZZ"]],
                                         names=["date", "ticker"])
    feats = pd.DataFrame({"f0": np.zeros(len(bad_idx))}, index=bad_idx)
    with pytest.raises(ValueError):
        fit_meta_model(feats, sides, labels, train_size=20, test_size=4)
