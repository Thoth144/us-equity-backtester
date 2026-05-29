"""Cross-sectional ML return forecasting with leakage-safe validation.

Combines the T4 signal panels into a single design matrix and learns a
non-linear map from signals to next-period returns, in the spirit of
Gu-Kelly-Xiu (2020), "Empirical Asset Pricing via Machine Learning". The model
is a gradient-boosted tree (captures interactions a linear factor model
cannot); a Ridge regression serves as the linear baseline, so we can see
whether the non-linearity actually buys anything.

Validation is the part that matters. Naive k-fold CV leaks the future: a
forward return labeled on date t overlaps the window [t, t+1], so training on
date t while testing on t-1 lets tomorrow inform yesterday. We use expanding
walk-forward folds with a *purge* gap (Lopez de Prado, "Advances in Financial
Machine Learning", 2018): training always ends `purge` dates before the test
window starts, dropping the overlapping label. Training labels are also
cross-sectionally demeaned per date, so the model learns relative ranking
(which name beats the cross-section) rather than market-level timing.

Skill is measured by the Information Coefficient — the per-date Spearman rank
correlation between predicted scores and realized returns — and its t-stat
across dates. A model with no edge scores IC ~ 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


@dataclass
class ForecastResult:
    predictions: pd.Series          # OOS predicted scores, (date, ticker) MultiIndex
    ic_by_date: pd.Series           # per-date Spearman rank IC of the model
    mean_ic: float                  # average OOS rank IC
    ic_tstat: float                 # mean_ic / standard error, across dates
    baseline_mean_ic: float         # same metric for the Ridge linear baseline
    feature_importance: pd.Series   # permutation importance per feature (GBM)
    n_folds: int


def build_design_matrix(
    signal_panels: dict[str, pd.DataFrame],
    fwd_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stack per-signal panels into a (date, ticker) design matrix and labels.

    `signal_panels` maps feature name -> panel (dates x tickers); `fwd_returns`
    is the label panel (dates x tickers), typically from `forward_returns`.
    Returns (X, y) sharing a (date, ticker) MultiIndex, complete cases only —
    any NaN in a feature or the label drops that row.
    """
    if not signal_panels:
        raise ValueError("signal_panels is empty")
    # yfinance can emit a duplicated ticker column; a duplicated (date, ticker)
    # index breaks the join below, so keep the first column for each name.
    panels = {
        name: panel.loc[:, ~panel.columns.duplicated()]
        for name, panel in signal_panels.items()
    }
    features = pd.DataFrame({name: panel.stack() for name, panel in panels.items()})
    labels = fwd_returns.loc[:, ~fwd_returns.columns.duplicated()].stack()
    labels.name = "_label"
    combined = features.join(labels, how="inner").dropna()
    X = combined[list(signal_panels)]
    y = combined["_label"]
    X.index.names = ["date", "ticker"]
    y.index.names = ["date", "ticker"]
    y.name = "fwd_return"
    return X, y


def fit_cross_sectional_forecast(
    features: pd.DataFrame,
    labels: pd.Series,
    train_size: int,
    test_size: int,
    *,
    purge: int = 1,
    model=None,
) -> ForecastResult:
    """Walk-forward cross-sectional forecast with purged, leakage-safe folds.

    `features`/`labels` come from `build_design_matrix` (shared (date, ticker)
    MultiIndex). `train_size`/`test_size` are counts of unique *dates*. Each fold
    trains on an expanding window ending `purge` dates before the test window
    (the purge drops labels that overlap the test period). Returns a
    `ForecastResult`; predictions are stitched across the out-of-sample folds.
    """
    if features.empty:
        raise ValueError("features is empty")
    labels = labels.reindex(features.index)
    dates = features.index.get_level_values(0).unique().sort_values()
    folds = _ml_folds(len(dates), train_size, test_size, purge)
    if not folds:
        raise ValueError(
            f"not enough dates for a fold: need > train_size+purge, have {len(dates)}"
        )
    if model is None:
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
        )
    baseline = Ridge(alpha=1.0)

    date_level = features.index.get_level_values(0)
    preds: list[pd.Series] = []
    base_preds: list[pd.Series] = []
    for train_a, train_b, test_a, test_b in folds:
        train_mask = date_level.isin(dates[train_a:train_b])
        test_mask = date_level.isin(dates[test_a:test_b])
        X_tr, y_tr = features[train_mask], labels[train_mask]
        X_te = features[test_mask]
        # Cross-sectionally demean labels: learn relative ranking, not market level.
        y_tr_dm = y_tr - y_tr.groupby(level=0).transform("mean")
        model.fit(X_tr.to_numpy(), y_tr_dm.to_numpy())
        baseline.fit(X_tr.to_numpy(), y_tr_dm.to_numpy())
        preds.append(pd.Series(model.predict(X_te.to_numpy()), index=X_te.index))
        base_preds.append(pd.Series(baseline.predict(X_te.to_numpy()), index=X_te.index))

    predictions = pd.concat(preds)
    ic = _ic_by_date(predictions, labels)
    baseline_ic = _ic_by_date(pd.concat(base_preds), labels)

    return ForecastResult(
        predictions=predictions,
        ic_by_date=ic,
        mean_ic=float(ic.mean()) if len(ic) else 0.0,
        ic_tstat=_ic_tstat(ic),
        baseline_mean_ic=float(baseline_ic.mean()) if len(baseline_ic) else 0.0,
        feature_importance=_permutation_importance(model, features, labels),
        n_folds=len(folds),
    )


def _ml_folds(
    n_dates: int, train_size: int, test_size: int, purge: int,
) -> list[tuple[int, int, int, int]]:
    """Expanding-window CV folds with a purge gap, as date-index positions.

    Train is [0, test_start - purge); test is [test_start, test_end). The purge
    drops the last `purge` train dates whose forward-return labels overlap the
    test window, preventing look-ahead across the train/test boundary.
    """
    folds = []
    k = 0
    while True:
        test_start = train_size + k * test_size
        test_end = test_start + test_size
        train_end = test_start - purge
        if test_end > n_dates or train_end <= 0:
            break
        folds.append((0, train_end, test_start, test_end))
        k += 1
    return folds


def _ic_by_date(predictions: pd.Series, labels: pd.Series) -> pd.Series:
    """Per-date Spearman rank IC between predictions and realized labels."""
    df = pd.DataFrame({"pred": predictions, "label": labels.reindex(predictions.index)})
    df = df.dropna()
    ics = {}
    for date, grp in df.groupby(level=0):
        if len(grp) >= 3:
            ics[date] = grp["pred"].corr(grp["label"], method="spearman")
    return pd.Series(ics, name="ic", dtype=float).dropna()


def _ic_tstat(ic: pd.Series) -> float:
    """t-stat of the mean IC across dates (IID-across-dates approximation)."""
    n = len(ic)
    if n < 2:
        return 0.0
    sd = ic.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(ic.mean() / sd * np.sqrt(n))


def _permutation_importance(model, features: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """GBM permutation importance, refit on the full per-date-demeaned panel."""
    y_dm = labels - labels.groupby(level=0).transform("mean")
    model.fit(features.to_numpy(), y_dm.to_numpy())
    result = permutation_importance(
        model, features.to_numpy(), y_dm.to_numpy(), n_repeats=5, random_state=0
    )
    return pd.Series(result.importances_mean, index=features.columns, name="importance")
