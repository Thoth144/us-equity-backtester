"""Meta-labeling: a secondary model that sizes the primary model's bets.

The pipeline so far produces a primary signal that picks a *side* — long the
top-quantile names, short the bottom-quantile (T6's `scores_to_weights` logic).
Meta-labeling (Lopez de Prado, "Advances in Financial Machine Learning", 2018,
Ch. 3) leaves the side alone and learns a *second*, binary question: given that
the primary wants to make this bet, how likely is it to pay off? The secondary
model's probability sizes (or filters) the bet — high conviction gets weight,
low conviction gets dropped — which lifts precision and F1 even when the
primary's raw edge is thin.

Two pieces, mirroring `forecast.py`:

- `meta_labels` turns a primary score panel and forward returns into the
  secondary training set: one row per bet the primary would place, a `side`
  (+1 long / -1 short), and a binary `meta_label` (1 if that directional bet was
  profitable). Only names the primary actually bets on become rows — the
  defining feature of meta-labeling, which conditions the second model on the
  first's selections rather than the whole cross-section.

- `fit_meta_model` trains a gradient-boosted *classifier* on those bets using
  the same purged, expanding walk-forward folds as the return forecaster (the
  bet's own side is included as a feature, so the model can learn that longs and
  shorts are reliable under different conditions). It reports the metrics that
  matter for a filter — precision, recall, F1, ROC AUC — against the base rate
  (how often a raw bet wins), and emits a `sized_score = side * P(win)` for
  downstream sizing.

Out of scope by design: triple-barrier / horizon labeling (this is a monthly
cross-sectional book, not an intraday path problem) and bet-sizing beyond the
linear `side * probability` map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .forecast import _ml_folds


@dataclass
class MetaResult:
    meta_prob: pd.Series       # OOS P(bet profitable), (date, ticker) MultiIndex
    sized_score: pd.Series     # signed conviction: side * meta_prob
    precision: float           # OOS precision at the 0.5 threshold
    recall: float              # OOS recall at the 0.5 threshold
    f1: float                  # OOS F1 at the 0.5 threshold
    base_rate: float           # fraction of bets that were profitable
    auc: float                 # OOS ROC AUC (nan if only one class OOS)
    n_folds: int


def meta_labels(
    primary_scores: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    *,
    quantile: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """Build the secondary training set from a primary signal and forward returns.

    For each date the primary goes long the top `quantile` of names and short the
    bottom `quantile` (matching `scores_to_weights`). Each bet becomes one row:
    `side` is +1 (long) or -1 (short), and `meta_label` is 1 if the bet was
    profitable (long & return > 0, or short & return < 0) else 0. Returns
    (`sides`, `labels`) sharing a (date, ticker) MultiIndex over bet rows only;
    bets whose forward return is missing are dropped.
    """
    if not 0.0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    rows = []
    for date, row in primary_scores.iterrows():
        if date not in fwd_returns.index:
            continue
        s = row.dropna()
        k = min(int(len(s) * quantile), len(s) // 2)  # disjoint long/short legs
        if k < 1:
            continue
        fr = fwd_returns.loc[date]
        ranked = s.sort_values()
        for ticker in ranked.index[-k:]:
            r = fr.get(ticker, np.nan)
            if np.isfinite(r):
                rows.append((date, ticker, 1.0, 1.0 if r > 0 else 0.0))
        for ticker in ranked.index[:k]:
            r = fr.get(ticker, np.nan)
            if np.isfinite(r):
                rows.append((date, ticker, -1.0, 1.0 if r < 0 else 0.0))
    frame = pd.DataFrame(rows, columns=["date", "ticker", "side", "meta_label"])
    frame = frame.set_index(["date", "ticker"])
    return frame["side"], frame["meta_label"]


def fit_meta_model(
    features: pd.DataFrame,
    sides: pd.Series,
    labels: pd.Series,
    train_size: int,
    test_size: int,
    *,
    purge: int = 1,
    model=None,
) -> MetaResult:
    """Walk-forward meta-model: classify whether each primary bet pays off.

    `features` is a (date, ticker) design matrix (e.g. from `build_design_matrix`);
    only rows that are bets — present in `sides`/`labels` from `meta_labels` — are
    used, with the bet's `side` appended as a feature. Folds are the same purged,
    expanding windows as `fit_cross_sectional_forecast` (`train_size`/`test_size`
    in unique dates). Returns a `MetaResult` with OOS probabilities and the
    precision/recall/F1/AUC of the filter against the base rate.
    """
    common = features.index.intersection(sides.index)
    if len(common) == 0:
        raise ValueError("no overlap between features and bet rows")
    X = features.loc[common].copy()
    X["_side"] = sides.loc[common]
    y = labels.loc[common]

    dates = X.index.get_level_values(0).unique().sort_values()
    folds = _ml_folds(len(dates), train_size, test_size, purge)
    if not folds:
        raise ValueError(
            f"not enough dates for a fold: need > train_size+purge, have {len(dates)}"
        )
    if model is None:
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=0
        )

    date_level = X.index.get_level_values(0)
    probs: list[pd.Series] = []
    for train_a, train_b, test_a, test_b in folds:
        train_mask = date_level.isin(dates[train_a:train_b])
        test_mask = date_level.isin(dates[test_a:test_b])
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te = X[test_mask]
        if y_tr.nunique() < 2:
            # Degenerate fold (one class only): predict that class's frequency.
            p = np.full(len(X_te), float(y_tr.mean()))
        else:
            model.fit(X_tr.to_numpy(), y_tr.to_numpy())
            p = model.predict_proba(X_te.to_numpy())[:, 1]
        probs.append(pd.Series(p, index=X_te.index))

    meta_prob = pd.concat(probs).rename("meta_prob")
    y_oos = y.reindex(meta_prob.index).to_numpy().astype(int)
    prob_arr = meta_prob.to_numpy()
    pred = (prob_arr >= 0.5).astype(int)
    auc = roc_auc_score(y_oos, prob_arr) if len(np.unique(y_oos)) == 2 else float("nan")
    sized_score = (sides.loc[common].reindex(meta_prob.index) * meta_prob).rename("sized_score")

    return MetaResult(
        meta_prob=meta_prob,
        sized_score=sized_score,
        precision=float(precision_score(y_oos, pred, zero_division=0)),
        recall=float(recall_score(y_oos, pred, zero_division=0)),
        f1=float(f1_score(y_oos, pred, zero_division=0)),
        base_rate=float(y_oos.mean()),
        auc=float(auc),
        n_folds=len(folds),
    )
