"""Walk-forward analysis: honest out-of-sample parameter selection.

Each fold selects the best parameter set on a train window, then evaluates it
on the following (unseen) test window. The out-of-sample test slices are
stitched into a single equity curve — the only return stream that wasn't used
for any selection decision.

Leak-free by construction: we run one full backtest per parameter set and slice
train/test windows out of the resulting return series. Because strategy signals
are causal (they depend only on data up to and including each bar), a return on
day t never embeds information from day t+1, so slicing cannot leak the future.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .costs import CostModel
from .engine import run_backtest
from .strategy import Strategy


@dataclass
class FoldResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_params: dict
    train_score: float
    test_score: float


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series        # stitched out-of-sample daily returns
    oos_equity: pd.Series         # equity curve from the OOS returns
    folds: list[FoldResult]       # per-fold selection log
    trial_sharpes: list[float]    # full-sample annualized Sharpe per param (for DSR)
    param_grid: list[dict]        # the searched grid, aligned with trial_sharpes


@dataclass
class CPCVSplit:
    test_groups: tuple[int, ...]  # group ids forming the test set of this split
    train_idx: np.ndarray         # train positions, after purge + embargo
    test_idx: np.ndarray          # test positions (union of the test groups)


@dataclass
class CombinatorialScheme:
    groups: list[np.ndarray]      # contiguous position partition (len == n_groups)
    splits: list[CPCVSplit]       # one per C(n_groups, n_test_groups) combination
    n_paths: int                  # paths = C(n_groups-1, n_test_groups-1)


@dataclass
class CombinatorialResult:
    paths: list[pd.Series]        # n_paths full-timeline OOS return paths
    path_sharpes: list[float]     # annualized Sharpe of each reconstructed path
    n_paths: int
    trial_sharpes: list[float]    # full-sample annualized Sharpe per param (for DSR)
    param_grid: list[dict]
    split_selections: list[dict]  # param selected on each split's (purged) train set


def walk_forward(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    strategy_factory: Callable[[dict], Strategy],
    param_grid: list[dict],
    train_size: int,
    test_size: int,
    *,
    anchored: bool = False,
    objective: Callable[[pd.Series], float] | None = None,
    cost_model: CostModel | None = None,
    starting_cash: float = 100_000.0,
    membership_mask: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run walk-forward parameter selection.

    strategy_factory: maps a params dict to a Strategy instance.
    param_grid: list of params dicts to search at each fold.
    train_size, test_size: window lengths in trading days.
    anchored: expanding train window if True, rolling (fixed-size) if False.
    objective: scores a returns slice; higher is better. Default: annualized Sharpe.
    """
    if not param_grid:
        raise ValueError("param_grid is empty")
    if objective is None:
        objective = _annualized_sharpe

    closes, opens = closes.align(opens, join="inner")
    dates = closes.index
    n = len(dates)

    folds = _generate_folds(n, train_size, test_size, anchored)
    if not folds:
        raise ValueError(
            f"not enough data: need >= train_size+test_size="
            f"{train_size + test_size} rows, have {n}"
        )

    # One full backtest per parameter set (returns are causal -> slicing is leak-free).
    param_returns: list[pd.Series] = []
    trial_sharpes: list[float] = []
    for params in param_grid:
        result = run_backtest(
            closes, opens, strategy_factory(params),
            cost_model=cost_model, starting_cash=starting_cash,
            membership_mask=membership_mask,
        )
        param_returns.append(result.returns)
        trial_sharpes.append(_annualized_sharpe(result.returns))

    fold_results: list[FoldResult] = []
    oos_slices: list[pd.Series] = []
    for train_a, train_b, test_a, test_b in folds:
        train_scores = [objective(r.iloc[train_a:train_b]) for r in param_returns]
        best = int(np.argmax(train_scores))
        test_returns = param_returns[best].iloc[test_a:test_b]
        fold_results.append(FoldResult(
            train_start=dates[train_a], train_end=dates[train_b - 1],
            test_start=dates[test_a], test_end=dates[test_b - 1],
            selected_params=param_grid[best],
            train_score=float(train_scores[best]),
            test_score=float(objective(test_returns)),
        ))
        oos_slices.append(test_returns)

    oos_returns = pd.concat(oos_slices)
    oos_equity = starting_cash * (1.0 + oos_returns).cumprod()
    oos_equity.name = "equity"

    return WalkForwardResult(
        oos_returns=oos_returns,
        oos_equity=oos_equity,
        folds=fold_results,
        trial_sharpes=trial_sharpes,
        param_grid=list(param_grid),
    )


def combinatorial_purged_splits(
    n_obs: int,
    n_groups: int,
    n_test_groups: int,
    *,
    purge: int = 0,
    embargo: int = 0,
) -> CombinatorialScheme:
    """Combinatorial purged cross-validation splits (Lopez de Prado 2018, ch. 12).

    Partition `n_obs` observations into `n_groups` contiguous groups, then form
    every way of choosing `n_test_groups` of them as the test set
    (C(n_groups, n_test_groups) splits). Training observations within `purge`
    positions before a test block, or `embargo` positions after it, are dropped
    so a forward-return label spanning the boundary cannot leak.

    Unlike a single walk-forward path, the scheme supports
    C(n_groups-1, n_test_groups-1) reconstructable backtest paths, each covering
    the whole timeline once -> a *distribution* of OOS performance, not a point.
    """
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("need 1 <= n_test_groups < n_groups")
    if n_groups > n_obs:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed n_obs ({n_obs})")

    positions = np.arange(n_obs)
    groups = list(np.array_split(positions, n_groups))
    splits: list[CPCVSplit] = []
    for combo in itertools.combinations(range(n_groups), n_test_groups):
        keep = np.ones(n_obs, dtype=bool)
        for g in combo:
            keep[groups[g]] = False                     # test groups are never train
            a, b = int(groups[g][0]), int(groups[g][-1]) + 1
            keep[max(0, a - purge):a] = False           # purge before the test block
            keep[b:min(n_obs, b + embargo)] = False     # embargo after it
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        splits.append(CPCVSplit(
            test_groups=combo, train_idx=positions[keep], test_idx=test_idx,
        ))

    return CombinatorialScheme(
        groups=groups,
        splits=splits,
        n_paths=math.comb(n_groups - 1, n_test_groups - 1),
    )


def combinatorial_walk_forward(
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    strategy_factory: Callable[[dict], Strategy],
    param_grid: list[dict],
    n_groups: int,
    n_test_groups: int,
    *,
    purge: int = 0,
    embargo: int = 0,
    objective: Callable[[pd.Series], float] | None = None,
    cost_model: CostModel | None = None,
    starting_cash: float = 100_000.0,
    membership_mask: pd.DataFrame | None = None,
) -> CombinatorialResult:
    """Combinatorial purged walk-forward: a distribution of OOS paths, not one.

    Like `walk_forward`, but instead of a single chained OOS path it builds the
    CPCV scheme (`combinatorial_purged_splits`) and reconstructs
    C(n_groups-1, n_test_groups-1) full-timeline OOS return paths. On each split
    the best param is chosen on the (purged) train groups; each test group then
    contributes that param's returns to the paths it belongs to. The spread of
    `path_sharpes` -- not a single number -- is the honest read when PBO is high.
    """
    if not param_grid:
        raise ValueError("param_grid is empty")
    if objective is None:
        objective = _annualized_sharpe

    closes, opens = closes.align(opens, join="inner")
    scheme = combinatorial_purged_splits(
        len(closes.index), n_groups, n_test_groups, purge=purge, embargo=embargo,
    )

    # One full backtest per param (causal returns -> position-slicing is leak-free).
    param_returns: list[pd.Series] = []
    trial_sharpes: list[float] = []
    for params in param_grid:
        result = run_backtest(
            closes, opens, strategy_factory(params),
            cost_model=cost_model, starting_cash=starting_cash,
            membership_mask=membership_mask,
        )
        param_returns.append(result.returns)
        trial_sharpes.append(_annualized_sharpe(result.returns))

    # Per group, the OOS return slice from each split in which it is a test group.
    group_slices: dict[int, list[pd.Series]] = {g: [] for g in range(n_groups)}
    split_selections: list[dict] = []
    for split in scheme.splits:
        train_scores = [objective(r.iloc[split.train_idx]) for r in param_returns]
        best = int(np.argmax(train_scores))
        split_selections.append(param_grid[best])
        for g in split.test_groups:
            group_slices[g].append(param_returns[best].iloc[scheme.groups[g]])

    # Reconstruct paths: path j takes the j-th test occurrence of each group.
    paths: list[pd.Series] = []
    for j in range(scheme.n_paths):
        path = pd.concat([group_slices[g][j] for g in range(n_groups)]).sort_index()
        paths.append(path)
    path_sharpes = [_annualized_sharpe(p) for p in paths]

    return CombinatorialResult(
        paths=paths,
        path_sharpes=path_sharpes,
        n_paths=scheme.n_paths,
        trial_sharpes=trial_sharpes,
        param_grid=list(param_grid),
        split_selections=split_selections,
    )


def _generate_folds(
    n: int, train_size: int, test_size: int, anchored: bool,
) -> list[tuple[int, int, int, int]]:
    """Positional (train_start, train_end, test_start, test_end) index tuples."""
    folds = []
    k = 0
    while True:
        train_start = 0 if anchored else k * test_size
        train_end = train_size + k * test_size if anchored else train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        folds.append((train_start, train_end, test_start, test_end))
        k += 1
    return folds


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    sigma = r.std()
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0
    return float(r.mean() / sigma * np.sqrt(periods_per_year))
