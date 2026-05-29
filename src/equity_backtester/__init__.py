"""US-equities backtesting engine."""

from .bab import BabResult, bab_factor, rolling_beta
from .costs import CostModel, corwin_schultz_spread
from .data import splice_delistings
from .engine import BacktestResult, run_backtest
from .factors import FactorAttribution, factor_attribution, load_factors
from .forecast import (
    ForecastResult,
    build_design_matrix,
    fit_cross_sectional_forecast,
)
from .fundamentals import (
    FactStore,
    asset_growth,
    gross_profitability,
    load_company_facts,
    shares_outstanding,
    standardized_unexpected_earnings,
)
from .meta import MetaResult, fit_meta_model, meta_labels
from .metrics import (
    PBOResult,
    PerformanceSummary,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    summarize,
)
from .portfolio import PortfolioResult, backtest_portfolio, scores_to_weights
from .risk import volatility_target_weights
from .signals import (
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
from .strategy import SMACrossover, Strategy
from .universe import (
    sp500_changes,
    sp500_members_at,
    sp500_membership_panel,
    sp500_ohlc,
    sp500_tickers,
    sp600_ohlc,
    sp600_tickers,
)
from .walkforward import (
    CombinatorialResult,
    CombinatorialScheme,
    WalkForwardResult,
    combinatorial_purged_splits,
    combinatorial_walk_forward,
    walk_forward,
)

__all__ = [
    "BabResult",
    "BacktestResult",
    "CombinatorialResult",
    "CombinatorialScheme",
    "CostModel",
    "FactStore",
    "FactorAttribution",
    "ForecastResult",
    "MetaResult",
    "PBOResult",
    "PerformanceSummary",
    "PortfolioResult",
    "SMACrossover",
    "Strategy",
    "WalkForwardResult",
    "asset_growth",
    "asset_growth_signal",
    "bab_factor",
    "backtest_portfolio",
    "build_design_matrix",
    "combinatorial_purged_splits",
    "corwin_schultz_spread",
    "combinatorial_walk_forward",
    "deflated_sharpe_ratio",
    "earnings_surprise_signal",
    "factor_attribution",
    "fit_cross_sectional_forecast",
    "fit_meta_model",
    "forward_returns",
    "gross_profitability",
    "load_company_facts",
    "load_factors",
    "low_vol_signal",
    "meta_labels",
    "momentum_signal",
    "monthly_rebalance_dates",
    "profitability_signal",
    "quantile_spread",
    "reversal_signal",
    "rolling_beta",
    "shares_outstanding",
    "standardized_unexpected_earnings",
    "value_signal",
    "zscore_cross_section",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "run_backtest",
    "scores_to_weights",
    "sp500_changes",
    "sp500_members_at",
    "sp500_membership_panel",
    "sp500_ohlc",
    "sp500_tickers",
    "sp600_ohlc",
    "sp600_tickers",
    "splice_delistings",
    "summarize",
    "volatility_target_weights",
    "walk_forward",
]
