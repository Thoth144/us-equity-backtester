"""Fama-French factor data (Ken French Data Library) and factor attribution.

Separates a strategy's returns into factor exposure (market, size, value,
profitability, investment, momentum, short-term reversal) and residual alpha.
Significance uses Newey-West (HAC) standard errors — the standard correction
for autocorrelated, heteroskedastic financial return regressions; plain OLS
errors understate uncertainty and manufacture false alpha.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_USER_AGENT = "us-equity-backtester/0.1 (educational backtesting tool)"

# (zip filename, data column names in file order)
_DATASETS = [
    ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
     ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]),
    ("F-F_Momentum_Factor_daily_CSV.zip", ["MOM"]),
    ("F-F_ST_Reversal_Factor_daily_CSV.zip", ["ST_Rev"]),
]

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "ST_Rev"]
_MIN_OBS = 60


@dataclass
class FactorAttribution:
    alpha_annual: float          # annualized intercept (the alpha)
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]      # factor -> loading
    tstats: dict[str, float]     # factor -> t-stat
    r_squared: float
    info_ratio: float            # annualized alpha / annualized residual vol
    n_obs: int


def load_factors(start, end) -> pd.DataFrame:
    """Daily Fama-French 5 + Momentum + Short-term-reversal factors and RF.

    Values are decimals (0.01 = 1%). Indexed by date, restricted to [start, end].
    """
    frames = [_fetch_french_csv(name, cols) for name, cols in _DATASETS]
    factors = pd.concat(frames, axis=1).dropna()
    return factors.loc[str(start):str(end)]


def factor_attribution(
    returns: pd.Series,
    factors: pd.DataFrame | None = None,
) -> FactorAttribution:
    """Regress strategy excess returns on the factor returns.

    returns: daily total returns of the strategy.
    factors: panel from load_factors(); fetched automatically if None.
    """
    if factors is None:
        factors = load_factors(returns.index.min(), returns.index.max())

    cols = [c for c in FACTOR_NAMES if c in factors.columns]
    data = pd.concat([returns.rename("ret"), factors[[*cols, "RF"]]], axis=1).dropna()
    if len(data) < _MIN_OBS:
        raise ValueError(f"need >= {_MIN_OBS} overlapping observations, got {len(data)}")

    y = (data["ret"] - data["RF"]).to_numpy()
    X = sm.add_constant(data[cols].to_numpy())
    n = len(y)
    maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    alpha_daily = float(model.params[0])
    resid_std = float(model.resid.std(ddof=1))
    info_ratio = alpha_daily / resid_std * np.sqrt(252) if resid_std > 0 else 0.0

    return FactorAttribution(
        alpha_annual=alpha_daily * 252,
        alpha_tstat=float(model.tvalues[0]),
        alpha_pvalue=float(model.pvalues[0]),
        betas={c: float(b) for c, b in zip(cols, model.params[1:], strict=True)},
        tstats={c: float(t) for c, t in zip(cols, model.tvalues[1:], strict=True)},
        r_squared=float(model.rsquared),
        info_ratio=float(info_ratio),
        n_obs=n,
    )


def _fetch_french_csv(zip_name: str, columns: list[str]) -> pd.DataFrame:
    """Download a Ken French daily factor zip and parse its daily table."""
    resp = requests.get(_FRENCH_BASE + zip_name,
                        headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    text = archive.read(archive.namelist()[0]).decode("latin-1")

    rows = [ln for ln in text.splitlines() if re.match(r"^\s*\d{8}\s*,", ln)]
    if not rows:
        raise ValueError(f"no daily rows found in {zip_name}")
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None,
                     names=["date", *columns], skipinitialspace=True)
    if df.shape[1] != len(columns) + 1:
        raise ValueError(
            f"{zip_name}: expected {len(columns)} data columns, got {df.shape[1] - 1} "
            "— Ken French format may have changed"
        )
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df.where(df > -99) / 100.0  # -99.99 / -999 are missing-value sentinels
