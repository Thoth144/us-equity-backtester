"""Strategy interface and the SMA-crossover implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


class Strategy(ABC):
    """A strategy maps a price panel to a target-signal panel.

    The returned DataFrame has the same shape as `prices`. Each value is the
    desired long-only signal (1.0 = take a position, 0.0 = flat) determined
    using data up to and including that bar's close. The engine assumes
    execution happens at the NEXT bar's open.
    """

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class SMACrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat otherwise.

    Long-only. Each ticker is treated independently. Bars without enough
    history to compute both SMAs produce a 0 signal.
    """

    fast: int = 50
    slow: int = 200

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")
        if self.fast < 1:
            raise ValueError(f"fast must be >= 1, got {self.fast}")

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast = prices.rolling(self.fast, min_periods=self.fast).mean()
        slow = prices.rolling(self.slow, min_periods=self.slow).mean()
        return (fast > slow).astype(float)
