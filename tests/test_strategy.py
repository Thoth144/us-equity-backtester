import numpy as np
import pandas as pd
import pytest

from equity_backtester.strategy import SMACrossover


def _series(values):
    return pd.DataFrame({"AAA": values}, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_long_in_steady_uptrend():
    prices = _series(np.linspace(100, 200, 300))
    sig = SMACrossover(fast=10, slow=50).generate_signals(prices)
    assert (sig.iloc[60:] == 1.0).all().all()


def test_flat_in_steady_downtrend():
    prices = _series(np.linspace(200, 100, 300))
    sig = SMACrossover(fast=10, slow=50).generate_signals(prices)
    assert (sig.iloc[60:] == 0.0).all().all()


def test_warmup_signal_is_zero():
    prices = _series(np.linspace(100, 200, 300))
    sig = SMACrossover(fast=10, slow=50).generate_signals(prices)
    # Slow SMA needs 50 bars; before that the comparison is NaN -> astype(float) gives 0.0.
    assert (sig.iloc[:49] == 0.0).all().all()


def test_signal_flips_at_crossover_point():
    # Build a series where the fast SMA crosses the slow SMA at a known point.
    rng = np.arange(120, dtype=float)
    prices = _series(100 + rng - np.maximum(0, rng - 60))  # rises, then plateaus
    sig = SMACrossover(fast=5, slow=20).generate_signals(prices)
    # Once flat, fast eventually equals slow then falls below.
    assert sig.iloc[30].iloc[0] == 1.0
    assert sig.iloc[-1].iloc[0] == 0.0


def test_invalid_windows_rejected():
    with pytest.raises(ValueError):
        SMACrossover(fast=50, slow=50)
    with pytest.raises(ValueError):
        SMACrossover(fast=100, slow=50)
    with pytest.raises(ValueError):
        SMACrossover(fast=0, slow=10)
