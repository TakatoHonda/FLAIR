"""Shared fixtures for flaircast tests."""

import numpy as np
import pytest


@pytest.fixture
def hourly_seasonal():
    """500 points of hourly data with period=24."""
    rng = np.random.RandomState(0)
    t = np.arange(500, dtype=float)
    y = 100 + 20 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 3, 500)
    return y, 48, "H"


@pytest.fixture
def daily_weekly():
    """200 points of daily data with period=7."""
    rng = np.random.RandomState(1)
    t = np.arange(200, dtype=float)
    y = 50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, 200)
    return y, 14, "D"


@pytest.fixture
def monthly():
    """120 points of monthly data with period=12."""
    rng = np.random.RandomState(2)
    t = np.arange(120, dtype=float)
    y = 1000 + 200 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 30, 120)
    return y, 12, "M"


@pytest.fixture
def weekly():
    """156 points (3 years) of weekly data with period=52."""
    rng = np.random.RandomState(3)
    t = np.arange(156, dtype=float)
    y = 500 + 100 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 15, 156)
    return y, 26, "W"


@pytest.fixture
def quarterly():
    """40 points (10 years) of quarterly data with period=4."""
    rng = np.random.RandomState(4)
    t = np.arange(40, dtype=float)
    y = 2000 + 300 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 50, 40)
    return y, 4, "Q"


@pytest.fixture
def short_series():
    """5 points of daily data (triggers P=1 degeneration)."""
    return np.array([10.0, 12.0, 11.0, 13.0, 12.5]), 7, "D"


@pytest.fixture
def very_short_series():
    """2 points of monthly data (triggers n_complete<3 early return)."""
    return np.array([100.0, 110.0]), 3, "M"


@pytest.fixture
def all_negative():
    """100 points of negative daily data."""
    rng = np.random.RandomState(5)
    y = -30 + rng.normal(0, 5, 100)
    return y, 7, "D"


@pytest.fixture
def constant_series():
    """100 points all equal to 42.0."""
    return np.full(100, 42.0), 24, "H"


@pytest.fixture
def nan_heavy():
    """100 points with 30% NaN."""
    rng = np.random.RandomState(6)
    y = 50 + 10 * np.sin(2 * np.pi * np.arange(100) / 7) + rng.normal(0, 2, 100)
    mask = rng.random(100) < 0.3
    y[mask] = np.nan
    return y, 7, "D"
