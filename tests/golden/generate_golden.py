#!/usr/bin/env python3
"""Generate golden snapshot files for regression testing.

Usage:
    uv run python tests/golden/generate_golden.py

Produces .npz files in tests/golden/ that serve as the regression oracle.
Each snapshot is generated with np.random.seed(42) for determinism.
"""

import os
import sys

import numpy as np

# Ensure flaircast is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from flaircast import forecast

GOLDEN_DIR = os.path.dirname(__file__)


def _make_fixtures():
    """Return dict of fixture_name -> (y, horizon, freq)."""
    rng0 = np.random.RandomState(0)
    rng1 = np.random.RandomState(1)
    rng2 = np.random.RandomState(2)
    rng3 = np.random.RandomState(3)
    rng4 = np.random.RandomState(4)
    rng5 = np.random.RandomState(5)
    rng6 = np.random.RandomState(6)

    t500 = np.arange(500, dtype=float)
    t200 = np.arange(200, dtype=float)
    t120 = np.arange(120, dtype=float)
    t156 = np.arange(156, dtype=float)
    t40 = np.arange(40, dtype=float)
    t100 = np.arange(100, dtype=float)

    fixtures = {
        "hourly_seasonal": (
            100 + 20 * np.sin(2 * np.pi * t500 / 24) + rng0.normal(0, 3, 500),
            48,
            "H",
        ),
        "daily_weekly": (
            50 + 10 * np.sin(2 * np.pi * t200 / 7) + rng1.normal(0, 2, 200),
            14,
            "D",
        ),
        "monthly": (
            1000 + 200 * np.sin(2 * np.pi * t120 / 12) + rng2.normal(0, 30, 120),
            12,
            "M",
        ),
        "weekly": (
            500 + 100 * np.sin(2 * np.pi * t156 / 52) + rng3.normal(0, 15, 156),
            26,
            "W",
        ),
        "quarterly": (
            2000 + 300 * np.sin(2 * np.pi * t40 / 4) + rng4.normal(0, 50, 40),
            4,
            "Q",
        ),
        "short_series": (
            np.array([10.0, 12.0, 11.0, 13.0, 12.5]),
            7,
            "D",
        ),
        "very_short": (
            np.array([100.0, 110.0]),
            3,
            "M",
        ),
        "all_negative": (
            -30 + rng5.normal(0, 5, 100),
            7,
            "D",
        ),
        "constant": (
            np.full(100, 42.0),
            24,
            "H",
        ),
        "nan_heavy": (
            _make_nan_heavy(t100, rng6),
            7,
            "D",
        ),
    }
    return fixtures


def _make_nan_heavy(t, rng):
    y = 50 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, len(t))
    mask = rng.random(len(t)) < 0.3
    y[mask] = np.nan
    return y


def main():
    fixtures = _make_fixtures()
    for name, (y, horizon, freq) in fixtures.items():
        samples = forecast(y, horizon, freq, n_samples=50, seed=42)
        path = os.path.join(GOLDEN_DIR, f"{name}.npz")
        np.savez_compressed(
            path,
            y=y,
            horizon=np.array(horizon),
            freq=np.array(freq),
            samples=samples,
        )
        print(f"  saved {name}.npz  shape={samples.shape}")

    print(f"\nGenerated {len(fixtures)} golden snapshots in {GOLDEN_DIR}/")


if __name__ == "__main__":
    main()
