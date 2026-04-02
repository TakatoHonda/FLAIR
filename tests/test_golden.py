"""Golden snapshot regression tests.

These tests verify that forecast() produces bit-identical output
for known inputs. They serve as the safety net for all refactoring.

Regenerate snapshots with:
    uv run python tests/golden/generate_golden.py
"""

import os

import numpy as np
import pytest

from flaircast import forecast

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")

FIXTURES = [
    "hourly_seasonal",
    "daily_weekly",
    "monthly",
    "weekly",
    "quarterly",
    "short_series",
    "very_short",
    "all_negative",
    "constant",
    "nan_heavy",
]


@pytest.mark.golden
class TestGoldenSnapshots:
    @pytest.mark.parametrize("fixture_name", FIXTURES)
    def test_matches_snapshot(self, fixture_name):
        path = os.path.join(GOLDEN_DIR, f"{fixture_name}.npz")
        if not os.path.exists(path):
            pytest.skip(f"Golden snapshot {path} not found. Run generate_golden.py first.")

        data = np.load(path, allow_pickle=True)
        y = data["y"]
        horizon = int(data["horizon"])
        freq = str(data["freq"])
        expected = data["samples"]

        actual = forecast(y, horizon, freq, n_samples=expected.shape[0], seed=42)

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-10,
            err_msg=f"Golden snapshot mismatch for {fixture_name}",
        )
