"""Edge case tests for flaircast.forecast()."""

import numpy as np

from flaircast import forecast


class TestEdgeCases:
    def test_short_series_p1_degeneration(self, short_series):
        """5 points with period=7 -> P=1 degeneration."""
        y, horizon, freq = short_series
        result = forecast(y, horizon, freq, n_samples=20)
        assert result.shape == (20, horizon)
        assert np.all(np.isfinite(result))

    def test_very_short_series_early_return(self, very_short_series):
        """2 points -> n_complete<3 early return path."""
        y, horizon, freq = very_short_series
        result = forecast(y, horizon, freq, n_samples=20)
        assert result.shape == (20, horizon)
        assert np.all(np.isfinite(result))

    def test_all_nan_input(self):
        """All NaN -> converted to zeros, should not crash."""
        y = np.full(100, np.nan)
        result = forecast(y, 7, "D", n_samples=10)
        assert result.shape == (10, 7)
        assert np.all(np.isfinite(result))

    def test_all_negative_values(self, all_negative):
        """Negative values handled by location shift."""
        y, horizon, freq = all_negative
        result = forecast(y, horizon, freq, n_samples=20)
        assert result.shape == (20, horizon)
        assert np.all(np.isfinite(result))

    def test_constant_series(self, constant_series):
        """All identical values -> near-constant forecast."""
        y, horizon, freq = constant_series
        result = forecast(y, horizon, freq, n_samples=20)
        assert result.shape == (20, horizon)
        assert np.all(np.isfinite(result))
        point = result.mean(axis=0)
        # Forecast should be in the ballpark of the constant value
        assert np.all(np.abs(point - 42.0) < 50.0)

    def test_nan_heavy_input(self, nan_heavy):
        """30% NaN -> nan_to_num fills zeros."""
        y, horizon, freq = nan_heavy
        result = forecast(y, horizon, freq, n_samples=20)
        assert result.shape == (20, horizon)
        assert np.all(np.isfinite(result))

    def test_large_values(self):
        """Very large values should not overflow."""
        rng = np.random.RandomState(42)
        y = 1e12 + rng.normal(0, 1e10, 100)
        result = forecast(y, 7, "D", n_samples=10)
        assert result.shape == (10, 7)
        assert np.all(np.isfinite(result))

    def test_mixed_sign_values(self):
        """Series crossing zero handled correctly."""
        rng = np.random.RandomState(42)
        t = np.arange(100, dtype=float)
        y = 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, 100)
        result = forecast(y, 14, "D", n_samples=10)
        assert result.shape == (10, 14)
        assert np.all(np.isfinite(result))

    def test_large_horizon(self):
        """Large horizon relative to series length."""
        rng = np.random.RandomState(42)
        y = 100 + rng.normal(0, 5, 50)
        result = forecast(y, 200, "D", n_samples=10)
        assert result.shape == (10, 200)
        assert np.all(np.isfinite(result))

    def test_three_points_minimum(self):
        """Exactly 3 points with P=1 -> minimum for non-degenerate."""
        y = np.array([10.0, 20.0, 15.0])
        result = forecast(y, 5, "Y", n_samples=10)
        assert result.shape == (10, 5)
        assert np.all(np.isfinite(result))

    def test_list_input(self):
        """Accept plain Python list as input."""
        y = [10.0, 20.0, 15.0, 25.0, 18.0, 30.0, 22.0] * 5
        result = forecast(y, 7, "D", n_samples=10)
        assert result.shape == (10, 7)
        assert np.all(np.isfinite(result))
