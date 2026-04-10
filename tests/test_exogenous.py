"""Tests for exogenous variable support in FLAIR."""
import numpy as np
import pytest

from flaircast import FLAIR, forecast


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def hourly_with_exog():
    """Hourly series where exogenous variable drives Level (multi-day trend)."""
    rng = np.random.RandomState(42)
    n = 500
    horizon = 48
    t = np.arange(n)
    # Exogenous: slow-moving signal (period ~120h) that affects Level, not Shape
    exog = 5 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.5, n)
    # y has daily seasonality + strong exogenous Level effect
    seasonal = 10 * np.sin(2 * np.pi * t / 24)
    y = 100 + 3.0 * exog + seasonal + rng.normal(0, 1, n)
    # Future exogenous: shift phase to create clearly different forecast
    t_future = np.arange(n, n + horizon)
    exog_future = 5 * np.sin(2 * np.pi * t_future / 120) + rng.normal(0, 0.5, horizon)
    return y, horizon, "H", exog, exog_future


@pytest.fixture()
def daily_multi_exog():
    """Daily series with multiple exogenous variables."""
    rng = np.random.RandomState(123)
    n = 200
    horizon = 14
    t = np.arange(n)
    x1 = np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, n)
    x2 = rng.normal(5, 1, n)
    X_hist = np.column_stack([x1, x2])
    y = 50 + 2 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
    t_f = np.arange(n, n + horizon)
    x1_f = np.sin(2 * np.pi * t_f / 7) + rng.normal(0, 0.3, horizon)
    x2_f = rng.normal(5, 1, horizon)
    X_future = np.column_stack([x1_f, x2_f])
    return y, horizon, "D", X_hist, X_future


# ── Smoke tests ─────────────────────────────────────────────────────────


class TestExogSmoke:
    def test_output_shape_1d_exog(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        samples = forecast(y, horizon, freq, n_samples=50, seed=42,
                           X_hist=temp, X_future=temp_f)
        assert samples.shape == (50, horizon)

    def test_output_shape_2d_exog(self, daily_multi_exog):
        y, horizon, freq, X_h, X_f = daily_multi_exog
        samples = forecast(y, horizon, freq, n_samples=50, seed=42,
                           X_hist=X_h, X_future=X_f)
        assert samples.shape == (50, horizon)

    def test_dtype_float64(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        samples = forecast(y, horizon, freq, n_samples=20, seed=42,
                           X_hist=temp, X_future=temp_f)
        assert samples.dtype == np.float64

    def test_no_nan_or_inf(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        samples = forecast(y, horizon, freq, n_samples=50, seed=42,
                           X_hist=temp, X_future=temp_f)
        assert np.all(np.isfinite(samples))

    def test_class_api_with_exog(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        model = FLAIR(freq=freq, n_samples=30, seed=42)
        samples = model.predict(y, horizon, X_hist=temp, X_future=temp_f)
        assert samples.shape == (30, horizon)

    def test_reproducibility_with_exog(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        s1 = forecast(y, horizon, freq, n_samples=20, seed=99,
                      X_hist=temp, X_future=temp_f)
        s2 = forecast(y, horizon, freq, n_samples=20, seed=99,
                      X_hist=temp, X_future=temp_f)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seed_different_result(self, hourly_with_exog):
        y, horizon, freq, temp, temp_f = hourly_with_exog
        s1 = forecast(y, horizon, freq, n_samples=20, seed=1,
                      X_hist=temp, X_future=temp_f)
        s2 = forecast(y, horizon, freq, n_samples=20, seed=2,
                      X_hist=temp, X_future=temp_f)
        assert not np.array_equal(s1, s2)


# ── Validation tests ────────────────────────────────────────────────────


class TestExogValidation:
    def test_X_hist_only_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_hist and X_future must be provided together"):
            forecast(y, 7, "D", X_hist=np.random.randn(100, 2))

    def test_X_future_only_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_hist and X_future must be provided together"):
            forecast(y, 7, "D", X_future=np.random.randn(7, 2))

    def test_X_hist_length_mismatch_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_hist length"):
            forecast(y, 7, "D", X_hist=np.random.randn(50, 2),
                     X_future=np.random.randn(7, 2))

    def test_X_future_length_mismatch_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_future length"):
            forecast(y, 7, "D", X_hist=np.random.randn(100, 2),
                     X_future=np.random.randn(10, 2))

    def test_column_mismatch_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_hist columns"):
            forecast(y, 7, "D", X_hist=np.random.randn(100, 2),
                     X_future=np.random.randn(7, 3))

    def test_3d_X_hist_raises(self):
        y = np.random.randn(100)
        with pytest.raises(ValueError, match="X_hist must be 1D or 2D"):
            forecast(y, 7, "D", X_hist=np.random.randn(100, 2, 3),
                     X_future=np.random.randn(7, 2))


# ── Effect tests ────────────────────────────────────────────────────────


class TestExogEffect:
    def test_informative_exog_changes_forecast(self, hourly_with_exog):
        """Forecast with informative exogenous should differ from without."""
        y, horizon, freq, temp, temp_f = hourly_with_exog
        s_no = forecast(y, horizon, freq, n_samples=100, seed=42)
        s_ex = forecast(y, horizon, freq, n_samples=100, seed=42,
                        X_hist=temp, X_future=temp_f)
        # Point forecasts should differ
        assert not np.allclose(s_no.mean(axis=0), s_ex.mean(axis=0), atol=0.1)

    def test_zero_exog_similar_to_no_exog(self):
        """Zero-valued exogenous should produce near-identical results."""
        rng = np.random.RandomState(42)
        n, horizon = 200, 14
        y = 100 + rng.normal(0, 5, n)
        X_h = np.zeros((n, 2))
        X_f = np.zeros((horizon, 2))
        s_no = forecast(y, horizon, "D", n_samples=100, seed=42)
        s_ex = forecast(y, horizon, "D", n_samples=100, seed=42,
                        X_hist=X_h, X_future=X_f)
        # Should be very close (Ridge shrinks zero-effect columns to ~0)
        np.testing.assert_allclose(
            s_no.mean(axis=0), s_ex.mean(axis=0), rtol=0.05
        )

    def test_multiple_frequencies_with_exog(self):
        """Exogenous should work across different frequencies."""
        rng = np.random.RandomState(42)
        for freq, n, horizon in [("H", 300, 24), ("D", 100, 7),
                                  ("W", 156, 12), ("M", 60, 6)]:
            y = 50 + rng.normal(0, 3, n)
            x = rng.normal(0, 1, n)
            x_f = rng.normal(0, 1, horizon)
            samples = forecast(y, horizon, freq, n_samples=20, seed=42,
                               X_hist=x, X_future=x_f)
            assert samples.shape == (20, horizon)
            assert np.all(np.isfinite(samples))
