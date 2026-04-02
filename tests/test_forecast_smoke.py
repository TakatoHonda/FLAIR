"""Integration smoke tests for flaircast.forecast() and FLAIR class."""

import numpy as np
import pytest

from flaircast import FLAIR, FREQ_TO_PERIOD, forecast


class TestForecastSmoke:
    """Basic smoke tests: correct shape, dtype, and no NaN/Inf."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "hourly_seasonal",
            "daily_weekly",
            "monthly",
            "weekly",
            "quarterly",
        ],
    )
    def test_output_shape(self, fixture_name, request):
        y, horizon, freq = request.getfixturevalue(fixture_name)
        n_samples = 50
        result = forecast(y, horizon, freq, n_samples=n_samples)
        assert result.shape == (n_samples, horizon)

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "hourly_seasonal",
            "daily_weekly",
            "monthly",
            "weekly",
            "quarterly",
        ],
    )
    def test_output_dtype_float(self, fixture_name, request):
        y, horizon, freq = request.getfixturevalue(fixture_name)
        result = forecast(y, horizon, freq, n_samples=10)
        assert result.dtype == np.float64

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "hourly_seasonal",
            "daily_weekly",
            "monthly",
            "weekly",
            "quarterly",
        ],
    )
    def test_no_nan_or_inf(self, fixture_name, request):
        y, horizon, freq = request.getfixturevalue(fixture_name)
        result = forecast(y, horizon, freq, n_samples=20)
        assert np.all(np.isfinite(result))

    def test_point_forecast_reasonable_range(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        result = forecast(y, horizon, freq, n_samples=50)
        point = result.mean(axis=0)
        y_mean = y.mean()
        y_std = y.std()
        # Point forecast should be within 5 std of historical mean
        assert np.all(np.abs(point - y_mean) < 5 * y_std)

    def test_n_samples_1(self, daily_weekly):
        y, horizon, freq = daily_weekly
        result = forecast(y, horizon, freq, n_samples=1)
        assert result.shape == (1, horizon)
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("freq", list(FREQ_TO_PERIOD.keys()))
    def test_all_frequencies_dont_crash(self, freq):
        """Every frequency in FREQ_TO_PERIOD should work with sufficient data."""
        period = FREQ_TO_PERIOD[freq]
        n = max(period * 5, 20)  # At least 5 periods
        rng = np.random.RandomState(42)
        y = 100 + rng.normal(0, 10, n)
        result = forecast(y, horizon=max(period, 1), freq=freq, n_samples=10)
        assert result.shape == (10, max(period, 1))
        assert np.all(np.isfinite(result))


class TestFLAIRClass:
    """Tests for the FLAIR class API."""

    def test_basic_usage(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        model = FLAIR(freq=freq)
        result = model.predict(y, horizon=horizon)
        assert result.shape == (200, horizon)  # default n_samples

    def test_custom_n_samples(self, daily_weekly):
        y, horizon, freq = daily_weekly
        model = FLAIR(freq=freq, n_samples=50)
        result = model.predict(y, horizon=horizon)
        assert result.shape == (50, horizon)

    def test_override_n_samples_in_predict(self, monthly):
        y, horizon, freq = monthly
        model = FLAIR(freq=freq, n_samples=50)
        result = model.predict(y, horizon=horizon, n_samples=10)
        assert result.shape == (10, horizon)

    def test_backward_compat_alias(self, hourly_seasonal):
        """flair_forecast should be an alias for forecast."""
        from flaircast import flair_forecast

        assert flair_forecast is forecast
