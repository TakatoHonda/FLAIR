"""Tests for reproducibility via seed parameter."""

import numpy as np

from flaircast import FLAIR, forecast


class TestReproducibility:
    def test_same_seed_same_output(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        a = forecast(y, horizon, freq, n_samples=20, seed=42)
        b = forecast(y, horizon, freq, n_samples=20, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_output(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        a = forecast(y, horizon, freq, n_samples=20, seed=42)
        b = forecast(y, horizon, freq, n_samples=20, seed=99)
        assert not np.array_equal(a, b)

    def test_flair_class_seed(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        model = FLAIR(freq=freq, seed=42)
        a = model.predict(y, horizon=horizon, n_samples=20)
        b = forecast(y, horizon, freq, n_samples=20, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_predict_seed_override(self, hourly_seasonal):
        y, horizon, freq = hourly_seasonal
        model = FLAIR(freq=freq, seed=99)
        a = model.predict(y, horizon=horizon, n_samples=20, seed=42)
        b = forecast(y, horizon, freq, n_samples=20, seed=42)
        np.testing.assert_array_equal(a, b)
