"""Tests for input validation in forecast() and FLAIR class."""

import numpy as np
import pytest

from flaircast import FLAIR, forecast


class TestForecastValidation:
    """Validation errors from forecast()."""

    def test_horizon_zero_raises(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            forecast(np.ones(50), horizon=0, freq="D")

    def test_horizon_negative_raises(self):
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            forecast(np.ones(50), horizon=-1, freq="D")

    def test_horizon_float_raises(self):
        with pytest.raises(TypeError, match="horizon must be an integer"):
            forecast(np.ones(50), horizon=1.5, freq="D")  # type: ignore[arg-type]

    def test_freq_not_string_raises(self):
        with pytest.raises(TypeError, match="freq must be a string"):
            forecast(np.ones(50), horizon=7, freq=123)  # type: ignore[arg-type]

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="y must not be empty"):
            forecast(np.array([]), horizon=7, freq="D")

    def test_2d_array_raises(self):
        with pytest.raises(ValueError, match="y must be 1-dimensional"):
            forecast(np.ones((10, 2)), horizon=7, freq="D")

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            forecast(np.ones(50), horizon=7, freq="D", n_samples=0)

    def test_n_samples_negative_raises(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            forecast(np.ones(50), horizon=7, freq="D", n_samples=-5)

    def test_n_samples_float_raises(self):
        with pytest.raises(TypeError, match="n_samples must be an integer"):
            forecast(np.ones(50), horizon=7, freq="D", n_samples=1.5)  # type: ignore[arg-type]


class TestFLAIRValidation:
    """Validation errors from FLAIR class."""

    def test_freq_not_string_raises(self):
        with pytest.raises(TypeError, match="freq must be a string"):
            FLAIR(freq=42)  # type: ignore[arg-type]

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            FLAIR(freq="H", n_samples=0)

    def test_n_samples_float_raises(self):
        with pytest.raises(TypeError, match="n_samples must be an integer"):
            FLAIR(freq="H", n_samples=1.5)  # type: ignore[arg-type]
