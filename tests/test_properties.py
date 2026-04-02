"""Property-based tests using Hypothesis."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from flaircast import FREQ_TO_PERIOD, forecast

# Keep tests fast: small arrays, few examples
_y_elements = st.one_of(
    st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)
_y_strategy = arrays(
    dtype=np.float64,
    shape=st.integers(min_value=3, max_value=200),
    elements=_y_elements,
)
_freq_strategy = st.sampled_from(list(FREQ_TO_PERIOD.keys()))


class TestProperties:
    @given(y=_y_strategy, freq=_freq_strategy)
    @settings(max_examples=30, deadline=30000)
    def test_output_shape_always_correct(self, y, freq):
        horizon = max(1, FREQ_TO_PERIOD[freq])
        n_samples = 5
        result = forecast(y, horizon, freq, n_samples=n_samples)
        assert result.shape == (n_samples, horizon)

    @given(y=_y_strategy, freq=_freq_strategy)
    @settings(max_examples=30, deadline=30000)
    def test_no_nan_in_output(self, y, freq):
        horizon = max(1, FREQ_TO_PERIOD[freq])
        result = forecast(y, horizon, freq, n_samples=5)
        assert not np.any(np.isnan(result))

    @given(y=_y_strategy, freq=_freq_strategy)
    @settings(max_examples=30, deadline=30000)
    def test_no_inf_in_output(self, y, freq):
        horizon = max(1, FREQ_TO_PERIOD[freq])
        result = forecast(y, horizon, freq, n_samples=5)
        assert not np.any(np.isinf(result))

    @given(y=_y_strategy, freq=_freq_strategy)
    @settings(max_examples=30, deadline=30000)
    def test_dtype_is_float64(self, y, freq):
        horizon = max(1, FREQ_TO_PERIOD[freq])
        result = forecast(y, horizon, freq, n_samples=5)
        assert result.dtype == np.float64
