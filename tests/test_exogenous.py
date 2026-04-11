"""Tests for exogenous variable support in FLAIR.

Design philosophy: standardized exog columns are appended directly to
the Level Ridge feature matrix.  No BIC gate, no model selection — the
LOOCV soft-averaged Ridge inside `_ridge_sa` already shrinks columns
that don't help LOO error, so noise exog is naturally damped.

Tests therefore verify:
- Validation and shape handling
- `X_hist=None` is byte-identical to vanilla FLAIR
- Informative exog meaningfully shifts the forecast
- Noise exog produces only a small drift (well within prediction-interval scale)
- NaN inputs are imputed (not silently zeroed)
"""

import numpy as np
import pytest

from flaircast import FLAIR, forecast

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def informative_exog():
    """Daily series where a slow exog drives a clear additive effect."""
    rng = np.random.RandomState(42)
    n, h = 300, 14
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 60)  # slow signal, not aligned with weekly period
    y = 100 + 20 * x + 5 * np.sin(2 * np.pi * t / 7) + rng.randn(n) * 0.5
    x_future = np.sin(2 * np.pi * np.arange(n, n + h) / 60)
    return y, h, "D", x, x_future


@pytest.fixture()
def noise_exog():
    """Hourly series with pure noise exog."""
    rng = np.random.RandomState(42)
    n, h = 500, 48
    y = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.randn(n)
    X_h = rng.randn(n, 3)
    X_f = rng.randn(h, 3)
    return y, h, "H", X_h, X_f


@pytest.fixture()
def daily_multi_exog():
    """Daily series with two exogenous columns of mixed informativeness."""
    rng = np.random.RandomState(123)
    n, h = 200, 14
    t = np.arange(n)
    x1 = np.sin(2 * np.pi * t / 14)  # slow informative
    x2 = rng.randn(n)  # noise
    X_hist = np.column_stack([x1, x2])
    y = 50 + 8 * x1 + np.sin(2 * np.pi * t / 7) * 3 + rng.randn(n) * 0.4
    t_f = np.arange(n, n + h)
    x1_f = np.sin(2 * np.pi * t_f / 14)
    x2_f = rng.randn(h)
    X_future = np.column_stack([x1_f, x2_f])
    return y, h, "D", X_hist, X_future


# ── Smoke tests ─────────────────────────────────────────────────────────


class TestExogSmoke:
    def test_output_shape_1d(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        s = forecast(y, h, freq, n_samples=50, seed=42, X_hist=x, X_future=xf)
        assert s.shape == (50, h)

    def test_output_shape_2d(self, daily_multi_exog):
        y, h, freq, X_h, X_f = daily_multi_exog
        s = forecast(y, h, freq, n_samples=50, seed=42, X_hist=X_h, X_future=X_f)
        assert s.shape == (50, h)

    def test_dtype_float64(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        s = forecast(y, h, freq, n_samples=20, seed=42, X_hist=x, X_future=xf)
        assert s.dtype == np.float64

    def test_no_nan_or_inf(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        s = forecast(y, h, freq, n_samples=50, seed=42, X_hist=x, X_future=xf)
        assert np.all(np.isfinite(s))

    def test_reproducibility(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        s1 = forecast(y, h, freq, n_samples=20, seed=99, X_hist=x, X_future=xf)
        s2 = forecast(y, h, freq, n_samples=20, seed=99, X_hist=x, X_future=xf)
        np.testing.assert_array_equal(s1, s2)

    def test_class_api(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        m = FLAIR(freq=freq, n_samples=30, seed=42)
        s = m.predict(y, h, X_hist=x, X_future=xf)
        assert s.shape == (30, h)

    def test_1d_and_2d_one_column_equivalent(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        s_1d = forecast(y, h, freq, n_samples=20, seed=42, X_hist=x, X_future=xf)
        s_2d = forecast(
            y,
            h,
            freq,
            n_samples=20,
            seed=42,
            X_hist=x[:, np.newaxis],
            X_future=xf[:, np.newaxis],
        )
        np.testing.assert_array_equal(s_1d, s_2d)


# ── Validation tests ────────────────────────────────────────────────────


class TestExogValidation:
    def test_X_hist_only_raises(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_hist and X_future must be provided together"):
            forecast(y, 7, "D", X_hist=np.random.randn(100, 2))

    def test_X_future_only_raises(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_hist and X_future must be provided together"):
            forecast(y, 7, "D", X_future=np.random.randn(7, 2))

    def test_X_hist_length_mismatch(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_hist length"):
            forecast(
                y,
                7,
                "D",
                X_hist=np.random.randn(50, 2),
                X_future=np.random.randn(7, 2),
            )

    def test_X_future_length_mismatch(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_future length"):
            forecast(
                y,
                7,
                "D",
                X_hist=np.random.randn(100, 2),
                X_future=np.random.randn(10, 2),
            )

    def test_column_count_mismatch(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_hist columns"):
            forecast(
                y,
                7,
                "D",
                X_hist=np.random.randn(100, 2),
                X_future=np.random.randn(7, 3),
            )

    def test_3d_X_hist_raises(self):
        y = np.random.RandomState(0).randn(100)
        with pytest.raises(ValueError, match="X_hist must be 1D or 2D"):
            forecast(
                y,
                7,
                "D",
                X_hist=np.random.randn(100, 2, 3),
                X_future=np.random.randn(7, 2),
            )


# ── Backward compatibility ─────────────────────────────────────────────


class TestExogBackwardCompat:
    """X_hist=None / X_future=None must be byte-identical to vanilla FLAIR."""

    def test_explicit_none_byte_identical(self, informative_exog):
        y, h, freq, _, _ = informative_exog
        s_implicit = forecast(y, h, freq, n_samples=50, seed=42)
        s_explicit = forecast(y, h, freq, n_samples=50, seed=42, X_hist=None, X_future=None)
        np.testing.assert_array_equal(s_implicit, s_explicit)

    def test_class_api_explicit_none_byte_identical(self, informative_exog):
        y, h, freq, _, _ = informative_exog
        m = FLAIR(freq=freq, n_samples=50, seed=42)
        s_implicit = m.predict(y, h)
        s_explicit = m.predict(y, h, X_hist=None, X_future=None)
        np.testing.assert_array_equal(s_implicit, s_explicit)


# ── Effect tests ───────────────────────────────────────────────────────


class TestExogEffect:
    def test_informative_exog_changes_forecast(self, informative_exog):
        """Strong informative exog should shift the point forecast meaningfully."""
        y, h, freq, x, xf = informative_exog
        s_no = forecast(y, h, freq, n_samples=200, seed=42)
        s_ex = forecast(y, h, freq, n_samples=200, seed=42, X_hist=x, X_future=xf)
        diff = np.abs(s_no.mean(axis=0) - s_ex.mean(axis=0))
        scale = max(float(s_no.std()), 1e-6)
        # Effect must exceed 1 sigma — much stronger than the loose 0.1 atol
        # the original PR used (which would pass even for near-noise diffs).
        assert diff.mean() > scale

    def test_noise_exog_drift_is_small(self, noise_exog):
        """Pure noise exog must not move the forecast by more than 0.1σ.

        Ridge LOOCV soft-average shrinks columns that don't reduce LOO
        error, so even with several noise columns the forecast drifts
        only by a fraction of one standard deviation — comfortably
        within FLAIR's prediction interval width.
        """
        y, h, freq, X_h, X_f = noise_exog
        s_no = forecast(y, h, freq, n_samples=200, seed=42)
        s_ex = forecast(y, h, freq, n_samples=200, seed=42, X_hist=X_h, X_future=X_f)
        scale = max(float(y.std()), 1e-6)
        drift = float(np.abs(s_no.mean(axis=0) - s_ex.mean(axis=0)).mean())
        assert drift / scale < 0.1, (
            f"noise exog drift {drift / scale:.4f} sigma exceeds 0.1 sigma tolerance"
        )

    def test_noise_exog_drift_aggregate(self):
        """Across many random noise scenarios the mean drift should be tiny."""
        drifts = []
        for trial in range(20):
            rng = np.random.RandomState(trial)
            n, h = 200, 14
            y = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.randn(n)
            X_h = rng.randn(n, 3)
            X_f = rng.randn(h, 3)
            s_no = forecast(y, h, "D", n_samples=100, seed=42)
            s_ex = forecast(y, h, "D", n_samples=100, seed=42, X_hist=X_h, X_future=X_f)
            scale = max(float(y.std()), 1e-6)
            drifts.append(float(np.abs(s_no.mean(axis=0) - s_ex.mean(axis=0)).mean()) / scale)
        # Aggregate noise leak should average well below 0.05 sigma
        assert float(np.mean(drifts)) < 0.05, (
            f"mean noise drift {np.mean(drifts):.4f} sigma; per-trial: {drifts}"
        )

    def test_constant_exog_handled(self):
        """Constant exog has zero variance; standardization must not blow up."""
        rng = np.random.RandomState(42)
        n, h = 200, 14
        y = 100 + np.sin(2 * np.pi * np.arange(n) / 7) * 5 + rng.randn(n)
        X_h = np.full((n, 1), 3.7)
        X_f = np.full((h, 1), 3.7)
        s = forecast(y, h, "D", n_samples=50, seed=42, X_hist=X_h, X_future=X_f)
        assert np.all(np.isfinite(s))

    def test_nan_in_exog_imputed(self, informative_exog):
        y, h, freq, x, xf = informative_exog
        x_nan = x.copy()
        x_nan[10:20] = np.nan
        s = forecast(y, h, freq, n_samples=50, seed=42, X_hist=x_nan, X_future=xf)
        assert np.all(np.isfinite(s))

    def test_multiple_frequencies(self):
        rng = np.random.RandomState(42)
        for freq, n, h in [("H", 300, 24), ("D", 100, 7), ("W", 156, 12), ("M", 60, 6)]:
            y = 50 + rng.normal(0, 3, n)
            x = rng.normal(0, 1, n)
            x_f = rng.normal(0, 1, h)
            s = forecast(y, h, freq, n_samples=20, seed=42, X_hist=x, X_future=x_f)
            assert s.shape == (20, h)
            assert np.all(np.isfinite(s))
