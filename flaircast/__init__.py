"""FLAIR: Factored Level And Interleaved Ridge.

A single-equation time series forecasting method that separates
WHAT (level) from HOW (shape) via period-aligned matrix reshaping.

    y(phase, period) = Level(period) × Shape(phase)

Level is forecast by Ridge regression with soft-average GCV.
Shape is estimated via Dirichlet-Multinomial empirical Bayes, with
context derived from the secondary period structure (e.g., day-of-week
for hourly data).  When no secondary period exists, degenerates to
the simple K-period average.

Secondary periodicity in Level is handled by Shape₂ deseasonalization:
the proportional decomposition is applied recursively, with the raw
Shape₂ estimate shrunk toward a BIC-selected prior (first harmonic
or flat) via empirical Bayes — the same MDL principle used for
primary period selection.

When data has fewer than 3 complete periods, FLAIR degenerates to
P=1 (Ridge on raw series) — no separate fallback model needed.

Prediction intervals via SVD Residual Quantiles: the residual matrix
E = M − σ₁u₁v₁ᵀ gives phase-specific uncertainty, combined with
Ridge LOO residuals scaled by √(recursive step) for horizon fan-out.

One SVD. Zero hyperparameters. No neural network. One code path.

Module map (paper section ↔ file):

- `_constants`  — numerical fudge factors and feature-flag toggles
- `_frequency`  — calendar tables and frequency-string resolution
- `_period`     — MDL period selection (BIC on the SVD spectrum)
- `_shape`      — Shape₁ (Dirichlet-Multinomial EB) and Shape₂ (BIC-gated)
- `_level`      — Box-Cox, Ridge with LOOCV soft-average + LWCP, damped trend
- `_forecast`   — `forecast()` orchestrator and `FLAIR` class wrapper

Example:
    >>> from flaircast import forecast
    >>> samples = forecast(y, horizon=24, freq='H')
    >>> point = samples.mean(axis=0)

    >>> from flaircast import FLAIR
    >>> model = FLAIR(freq='H')
    >>> samples = model.predict(y, horizon=24)
"""

from __future__ import annotations

__version__ = "0.5.0"
__all__ = ["FLAIR", "FREQ_TO_PERIOD", "FREQ_TO_PERIODS", "forecast"]

# Public API.
from ._forecast import FLAIR, forecast
from ._frequency import FREQ_TO_PERIOD, FREQ_TO_PERIODS

# Private helpers re-exported at the package root for backward
# compatibility with existing tests and external callers that imported
# them from `flaircast` directly before the file split.  The redundant
# `as <name>` aliases are PEP 484 explicit re-exports — they tell ruff
# (and mypy) that these imports are intentional and not dead code.
from ._frequency import _get_period as _get_period
from ._frequency import _get_periods as _get_periods
from ._frequency import _resolve_freq as _resolve_freq
from ._level import _bc as _bc
from ._level import _bc_inv as _bc_inv
from ._level import _bc_lambda as _bc_lambda
from ._level import _estimate_phi as _estimate_phi
from ._level import _ridge_sa as _ridge_sa
from ._period import _select_period as _select_period
from ._shape import _compute_cross_periods as _compute_cross_periods
from ._shape import _compute_shape2 as _compute_shape2
from ._shape import _estimate_shape as _estimate_shape

# Backward compatibility alias.
flair_forecast = forecast
