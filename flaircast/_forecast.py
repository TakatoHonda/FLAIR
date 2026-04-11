"""Main forecast pipeline.

Public entry points:

- :func:`forecast` — functional API used by both notebooks and the
  :class:`FLAIR` wrapper.
- :class:`FLAIR` — class wrapper that holds frequency / sample-count /
  seed defaults so users can call ``model.predict(y, horizon)`` in a
  loop without re-passing them.

Internally, ``forecast()`` is a thin orchestrator over a sequence of
single-purpose helpers, each of which corresponds to one step of
Algorithm 1 in the paper:

    1. ``_validate_inputs``           — type / shape / value checks
    2. ``_validate_and_clean_exog``   — exog shape checks + NaN impute
    3. ``_preprocess_y``              — y NaN interp + positivity shift
    4. ``_degenerate_p1_fallback``    — short-series Gaussian fallback
    5. ``_truncate_to_max_complete``  — cap on memory / runtime
    6. ``_aggregate_exog_to_level``   — period-mean aggregation of X
    7. ``_apply_shape2_deseason``     — Shape₂ deseasonalization of L
    8. ``_build_level_design``        — Ridge feature matrix construction
    9. ``_compute_damped_trend``      — LSR1 boundary extrapolation
    10. ``_compute_lwcp_leverages``   — per-horizon test-point leverage
    11. ``_sample_level_paths``       — Student-t noise + recursive paths
    12. ``_phase_noise_sample``       — SVD residual column sampling
    13. ``_assemble_and_calibrate``   — L × S × (1+ε) + clip + shrinkage
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._constants import (
    _DIFF_TARGET,
    _EPS,
    _EPS_BOXCOX,
    _EPS_SHAPE,
    _MAX_COMPLETE,
    _MIN_COMPLETE,
    _PHASE_NOISE_K,
)
from ._level import _bc, _bc_inv, _bc_lambda, _estimate_phi, _ridge_sa
from ._period import _select_period
from ._shape import _compute_cross_periods, _compute_shape2, _estimate_shape

# ── Input validation ───────────────────────────────────────────────────


def _validate_inputs(
    y_raw: ArrayLike,
    horizon: int,
    freq: str,
    n_samples: int,
) -> NDArray[np.floating]:
    """Type-check the public arguments and return ``y`` as a 1-D float array."""
    if not isinstance(freq, str):
        raise TypeError(f"freq must be a string, got {type(freq).__name__}")
    if not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"horizon must be an integer, got {type(horizon).__name__}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    y_arr = np.asarray(y_raw, dtype=float)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape}")
    if len(y_arr) == 0:
        raise ValueError("y must not be empty")
    return y_arr


def _interp_nan_1d(
    arr: NDArray[np.floating],
    all_nan_fallback: float = 0.0,
) -> NDArray[np.floating]:
    """Linear-interpolate NaNs in a 1-D array.

    Tiered fallback (the same policy used by both ``y`` preprocessing
    and per-column exog cleanup):

    - 2+ valid values  →  linear interpolation against the index axis
    - 1 valid value    →  fill with that single value
    - 0 valid values   →  fill with ``all_nan_fallback``

    The ``all_nan_fallback`` parameter is the only thing that varies
    between consumers: ``y`` and ``X_hist`` use ``0.0`` (no useful
    boundary information), while ``X_future`` columns use the last
    valid ``X_hist`` value of the same column (one-step persistence).
    """
    out = arr.copy()
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out
    valid = ~nan_mask
    n_valid = int(valid.sum())
    if n_valid >= 2:
        idx = np.arange(len(out))
        out[nan_mask] = np.interp(idx[nan_mask], idx[valid], out[valid])
    elif n_valid == 1:
        out[nan_mask] = out[valid][0]
    else:
        out[:] = all_nan_fallback
    return out


def _validate_and_clean_exog(
    X_hist: ArrayLike | None,
    X_future: ArrayLike | None,
    n: int,
    horizon: int,
) -> tuple[NDArray[np.floating] | None, NDArray[np.floating] | None, int]:
    """Validate exogenous arrays and interpolate NaNs column by column.

    Returns ``(X_hist_arr, X_future_arr, n_exog)``.  When neither array
    is provided, returns ``(None, None, 0)``.

    NaN policy (column-wise, mirrors the y-side ``_interp_nan_1d``):

    - For each column of ``X_hist`` we linear-interpolate over the
      column index axis (matching ``_preprocess_y``).
    - For each column of ``X_future`` we likewise interpolate when the
      column has at least one valid value, but if the entire future
      column is NaN we forward-fill with the *last valid X_hist value*
      of that column rather than 0 — a one-step persistence assumption,
      which is more physical than zeroing an unknown covariate.
    """
    has_exog = X_hist is not None or X_future is not None
    if not has_exog:
        return None, None, 0

    if X_hist is None or X_future is None:
        raise ValueError("X_hist and X_future must be provided together")

    X_hist_arr = np.asarray(X_hist, dtype=float).copy()
    X_future_arr = np.asarray(X_future, dtype=float).copy()
    if X_hist_arr.ndim == 1:
        X_hist_arr = X_hist_arr[:, np.newaxis]
    if X_future_arr.ndim == 1:
        X_future_arr = X_future_arr[:, np.newaxis]
    if X_hist_arr.ndim != 2:
        raise ValueError(f"X_hist must be 1D or 2D, got shape {X_hist_arr.shape}")
    if X_future_arr.ndim != 2:
        raise ValueError(f"X_future must be 1D or 2D, got shape {X_future_arr.shape}")
    if X_hist_arr.shape[0] != n:
        raise ValueError(f"X_hist length ({X_hist_arr.shape[0]}) must match y length ({n})")
    if X_future_arr.shape[0] != horizon:
        raise ValueError(
            f"X_future length ({X_future_arr.shape[0]}) must match horizon ({horizon})"
        )
    if X_hist_arr.shape[1] != X_future_arr.shape[1]:
        raise ValueError(
            f"X_hist columns ({X_hist_arr.shape[1]}) must match "
            f"X_future columns ({X_future_arr.shape[1]})"
        )

    n_exog = X_hist_arr.shape[1]
    hist_has_nan = bool(np.isnan(X_hist_arr).any())
    future_has_nan = bool(np.isnan(X_future_arr).any())
    if hist_has_nan or future_has_nan:
        for j in range(n_exog):
            X_hist_arr[:, j] = _interp_nan_1d(X_hist_arr[:, j])
            # X_hist_arr[:, j] is now NaN-free; use its last value as
            # the all-NaN fallback for the matching X_future column.
            X_future_arr[:, j] = _interp_nan_1d(
                X_future_arr[:, j],
                all_nan_fallback=float(X_hist_arr[-1, j]),
            )

    return X_hist_arr, X_future_arr, n_exog


# ── y preprocessing ────────────────────────────────────────────────────


def _preprocess_y(y_arr: NDArray[np.floating]) -> tuple[NDArray[np.floating], float]:
    """Interpolate NaNs in ``y`` and apply a location shift to ensure positivity.

    Returns ``(y, y_shift)`` where ``y_shift`` will be subtracted from
    the final samples to undo the shift.  NaN handling delegates to the
    shared ``_interp_nan_1d`` helper so y and exog use the same policy.
    """
    y = _interp_nan_1d(y_arr)
    y_shift = max(1 - float(y.min()), 1.0)
    return y + y_shift, y_shift


def _degenerate_p1_fallback(
    y: NDArray[np.floating],
    horizon: int,
    n_samples: int,
    n: int,
    y_shift: float,
    rng: np.random.RandomState,
) -> NDArray[np.floating]:
    """Gaussian-noise fallback for series with fewer than ``_MIN_COMPLETE`` periods."""
    fc = np.full(horizon, y[-1] - y_shift)
    sigma = float(np.std(np.diff(y[-min(_PHASE_NOISE_K, n) :]))) if n > 1 else 1.0
    if sigma < _EPS_SHAPE:
        sigma = _EPS_SHAPE
    fc_mean = float(fc.mean())
    return np.clip(
        np.array([fc + rng.normal(0, sigma, horizon) for _ in range(n_samples)]),
        fc_mean - sigma * 10,
        fc_mean + sigma * 10,
    )


# ── Exogenous aggregation ──────────────────────────────────────────────


def _aggregate_exog_to_level(
    X_hist_arr: NDArray[np.floating],
    X_future_arr: NDArray[np.floating],
    n_complete: int,
    P: int,
    n_exog: int,
    m: int,
    horizon: int,
    usable: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Aggregate exog to the per-period (Level) timescale via period mean.

    Returns ``(X_L_raw, X_future_L_raw)`` of shapes ``(n_complete, n_exog)``
    and ``(m, n_exog)`` respectively.  Intra-period variation is dropped
    by design — exog lives on the same time grid as ``L``.
    """
    X_hist_trim = X_hist_arr[-usable:]
    X_L_raw = X_hist_trim.reshape(n_complete, P, n_exog).mean(axis=1)

    X_future_L_raw = np.zeros((m, n_exog))
    for j in range(m):
        s_idx = j * P
        e_idx = min((j + 1) * P, horizon)
        X_future_L_raw[j] = X_future_arr[s_idx:e_idx].mean(axis=0)

    return X_L_raw, X_future_L_raw


def _standardize_exog(
    X_L_raw: NDArray[np.floating],
    X_future_L_raw: NDArray[np.floating],
    start: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """z-score the exog columns using training statistics only."""
    mu_X = X_L_raw[start:].mean(axis=0)
    sd_X = X_L_raw[start:].std(axis=0)
    sd_X = np.where(sd_X < _EPS, 1.0, sd_X)
    return (X_L_raw - mu_X) / sd_X, (X_future_L_raw - mu_X) / sd_X


# ── Shape₂ deseasonalization ──────────────────────────────────────────


def _apply_shape2_deseason(
    L: NDArray[np.floating],
    cross_periods: list[int],
    n_complete: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None, int]:
    """Compute Shape₂ for the secondary period and return the deseasonalized Level.

    Returns ``(L_work, S2, cp_main)`` where ``L_work = L / S2[pos]``
    when Shape₂ is applied (and ``L_work = L`` otherwise), ``S2`` is
    the per-secondary-phase factor (or ``None``), and ``cp_main`` is
    the secondary period (or ``0`` if no Shape₂ is applied).
    """
    cp_main = cross_periods[0] if cross_periods else 0
    if cp_main < 2:
        return L, None, 0
    S2 = _compute_shape2(L, cp_main, n_complete)
    if S2 is None:
        return L, None, 0
    pos = np.arange(n_complete) % cp_main
    L_work = L / np.maximum(S2[pos], _EPS)
    return L_work, S2, cp_main


# ── Level Ridge design matrix and fit ─────────────────────────────────


def _build_level_design(
    L_innov: NDArray[np.floating],
    X_L_std: NDArray[np.floating] | None,
    n_complete: int,
    max_cp: int,
    n_train: int,
    nb: int,
    base: NDArray[np.floating],
    n_exog: int,
    start: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], bool, int, int]:
    """Build the Ridge design matrix and the regression target.

    Returns ``(X_full, y_target, is_diff, nf, nf_total)``.

    Under the LSR1 reparameterization (the default), the target is
    ``ΔL_innov`` and the lag-1 column carries a sign flip so that the
    Ridge shrinks ``δ₂ = 1 − β₂ → 0`` (random-walk prior).  In the
    short-series fallback the original ``L_innov`` target and a
    standard lag-1 column are used.

    Exog columns enter linearly on both sides of the LSR1 rewrite and
    pass through unchanged — no extra correction is needed when
    recovering the original parameterization.
    """
    n_lag = 1 + (1 if max_cp >= 2 else 0)
    nf = nb + n_lag
    nf_total = nf + n_exog

    X_full = np.zeros((n_train, nf_total))
    X_full[:, :nb] = base[start:]
    if X_L_std is not None:
        X_full[:, nf:] = X_L_std[start:]

    if _DIFF_TARGET and n_train >= 3:
        y_target = np.diff(L_innov[start - 1 :])  # ΔL_innov
        X_full[:, nb] = -L_innov[start - 1 : -1]  # negated lag-1 → δ₂
        if max_cp >= 2:
            X_full[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]
        is_diff = True
    else:
        X_full[:, nb] = L_innov[start - 1 : -1]
        if max_cp >= 2:
            X_full[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]
        y_target = L_innov[start:]
        is_diff = False

    return X_full, y_target, is_diff, nf, nf_total


def _recover_beta(
    theta: NDArray[np.floating],
    is_diff: bool,
    nb: int,
) -> NDArray[np.floating]:
    """Undo the LSR1 reparameterization on the Ridge coefficient vector.

    In diff mode, the Ridge fit produces ``δ₂``; the recovered Level
    coefficient is ``β₂ = 1 − δ₂``.  Exogenous coefficients pass
    through unchanged.
    """
    beta = theta.copy()
    if is_diff:
        beta[nb] = 1.0 - theta[nb]
    return beta


# ── Damped trend ───────────────────────────────────────────────────────


def _compute_damped_trend(
    L_bc: NDArray[np.floating],
    m: int,
    n_complete: int,
) -> NDArray[np.floating]:
    """Compute the damped-trend coefficients for ``m`` forecast steps."""
    phi = _estimate_phi(L_bc)
    if phi <= _EPS:
        return np.full(m, (n_complete - 1.0) / n_complete)
    out = np.empty(m)
    for j in range(m):
        out[j] = ((n_complete - 1) + phi * (1.0 - phi ** (j + 1)) / (1.0 - phi)) / n_complete
    return out


# ── LWCP test-point leverages ──────────────────────────────────────────


def _compute_lwcp_leverages(
    beta: NDArray[np.floating],
    L_innov: NDArray[np.floating],
    damped_trend: NDArray[np.floating],
    X_future_L_std: NDArray[np.floating] | None,
    Vt_r: NDArray[np.floating],
    s_r: NDArray[np.floating],
    d_avg_r: NDArray[np.floating],
    n_complete: int,
    m: int,
    nb: int,
    nf: int,
    nf_total: int,
    max_cp: int,
    n_train: int,
) -> NDArray[np.floating]:
    """Per-horizon LWCP leverages projected through the Ridge SVD.

    ``h_test[j]`` grows with horizon because the trend extrapolates and
    the lag features become themselves predicted values.  When exog is
    provided, the standardized future exog row enters the leverage
    feature vector as well.
    """
    h_test = np.empty(m)
    L_point = np.concatenate([L_innov, np.zeros(m)])
    for j in range(m):
        ti = n_complete + j
        pred_pt = beta[0] + beta[1] * damped_trend[j] + beta[nb] * L_point[ti - 1]
        if max_cp >= 2:
            pred_pt += beta[nb + 1] * L_point[ti - max_cp]
        if X_future_L_std is not None:
            pred_pt += float(X_future_L_std[j] @ beta[nf:])
        L_point[ti] = pred_pt

        x_j = np.zeros(nf_total)
        x_j[0] = 1.0
        x_j[1] = damped_trend[j]
        if _DIFF_TARGET and n_train >= 3:
            x_j[nb] = -L_point[ti - 1]
        else:
            x_j[nb] = L_point[ti - 1]
        if max_cp >= 2:
            x_j[nb + 1] = L_point[ti - max_cp]
        if X_future_L_std is not None:
            x_j[nf:] = X_future_L_std[j]

        v = Vt_r @ x_j
        u_test = v / np.maximum(s_r, _EPS)
        h_test[j] = float(np.sum(u_test**2 * d_avg_r))

    return np.clip(h_test, 0.0, 10.0)


# ── Stochastic Level paths ─────────────────────────────────────────────


def _sample_level_paths(
    beta: NDArray[np.floating],
    loo_resid: NDArray[np.floating],
    h_test: NDArray[np.floating],
    L_innov: NDArray[np.floating],
    damped_trend: NDArray[np.floating],
    X_future_L_std: NDArray[np.floating] | None,
    n_complete: int,
    m: int,
    nb: int,
    nf: int,
    max_cp: int,
    n_samples: int,
    n_train: int,
    rng: np.random.RandomState,
) -> tuple[NDArray[np.floating], int]:
    """Recursive Level path sampling under a Student-t innovation model.

    When ``σ²`` is unknown and estimated from the LWCP-normalized LOO
    residuals, the predictive distribution in Box-Cox space is
    Student-t with ``ν = n_train − p`` degrees of freedom.  As
    ``n → ∞``, ``t_ν → N(0, 1)``; for short series, the heavier tails
    reflect higher uncertainty automatically — zero new hyperparameters.

    Returns ``(L_paths, nu)``: the ``(n_samples, n_complete + m)``
    augmented Level path array and the Student-t degrees of freedom
    used (so the caller can reuse it for post-hoc shrinkage).
    """
    sigma2_loo = float(np.mean(loo_resid**2))
    nu = max(n_train - nf, 3)  # floor at 3 for stability

    noise_pool = (
        rng.standard_t(df=nu, size=(n_samples, m))
        * np.sqrt(sigma2_loo * (1.0 + h_test))[np.newaxis, :]
    )
    L_paths = np.column_stack([np.tile(L_innov, (n_samples, 1)), np.zeros((n_samples, m))])

    for j in range(m):
        ti = n_complete + j
        pred = beta[0] + beta[1] * damped_trend[j] + beta[nb] * L_paths[:, ti - 1]
        if max_cp >= 2:
            pred += beta[nb + 1] * L_paths[:, ti - max_cp]
        if X_future_L_std is not None:
            pred += float(X_future_L_std[j] @ beta[nf:])
        L_paths[:, ti] = pred + noise_pool[:, j]

    return L_paths, nu


# ── Phase noise ────────────────────────────────────────────────────────


def _phase_noise_sample(
    mat: NDArray[np.floating],
    S_hist: NDArray[np.floating],
    L: NDArray[np.floating],
    n_complete: int,
    P: int,
    horizon: int,
    m: int,
    n_samples: int,
    rng: np.random.RandomState,
) -> tuple[NDArray[np.floating], NDArray[np.intp], NDArray[np.intp]]:
    """Scenario-coherent phase-noise sampling from SVD residual columns.

    ``R[p, k] = (observed − fitted) / |fitted|`` captures phase-specific
    multiplicative noise.  Each *column* of ``R`` is one historical
    period's residual pattern across all P phases.  Sampling whole
    columns preserves cross-phase correlation: every phase within a
    forecast block shares one historical period's deviation pattern.
    This is the empirical distribution of the rank-1 residual, which
    is the natural uncertainty source for a factored model.

    Returns ``(phase_noise, step_idx, phase_idx)``.
    """
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(_PHASE_NOISE_K, n_complete)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), _EPS_BOXCOX)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    col_idx = rng.randint(0, K_r, size=(n_samples, m))
    phase_noise = R[phase_idx[np.newaxis, :], col_idx[:, step_idx]]
    return phase_noise, step_idx, phase_idx


# ── Sample assembly + post-hoc calibration ────────────────────────────


def _assemble_and_calibrate(
    L_hat_all: NDArray[np.floating],
    S_forecast: NDArray[np.floating],
    phase_noise: NDArray[np.floating],
    step_idx: NDArray[np.intp],
    phase_idx: NDArray[np.intp],
    y_arr: NDArray[np.floating],
    y_shift: float,
    P: int,
    horizon: int,
    nu: int,
) -> NDArray[np.floating]:
    """Combine Level × Shape × (1 + phase_noise), clip, and shrink intervals.

    Three post-processing steps:

    1. Multiplicative assembly with the per-block Shape vector and the
       sampled phase noise; subtract the location shift.
    2. Clip to a recent-history window so that inverse Box-Cox blow-ups
       don't propagate into the final samples.
    3. Post-hoc Student-t shrinkage toward the median (only when
       ``ν < 50``) to remove the heavy-tail variance inflation while
       preserving the median exactly — a monotone, centered transform.
    """
    S_h = S_forecast[step_idx, phase_idx]
    samples = L_hat_all[:, step_idx] * S_h[np.newaxis, :] * (1 + phase_noise) - y_shift

    # Clip to recent-history range.  Inverse Box-Cox combined with
    # recursive Student-t simulation can produce extreme values; the
    # window matches the P=1 fallback's ±10σ idea.
    tail = y_arr[-min(horizon * 2, max(50, P * 3)) :]
    y_lo, y_hi = float(np.nanmin(tail)), float(np.nanmax(tail))
    y_range = max(y_hi - y_lo, max(abs(y_hi), abs(y_lo), 1.0))
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)

    # Post-hoc Student-t shrinkage to undo heavy-tail interval inflation
    # (`sqrt(ν / (ν−2))` ≈ 1.73× at ν = 3).  The transform is monotone
    # and centered on the median, so the point forecast is preserved.
    if nu < 50:
        shrink = np.sqrt(max(nu - 2.0, 0.5) / nu)
        med = np.median(samples, axis=0, keepdims=True)
        samples = med + shrink * (samples - med)

    return np.asarray(np.nan_to_num(samples, posinf=0.0, neginf=0.0), dtype=np.float64)


# ── Public orchestrator ────────────────────────────────────────────────


def forecast(
    y_raw: ArrayLike,
    horizon: int,
    freq: str,
    n_samples: int = 200,
    seed: int | None = None,
    X_hist: ArrayLike | None = None,
    X_future: ArrayLike | None = None,
) -> NDArray[np.floating]:
    """Generate probabilistic forecasts for a univariate time series.

    Parameters
    ----------
    y_raw : array-like, shape (n,)
        Historical observations.
    horizon : int
        Number of steps to forecast.
    freq : str
        Frequency string ('H', 'D', 'W', 'M', '5T', etc.).
    n_samples : int
        Number of sample paths for probabilistic forecast.
    seed : int or None
        Random seed for reproducibility. If None, uses OS entropy.
    X_hist : array-like, shape (n, k) or (n,), optional
        Historical exogenous variables aligned with y_raw.  Aggregated to
        the Level (per-period) timescale via period mean and z-scored
        using training statistics before joining the Level Ridge.  Must
        be provided together with X_future.
    X_future : array-like, shape (horizon, k) or (horizon,), optional
        Future exogenous variables for the forecast horizon.  Same
        treatment as X_hist.  Must be provided together with X_hist.

    Returns
    -------
    samples : ndarray, shape (n_samples, horizon)
        Probabilistic forecast sample paths.

    Raises
    ------
    TypeError
        If freq is not a string, or horizon/n_samples are not integers.
    ValueError
        If horizon < 1, n_samples < 1, y is empty, or y is not 1-dimensional.
        If only one of X_hist/X_future is provided, or shapes are inconsistent.

    Notes
    -----
    Standardized exog columns are appended directly to the Level Ridge
    feature matrix.  No separate gating step is used: FLAIR's existing
    LOOCV soft-averaged Ridge already shrinks columns whose contribution
    to LOO error is negligible, so noise exog is naturally damped toward
    zero by the regularization that already runs in `_ridge_sa`.  This
    keeps the "One SVD, One Ridge" property — exog support adds zero
    extra Ridge fits, no model selection, no auxiliary state.

    The exog signal is aggregated to the per-period (Level) timescale by
    taking the mean of each P-block.  Intra-period variation in X is not
    captured by this design.
    """
    # 1. Validate basic inputs and exog shapes
    y_arr = _validate_inputs(y_raw, horizon, freq, n_samples)
    X_hist_arr, X_future_arr, n_exog = _validate_and_clean_exog(
        X_hist, X_future, len(y_arr), horizon
    )

    rng = np.random.RandomState(seed)

    # 2. Preprocess y: NaN interp + positivity shift
    y, y_shift = _preprocess_y(y_arr)
    n = len(y)

    # 3. MDL period selection (with fallback to P=1 / Gaussian for very short series)
    P, secondary, period, _cal = _select_period(y, n, freq)
    n_complete = n // P
    if n_complete < _MIN_COMPLETE:
        if P > 1:
            P = 1
            secondary = []
            n_complete = n
        if n_complete < _MIN_COMPLETE:
            return _degenerate_p1_fallback(y, horizon, n_samples, n, y_shift, rng)

    # 4. Cap to MAX_COMPLETE periods (memory / runtime guard)
    if n_complete > _MAX_COMPLETE:
        y = y[-(_MAX_COMPLETE * P) :]
        if X_hist_arr is not None:
            X_hist_arr = X_hist_arr[-(_MAX_COMPLETE * P) :]
        n = len(y)
        n_complete = n // P

    # 5. Reshape into the (P, n_complete) period-folded matrix
    usable = n_complete * P
    y_trim = y[-usable:]
    mat = y_trim.reshape(n_complete, P).T
    L = mat.sum(axis=0)

    # 6. Estimate Shape (Dirichlet-Multinomial EB) — also returns m = ⌈h/P⌉
    S_forecast, S_hist, m = _estimate_shape(mat, n_complete, P, secondary, L, horizon)

    # 7. Aggregate exog to the Level timescale (period mean) — uses m from above
    X_L_raw: NDArray[np.floating] | None = None
    X_future_L_raw: NDArray[np.floating] | None = None
    if X_hist_arr is not None and X_future_arr is not None:
        X_L_raw, X_future_L_raw = _aggregate_exog_to_level(
            X_hist_arr, X_future_arr, n_complete, P, n_exog, m, horizon, usable
        )

    # 8. Cross-period lags + Shape₂ deseasonalization of L
    cross_periods, max_cp = _compute_cross_periods(secondary, P, period, n_complete)
    L_work, S2, cp_main = _apply_shape2_deseason(L, cross_periods, n_complete)

    # 9. Box-Cox transform + innovation centering on the deseasonalized Level
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # 10. Ridge feature setup: intercept, trend, lag-1, optional cross-period lag
    start = max(1, max_cp) if max_cp >= 2 else 1
    if n_complete - start < _MIN_COMPLETE and max_cp >= 2:
        max_cp = 0
        start = 1
    n_train = n_complete - start
    t = np.arange(n_complete, dtype=float)
    base = np.column_stack([np.ones(n_complete), t / n_complete])
    nb = base.shape[1]

    # 11. Standardize exog (training-window stats) and build the design matrix
    X_L_std: NDArray[np.floating] | None = None
    X_future_L_std: NDArray[np.floating] | None = None
    if X_L_raw is not None and X_future_L_raw is not None:
        X_L_std, X_future_L_std = _standardize_exog(X_L_raw, X_future_L_raw, start)
    X_full, y_target, is_diff, nf, nf_total = _build_level_design(
        L_innov, X_L_std, n_complete, max_cp, n_train, nb, base, n_exog, start
    )

    # 12. Single Ridge fit (LOOCV soft-average + LWCP normalization)
    theta, loo_resid, _, Vt_r, s_r, d_avg_r = _ridge_sa(X_full, y_target)
    beta = _recover_beta(theta, is_diff, nb)

    # 13. Damped-trend extrapolation coefficients
    damped_trend = _compute_damped_trend(L_bc, m, n_complete)

    # 14. Per-horizon LWCP leverages
    h_test = _compute_lwcp_leverages(
        beta,
        L_innov,
        damped_trend,
        X_future_L_std,
        Vt_r,
        s_r,
        d_avg_r,
        n_complete,
        m,
        nb,
        nf,
        nf_total,
        max_cp,
        n_train,
    )

    # 15. Stochastic Level paths (recursive Student-t innovations)
    L_paths, nu = _sample_level_paths(
        beta,
        loo_resid,
        h_test,
        L_innov,
        damped_trend,
        X_future_L_std,
        n_complete,
        m,
        nb,
        nf,
        max_cp,
        n_samples,
        n_train,
        rng,
    )

    # 16. Inverse Box-Cox + re-seasonalize via Shape₂
    L_hat_all = _bc_inv(L_paths[:, n_complete : n_complete + m] + last_L, lam)
    if S2 is not None:
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # 17. Phase-noise sampling (scenario-coherent column sampling)
    phase_noise, step_idx, phase_idx = _phase_noise_sample(
        mat, S_hist, L, n_complete, P, horizon, m, n_samples, rng
    )

    # 18. Multiplicative assembly + clipping + post-hoc shrinkage
    return _assemble_and_calibrate(
        L_hat_all,
        S_forecast,
        phase_noise,
        step_idx,
        phase_idx,
        y_arr,
        y_shift,
        P,
        horizon,
        nu,
    )


# ── Class API ───────────────────────────────────────────────────────────


class FLAIR:
    """Factored Level And Interleaved Ridge forecaster.

    Parameters
    ----------
    freq : str
        Frequency string ('H', 'D', 'W', 'M', '5T', etc.).
    n_samples : int
        Default number of sample paths for probabilistic forecast.

    Example
    -------
    >>> model = FLAIR(freq='H')
    >>> samples = model.predict(y, horizon=24)
    >>> point = samples.mean(axis=0)
    """

    def __init__(
        self,
        freq: str,
        n_samples: int = 200,
        seed: int | None = None,
    ) -> None:
        if not isinstance(freq, str):
            raise TypeError(f"freq must be a string, got {type(freq).__name__}")
        if not isinstance(n_samples, (int, np.integer)):
            raise TypeError(f"n_samples must be an integer, got {type(n_samples).__name__}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.freq = freq
        self.n_samples = n_samples
        self.seed = seed

    def predict(
        self,
        y: ArrayLike,
        horizon: int,
        n_samples: int | None = None,
        seed: int | None = None,
        X_hist: ArrayLike | None = None,
        X_future: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        """Generate probabilistic forecasts.

        Parameters
        ----------
        y : array-like, shape (n,)
            Historical observations.
        horizon : int
            Number of steps to forecast.
        n_samples : int, optional
            Override the default number of sample paths.
        seed : int or None, optional
            Override the default random seed.
        X_hist : array-like, shape (n, k) or (n,), optional
            Historical exogenous variables aligned with y.  See
            `forecast()` for details.
        X_future : array-like, shape (horizon, k) or (horizon,), optional
            Future exogenous variables for the forecast horizon.

        Returns
        -------
        samples : ndarray, shape (n_samples, horizon)
            Probabilistic forecast sample paths.
        """
        return forecast(
            y,
            horizon,
            self.freq,
            n_samples if n_samples is not None else self.n_samples,
            seed if seed is not None else self.seed,
            X_hist=X_hist,
            X_future=X_future,
        )
