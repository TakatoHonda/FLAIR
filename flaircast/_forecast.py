"""Main forecast pipeline.

Hosts the public `forecast()` function and the `FLAIR` class wrapper.
The function orchestrates the operators defined in `_level`, `_shape`,
and `_period` to produce probabilistic sample paths.

In a future commit `forecast()` will be decomposed into named helpers
that mirror the algorithm sections of the paper; for now it remains a
single procedure to keep the file-split commit a pure relocation.
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

    # ── Exogenous variable validation ────────────────────────────────
    has_exog = X_hist is not None or X_future is not None
    if has_exog:
        if X_hist is None or X_future is None:
            raise ValueError("X_hist and X_future must be provided together")
        X_hist_arr = np.asarray(X_hist, dtype=float)
        X_future_arr = np.asarray(X_future, dtype=float)
        if X_hist_arr.ndim == 1:
            X_hist_arr = X_hist_arr[:, np.newaxis]
        if X_future_arr.ndim == 1:
            X_future_arr = X_future_arr[:, np.newaxis]
        if X_hist_arr.ndim != 2:
            raise ValueError(f"X_hist must be 1D or 2D, got shape {X_hist_arr.shape}")
        if X_future_arr.ndim != 2:
            raise ValueError(f"X_future must be 1D or 2D, got shape {X_future_arr.shape}")
        if X_hist_arr.shape[0] != len(y_arr):
            raise ValueError(
                f"X_hist length ({X_hist_arr.shape[0]}) must match y length ({len(y_arr)})"
            )
        if X_future_arr.shape[0] != horizon:
            raise ValueError(
                f"X_future length ({X_future_arr.shape[0]}) must match horizon ({horizon})"
            )
        if X_hist_arr.shape[1] != X_future_arr.shape[1]:
            raise ValueError(
                f"X_hist columns ({X_hist_arr.shape[1]}) must match "
                f"X_future columns ({X_future_arr.shape[1]})"
            )
        # NaN handling: column-mean imputation using training statistics
        # only.  Consistent with the y-side `np.interp` policy in spirit
        # (no silent 0-fill), and works even when X_future is fully NaN.
        if np.isnan(X_hist_arr).any() or np.isnan(X_future_arr).any():
            col_means = np.nanmean(X_hist_arr, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            X_hist_arr = np.where(np.isnan(X_hist_arr), col_means, X_hist_arr)
            X_future_arr = np.where(np.isnan(X_future_arr), col_means, X_future_arr)
        n_exog = X_hist_arr.shape[1]
    else:
        X_hist_arr = None
        X_future_arr = None
        n_exog = 0

    rng = np.random.RandomState(seed)
    # Interpolate NaN instead of filling with 0 (which biases Shape/Level)
    y = np.asarray(y_arr, dtype=float).copy()
    nan_mask = np.isnan(y)
    if nan_mask.any():
        valid = ~nan_mask
        if valid.sum() >= 2:
            idx = np.arange(len(y))
            y[nan_mask] = np.interp(idx[nan_mask], idx[valid], y[valid])
        elif valid.sum() == 1:
            y[nan_mask] = y[valid][0]
        else:
            y[:] = 0.0
    # Location shift: make all values positive for multiplicative decomposition
    y_floor = y.min()
    y_shift = max(1 - y_floor, 1.0)  # shift so min(y) >= 1
    y = y + y_shift
    n = len(y)

    P, secondary, period, _cal = _select_period(y, n, freq)
    n_complete = n // P
    if n_complete < _MIN_COMPLETE:
        if P > 1:
            P = 1
            secondary = []
            n_complete = n
        if n_complete < _MIN_COMPLETE:
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

    if n_complete > _MAX_COMPLETE:
        y = y[-(_MAX_COMPLETE * P) :]
        if has_exog:
            assert X_hist_arr is not None
            X_hist_arr = X_hist_arr[-(_MAX_COMPLETE * P) :]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]

    # ── Reshape ──────────────────────────────────────────────────────
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)
    L = mat.sum(axis=0)

    # ── Aggregate exogenous to Level (per-period) timescale ─────────
    # Period mean over each P-block, mirroring the (n_complete, P) row
    # layout used to build `L`.  Intra-period variation in X is dropped
    # by design — exog lives in the same time grid as Level.
    X_L_raw: NDArray[np.floating] | None = None
    if has_exog:
        assert X_hist_arr is not None
        X_hist_trim = X_hist_arr[-usable:]
        X_L_raw = X_hist_trim.reshape(n_complete, P, n_exog).mean(axis=1)

    S_forecast, S_hist, m = _estimate_shape(
        mat,
        n_complete,
        P,
        secondary,
        L,
        horizon,
    )

    # ── Aggregate future exogenous block-by-block (uses m from above) ─
    X_future_L_raw: NDArray[np.floating] | None = None
    if has_exog:
        assert X_future_arr is not None
        X_future_L_raw = np.zeros((m, n_exog))
        for j in range(m):
            s_idx = j * P
            e_idx = min((j + 1) * P, horizon)
            X_future_L_raw[j] = X_future_arr[s_idx:e_idx].mean(axis=0)

    cross_periods, max_cp = _compute_cross_periods(
        secondary,
        P,
        period,
        n_complete,
    )

    # ── Shape₂: deseasonalize Level at secondary period ───────────────
    cp_main = cross_periods[0] if cross_periods else 0
    S2 = None
    use_deseason = False
    if cp_main >= 2:
        S2 = _compute_shape2(L, cp_main, n_complete)
        if S2 is not None:
            use_deseason = True

    if use_deseason:
        assert S2 is not None
        pos = np.arange(n_complete) % cp_main
        L_work = L / np.maximum(S2[pos], _EPS)
    else:
        L_work = L

    # ── Box-Cox → NLinear on (deseasonalized) Level ───────────────────
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # ── Ridge features: intercept + trend + lags (no Fourier) ─────────
    start = max(1, max_cp) if max_cp >= 2 else 1
    if n_complete - start < _MIN_COMPLETE and max_cp >= 2:
        max_cp = 0
        start = 1

    n_train = n_complete - start
    t = np.arange(n_complete, dtype=float)
    trend = t / n_complete
    cols = [np.ones(n_complete), trend]
    nb = len(cols)
    base = np.column_stack(cols)

    n_lag = 1 + (1 if max_cp >= 2 else 0)
    nf = nb + n_lag
    nf_total = nf + n_exog  # nf when has_exog is False (n_exog == 0)

    # ── Build Ridge feature matrix (intercept, trend, lags, [exog]) ──
    # Exog columns, when provided, are appended directly to the design
    # matrix and standardized using training statistics.  No separate
    # gating step: the LOOCV soft-average inside `_ridge_sa` already
    # damps columns whose contribution to LOO error is negligible, so
    # noise exog converges to ≈0 coefficients without any model
    # selection machinery on top.  This is the "One Ridge" extension
    # of FLAIR — exog adds columns, never adds Ridge calls.
    X_full = np.zeros((n_train, nf_total))
    X_full[:, :nb] = base[start:]

    X_future_L_std: NDArray[np.floating] | None = None
    if has_exog:
        assert X_L_raw is not None and X_future_L_raw is not None
        # z-score using training rows only ([start:]); constant cols → 1
        mu_X = X_L_raw[start:].mean(axis=0)
        sd_X = X_L_raw[start:].std(axis=0)
        sd_X = np.where(sd_X < _EPS, 1.0, sd_X)
        X_L_std = (X_L_raw - mu_X) / sd_X
        X_future_L_std = (X_future_L_raw - mu_X) / sd_X
        X_full[:, nf:] = X_L_std[start:]

    if _DIFF_TARGET and n_train >= 3:
        # ── LSR1 reparameterization ─────────────────────────────────
        # Under the LSR1 model, L ∈ Hölder(2) implies the Level is
        # smooth, so consecutive values satisfy L(i) ≈ L(i-1), i.e.,
        # the "natural" AR coefficient β₂ ≈ 1.  Standard Ridge shrinks
        # β₂ → 0 (stationarity prior), which is inconsistent.
        #
        # Reparameterize: let δ₂ = 1 - β₂.  Rewrite the model as
        #   ΔL_innov[i] = β₀ + β₁(i/n) − δ₂ L_innov[i-1] + β₃ lag_cp
        #                  + X_exog[i] · β_exog
        # Now Ridge shrinks δ₂ → 0, i.e., β₂ → 1 (random walk prior).
        # The exog coefficients enter linearly on both sides of the
        # rewrite and pass through unchanged — no extra correction.
        y_target = np.diff(L_innov[start - 1 :])  # ΔL_innov
        X_full[:, nb] = -L_innov[start - 1 : -1]  # negated lag1 → δ₂
        if max_cp >= 2:
            X_full[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]
        is_diff = True
    else:
        # ── Standard Ridge (fallback for very short series) ──────────
        X_full[:, nb] = L_innov[start - 1 : -1]
        if max_cp >= 2:
            X_full[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]
        y_target = L_innov[start:]
        is_diff = False

    theta, loo_resid, _, Vt_r, s_r, d_avg_r = _ridge_sa(X_full, y_target)

    # Recover original parameterization (only β₂ needs the rewrite)
    beta = theta.copy()
    if is_diff:
        beta[nb] = 1.0 - theta[nb]  # β₂ = 1 − δ₂

    # ── Damped trend (LSR1 boundary extrapolation) ───────────────────
    phi = _estimate_phi(L_bc)
    if phi > _EPS:
        _damped_trend = np.empty(m)
        for j in range(m):
            _damped_trend[j] = (
                (n_complete - 1) + phi * (1.0 - phi ** (j + 1)) / (1.0 - phi)
            ) / n_complete
    else:
        _damped_trend = np.full(m, (n_complete - 1.0) / n_complete)

    # ── Test-point leverages for LWCP ──────────────────────────────────
    # Compute h_test[j] for each forecast step using the point-forecast
    # feature vector projected through the Ridge SVD.  h_test grows with
    # horizon because trend extrapolates and lags become predicted values.
    # When exog is provided, the feature vector also carries the
    # standardized future exog row.
    h_test = np.empty(m)
    L_point = np.concatenate([L_innov, np.zeros(m)])
    for j in range(m):
        ti = n_complete + j
        pred_pt = beta[0] + beta[1] * _damped_trend[j] + beta[nb] * L_point[ti - 1]
        if max_cp >= 2:
            pred_pt += beta[nb + 1] * L_point[ti - max_cp]
        if has_exog:
            assert X_future_L_std is not None
            pred_pt += float(X_future_L_std[j] @ beta[nf:])
        L_point[ti] = pred_pt

        x_j = np.zeros(nf_total)
        x_j[0] = 1.0
        x_j[1] = _damped_trend[j]
        if _DIFF_TARGET and n_train >= 3:
            x_j[nb] = -L_point[ti - 1]
        else:
            x_j[nb] = L_point[ti - 1]
        if max_cp >= 2:
            x_j[nb + 1] = L_point[ti - max_cp]
        if has_exog:
            assert X_future_L_std is not None
            x_j[nf:] = X_future_L_std[j]

        v = Vt_r @ x_j
        u_test = v / np.maximum(s_r, _EPS)
        h_test[j] = float(np.sum(u_test**2 * d_avg_r))

    h_test = np.clip(h_test, 0.0, 10.0)

    # ── Stochastic Level paths (Student-t in Box-Cox space) ────────────
    # When σ² is unknown and estimated from LOO residuals, the predictive
    # distribution in BC space is Student-t with ν = n_train − p degrees
    # of freedom, not Gaussian.  This is a standard result from Bayesian
    # linear regression (marginalizing over σ²).
    #
    # The Student-t naturally produces heavier tails for short series
    # (small ν) where uncertainty is highest.  As n → ∞, t_ν → N(0,1).
    # Combined with the inverse Box-Cox, this gives an asymmetric,
    # heavy-tailed predictive distribution with zero additional parameters.
    sigma2_loo = float(np.mean(loo_resid**2))
    nu = max(n_train - nf, 3)  # degrees of freedom, floor at 3 for stability
    noise_pool = (
        rng.standard_t(df=nu, size=(n_samples, m))
        * np.sqrt(sigma2_loo * (1.0 + h_test))[np.newaxis, :]
    )
    L_paths = np.column_stack(
        [np.tile(L_innov, (n_samples, 1)), np.zeros((n_samples, m))]
    )  # (n_samples, n_complete + m)

    for j in range(m):
        ti = n_complete + j
        pred = beta[0] + beta[1] * _damped_trend[j] + beta[nb] * L_paths[:, ti - 1]
        if max_cp >= 2:
            pred += beta[nb + 1] * L_paths[:, ti - max_cp]
        if has_exog:
            assert X_future_L_std is not None
            pred += float(X_future_L_std[j] @ beta[nf:])
        L_paths[:, ti] = pred + noise_pool[:, j]

    # Point forecast = median path (noise_pool row 0 is arbitrary, use mean)
    L_hat_all = _bc_inv(L_paths[:, n_complete : n_complete + m] + last_L, lam)

    if use_deseason:
        assert S2 is not None
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # ── Phase noise (scenario-coherent column sampling) ──────────────
    # R[p,k] = (observed - fitted)/|fitted| captures phase-specific noise.
    # Each column of R is one historical period's residual pattern across
    # all P phases.  Sampling entire columns preserves cross-phase
    # correlation: all phases within one forecast block share the same
    # historical period's deviation pattern.  This is not ad-hoc — it is
    # the empirical distribution of the rank-1 residual, the natural
    # uncertainty source for a factored model.
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(_PHASE_NOISE_K, n_complete)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), _EPS_BOXCOX)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    col_idx = rng.randint(0, K_r, size=(n_samples, m))
    phase_noise = R[phase_idx[np.newaxis, :], col_idx[:, step_idx]]

    # ── Assemble: Level_path × Shape × (1 + phase_noise) ───────────
    S_h = S_forecast[step_idx, phase_idx]

    samples = L_hat_all[:, step_idx] * S_h[np.newaxis, :] * (1 + phase_noise) - y_shift

    # ── Clip to historical range ──────────────────────────────────────
    # Inverse Box-Cox with recursive simulation can produce extreme values.
    # Clip to [y_lo - range, y_hi + range] based on the recent history,
    # consistent with the P=1 fallback path (±10σ clipping).
    tail = y_arr[-min(horizon * 2, max(50, P * 3)) :]
    y_lo, y_hi = float(np.nanmin(tail)), float(np.nanmax(tail))
    y_range = max(y_hi - y_lo, max(abs(y_hi), abs(y_lo), 1.0))
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)

    # ── Post-hoc interval calibration ─────────────────────────────────
    # Student-t noise during recursion creates path diversity that helps
    # the median (point forecast) via Box-Cox asymmetry.  But the heavy
    # tails inflate interval width by sqrt(nu/(nu-2)), which is 1.73x
    # for nu=3.  Shrinking samples toward the median removes this excess
    # while preserving the median exactly (monotone, centered transform).
    if nu < 50:
        shrink = np.sqrt(max(nu - 2.0, 0.5) / nu)
        med = np.median(samples, axis=0, keepdims=True)
        samples = med + shrink * (samples - med)

    return np.asarray(np.nan_to_num(samples, posinf=0.0, neginf=0.0), dtype=np.float64)


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
