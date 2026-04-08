"""FLAIR: Factored Level And Interleaved Ridge.

A single-equation time series forecasting method that separates
WHAT (level) from HOW (shape) via period-aligned matrix reshaping.

    y(phase, period) = Level(period) × Shape(phase)

Level is forecast by Ridge regression with soft-average GCV.
Shape is estimated via Dirichlet-Multinomial empirical Bayes, with
context derived from the secondary period structure (e.g., day-of-week
for hourly data). When no secondary period exists, degenerates to
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

Example:
    >>> from flaircast import forecast
    >>> samples = forecast(y, horizon=24, freq='H')
    >>> point = samples.mean(axis=0)

    >>> from flaircast import FLAIR
    >>> model = FLAIR(freq='H')
    >>> samples = model.predict(y, horizon=24)
"""

from __future__ import annotations

__version__ = "0.4.1"
__all__ = ["FLAIR", "FREQ_TO_PERIOD", "FREQ_TO_PERIODS", "forecast"]

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import boxcox as scipy_boxcox

# ── Numerical constants ─────────────────────────────────────────────────

_EPS = 1e-10  # General-purpose division guard
_EPS_BOXCOX = 1e-8  # Floor for Box-Cox input (y must be positive)
_EPS_LOG = 1e-30  # Floor inside log() to avoid -inf in BIC
_EPS_WEIGHT = 1e-15  # Threshold for skipping negligible softmax weights
_EPS_SHAPE = 1e-6  # Floor for Shape proportions
_BC_EXP_CLIP = 30  # Clip range for exp() in Box-Cox inverse (lam=0)
_MIN_POSITIVE_FOR_BC = 10  # Minimum positive values for Box-Cox lambda estimation
_MIN_COMPLETE = 3  # Minimum complete periods for non-degenerate mode
_MAX_COMPLETE = 500  # Cap on complete periods (memory/speed guard)
_DIFF_TARGET = True  # LSR1 reparameterization: target ΔL_innov, shrink δ₂=1-β₂ → 0
_SHAPE_K = 2  # Number of recent periods for Shape estimation (insensitive; see paper)
_PHASE_NOISE_K = 50  # Number of recent periods for phase noise
_N_ALPHAS = 25  # Number of log-spaced GCV alphas
_ALPHA_LOG_MIN = -4  # log10 of minimum Ridge alpha
_ALPHA_LOG_MAX = 4  # log10 of maximum Ridge alpha
_DIAG: dict = {}  # Diagnostic output (populated when non-empty)

# ── Calendar tables ──────────────────────────────────────────────────────

FREQ_TO_PERIOD = {
    "S": 60,
    "T": 60,
    "5T": 12,
    "10T": 6,
    "15T": 4,
    "10S": 6,
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "Q": 4,
    "A": 1,
    "Y": 1,
}

FREQ_TO_PERIODS = {
    "10S": [6, 360],
    "S": [60],
    "5T": [12, 288],
    "10T": [6, 144],
    "15T": [4, 96],
    "H": [24, 168],
    "D": [7, 365],
    "W": [52],
    "M": [12],
    "Q": [4],
    "A": [],
    "Y": [],
}


def _resolve_freq(freq: str) -> str:
    f = freq.upper().replace("MIN", "T")
    # Strip pandas offset anchors: W-SUN → W, Q-DEC → Q, A-DEC → A, etc.
    for base in ("W", "Q", "A", "Y"):
        if f.startswith(base + "-"):
            return base
    return f


def _get_period(freq: str) -> int:
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIOD:
        return FREQ_TO_PERIOD[f]
    for k in sorted(FREQ_TO_PERIOD, key=len, reverse=True):
        if f.endswith(k):
            return FREQ_TO_PERIOD[k]
    return 1


def _get_periods(freq: str) -> list[int]:
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIODS:
        return list(FREQ_TO_PERIODS[f])
    for k in sorted(FREQ_TO_PERIODS, key=len, reverse=True):
        if f.endswith(k):
            return list(FREQ_TO_PERIODS[k])
    return []


# ── Box-Cox ──────────────────────────────────────────────────────────────


def _bc_lambda(y: NDArray[np.floating]) -> float:
    yp = y[y > 0]
    if len(yp) < _MIN_POSITIVE_FOR_BC:
        return 1.0
    try:
        _, lam = scipy_boxcox(yp)
        return float(np.clip(lam, 0.0, 1.0))
    except (ValueError, RuntimeError):
        return 1.0


def _bc(y: NDArray[np.floating], lam: float) -> NDArray[np.floating]:
    y = np.maximum(y, _EPS_BOXCOX)
    return np.log(y) if lam == 0.0 else (y**lam - 1) / lam


def _bc_inv(z: NDArray[np.floating], lam: float) -> NDArray[np.floating]:
    if lam == 0.0:
        return np.exp(np.clip(z, -_BC_EXP_CLIP, _BC_EXP_CLIP))
    return np.maximum(z * lam + 1, _EPS) ** (1 / lam)


# ── Ridge with Soft-Average GCV ─────────────────────────────────────────


def _ridge_sa(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    float,
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Ridge regression with LOOCV soft-average over 25 log-spaced alphas.

    Under the LSR1 model, this is local linear regression at the boundary
    u=1 with bandwidth h=∞ (global fit).  The regularization alpha is
    selected by LOOCV soft-averaging.

    Returns (beta, loo_residuals, gcv_min, Vt, s, d_avg).
    LOO residuals are LWCP-normalized: e_i^LOO / sqrt(1 + h_ii).
    SVD components are returned for test-point leverage computation.
    Everything from one SVD.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2, Uty = s**2, U.T @ y
    alphas = np.logspace(_ALPHA_LOG_MIN, _ALPHA_LOG_MAX, _N_ALPHAS)

    # LOOCV for each alpha
    gcv = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        d = s2 / (s2 + a)
        h = (U**2) @ d
        r = y - U @ (d * Uty)
        gcv[i] = np.mean((r / np.maximum(1 - h, _EPS)) ** 2)

    # Softmax weights (temperature = gcv_min)
    gcv_min = gcv.min()
    log_w = -(gcv - gcv_min) / max(gcv_min, _EPS)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    # Weighted-average beta and hat-matrix diagonal
    beta = np.zeros(X.shape[1])
    d_avg = np.zeros(len(s))
    for wi, a in zip(w, alphas):
        if wi < _EPS_WEIGHT:
            continue
        d = s2 / (s2 + a)
        beta += wi * (Vt.T @ (d * Uty / np.maximum(s, _EPS)))
        d_avg += wi * d

    # LWCP-normalized LOO residuals: e_i^LOO / sqrt(1 + h_ii)
    # Under LWCP (Fadnavis et al., 2026), this normalization removes
    # leverage-dependent heteroscedasticity, producing approximately
    # exchangeable scores.  The test-point interval is then scaled by
    # sqrt(1 + h_test) to restore the correct variance at each horizon.
    residuals = y - X @ beta
    h_avg = (U**2) @ d_avg
    loo_raw = residuals / np.maximum(1 - h_avg, _EPS)
    loo = loo_raw / np.sqrt(np.maximum(1 + h_avg, _EPS))

    return beta, loo, gcv_min, Vt, s, d_avg


# ── Shape₂: MDL-Gated Prior Shrinkage ──────────────────────────────────


def _compute_shape2(
    L: NDArray[np.floating],
    cp: int,
    n_complete: int,
) -> NDArray[np.floating] | None:
    """Shape₂ with MDL-gated empirical Bayes shrinkage.

    Shape₂ = w × raw_proportions + (1-w) × prior
    w = nc₂ / (nc₂ + cp)

    The prior is selected by BIC (MDL): first harmonic (2 params) vs flat (0 params).
    When the harmonic is not justified by data, the flat prior S₂=1 keeps
    deseasonalization negligible — same MDL principle as BIC period selection.
    """
    nc2 = n_complete // cp
    if nc2 < 2:
        return None

    pos = np.arange(n_complete) % cp
    S2_raw = np.zeros(cp)
    for d in range(cp):
        vals = L[pos == d]
        S2_raw[d] = vals.mean() if len(vals) > 0 else 1.0
    raw_mean = S2_raw.mean()
    if raw_mean < _EPS:
        return None
    S2_raw = S2_raw / raw_mean

    # First harmonic fit
    t = np.arange(cp, dtype=float)
    cos_b = np.cos(2 * np.pi * t / cp)
    sin_b = np.sin(2 * np.pi * t / cp)
    S2_c = S2_raw - 1.0
    a = 2.0 * np.mean(S2_c * cos_b)
    b = 2.0 * np.mean(S2_c * sin_b)
    S2_harmonic = 1.0 + a * cos_b + b * sin_b

    # MDL gate: BIC selects harmonic (2 params) vs flat (0 params)
    RSS_flat: float = float(np.sum(S2_c**2))
    RSS_harmonic: float = float(np.sum((S2_raw - S2_harmonic) ** 2))
    bic_flat = cp * np.log(max(RSS_flat / cp, _EPS_LOG))
    bic_harmonic = cp * np.log(max(RSS_harmonic / cp, _EPS_LOG)) + 2 * np.log(cp)
    S2_prior = S2_harmonic if bic_harmonic < bic_flat else np.ones(cp)

    # Empirical Bayes weight
    w = nc2 / (nc2 + cp)
    S2 = w * S2_raw + (1 - w) * S2_prior

    S2 = np.maximum(S2, _EPS_SHAPE)
    S2 = S2 / S2.mean()
    return np.asarray(S2, dtype=np.float64)


# ── Period selection ────────────────────────────────────────────────────


def _select_period(
    y: NDArray[np.floating],
    n: int,
    freq: str,
) -> tuple[int, list[int], int, list[int]]:
    """MDL period selection via BIC on SVD spectrum.

    Returns (P, secondary, period, cal).
    """
    period = _get_period(freq)
    cal = _get_periods(freq)
    candidates = [p for p in cal if p >= 1 and n // p >= _MIN_COMPLETE] if cal else []
    if not candidates:
        candidates = [max(period, 1)] if n // max(period, 1) >= _MIN_COMPLETE else [1]

    if len(candidates) == 1:
        P = candidates[0]
    else:
        T_max = min(n, _MAX_COMPLETE * min(candidates))
        y_sel = y[-T_max:]
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = T_max // p_cand
            if nc < _MIN_COMPLETE:
                continue
            mat_c = y_sel[-(nc * p_cand) :].reshape(nc, p_cand).T
            s = np.linalg.svd(mat_c, compute_uv=False)
            rss1 = float(np.sum(s[1:] ** 2))
            T = nc * p_cand
            bic = T * np.log(max(rss1 / T, _EPS_LOG)) + (p_cand + nc - 1) * np.log(T)
            if bic < best_bic:
                best_P, best_bic = p_cand, bic
        P = best_P

    secondary = [p for p in cal if p != P and p > P] if cal else []
    return P, secondary, period, cal


def _estimate_shape(
    mat: NDArray[np.floating],
    n_complete: int,
    P: int,
    secondary: list[int],
    L: NDArray[np.floating],
    horizon: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Dirichlet-Multinomial empirical Bayes Shape estimation.

    Returns (S_forecast, S_hist, m).
    """
    K = min(_SHAPE_K, n_complete)
    recent = mat[:, -K:]
    totals = recent.sum(axis=0, keepdims=True)
    S_global = np.where(totals > _EPS, recent / totals, 1.0 / P).mean(axis=1)
    S_global /= max(S_global.sum(), _EPS)

    C = (
        secondary[0] // P
        if (secondary and secondary[0] % P == 0 and n_complete >= secondary[0] // P)
        else 1
    )
    m = int(np.ceil(horizon / P))

    if C <= 1:
        S_forecast = np.tile(S_global, (m, 1))
        S_hist = np.tile(S_global, (n_complete, 1))
    else:
        K_ds = min(K * C, n_complete)
        ds_mat = mat[:, -K_ds:]
        ds_L = L[-K_ds:]
        ds_ctx = np.arange(n_complete - K_ds, n_complete) % C

        ds_totals = ds_mat.sum(axis=0, keepdims=True)
        ds_props = np.where(ds_totals > _EPS, ds_mat / ds_totals, 1.0 / P)
        mp = ds_props.mean(axis=1)
        vp = ds_props.var(axis=1, ddof=1)
        valid = (mp > _EPS_SHAPE) & (vp > _EPS)
        kappa = (
            max(float(np.median(mp[valid] * (1 - mp[valid]) / vp[valid] - 1)), 0.0)
            if valid.sum() >= 2
            else 1e6
        )

        S_ctx = np.empty((C, P))
        for c_val in range(C):
            mask = ds_ctx == c_val
            if mask.sum() == 0:
                S_ctx[c_val] = S_global
            else:
                S_c = (kappa * S_global + ds_mat[:, mask].sum(axis=1)) / max(
                    kappa + ds_L[mask].sum(), _EPS
                )
                S_ctx[c_val] = S_c / max(S_c.sum(), _EPS)

        S_forecast = S_ctx[(n_complete + np.arange(m)) % C]
        S_hist = S_ctx[np.arange(n_complete) % C]

    return S_forecast, S_hist, m


def _compute_cross_periods(
    secondary: list[int],
    P: int,
    period: int,
    n_complete: int,
) -> tuple[list[int], int]:
    """Compute cross-period lags for Ridge features.

    Returns (cross_periods, max_cp).
    """
    cross_periods: list[int] = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})
    max_cp = max(cross_periods) if cross_periods else 0
    return cross_periods, max_cp


def _estimate_phi(L_bc: NDArray[np.floating]) -> float:
    """Estimate trend damping factor from lag-1 autocorrelation of diff(L_bc).

    Under the LSR1 model, L belongs to Hölder(2), so the trend change rate
    is bounded.  phi = max(rho_1(Delta L), 0) measures trend persistence:
    phi > 0 means the trend is self-reinforcing; phi = 0 means mean-reverting
    or noisy, warranting full damping of the linear extrapolation.
    """
    dL = np.diff(L_bc)
    if len(dL) < 5:
        return 0.0
    dL_c = dL - dL.mean()
    c0 = float(np.dot(dL_c, dL_c))
    if c0 < _EPS:
        return 0.0
    c1 = float(np.dot(dL_c[:-1], dL_c[1:]))
    return max(c1 / c0, 0.0)


# ── FLAIR core ───────────────────────────────────────────────────────────


def forecast(
    y_raw: ArrayLike,
    horizon: int,
    freq: str,
    n_samples: int = 200,
    seed: int | None = None,
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
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]

    # ── Reshape ──────────────────────────────────────────────────────
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)
    L = mat.sum(axis=0)

    S_forecast, S_hist, m = _estimate_shape(
        mat,
        n_complete,
        P,
        secondary,
        L,
        horizon,
    )

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

    X = np.zeros((n_train, nf))
    X[:, :nb] = base[start:]

    if _DIFF_TARGET and n_train >= 3:
        # ── LSR1 reparameterization ─────────────────────────────────
        # Under the LSR1 model, L ∈ Hölder(2) implies the Level is
        # smooth, so consecutive values satisfy L(i) ≈ L(i-1), i.e.,
        # the "natural" AR coefficient β₂ ≈ 1.  Standard Ridge shrinks
        # β₂ → 0 (stationarity prior), which is inconsistent.
        #
        # Reparameterize: let δ₂ = 1 - β₂.  Rewrite the model as
        #   ΔL_innov[i] = β₀ + β₁(i/n) − δ₂ L_innov[i-1] + β₃ lag_cp
        # Now Ridge shrinks δ₂ → 0, i.e., β₂ → 1 (random walk prior).
        # No magic numbers.  Standard isotropic Ridge.
        y_target = np.diff(L_innov[start - 1 :])  # ΔL_innov
        X[:, nb] = -L_innov[start - 1 : -1]  # negated lag1 → δ₂ coeff
        if max_cp >= 2:
            X[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]

        theta, loo_resid, _, Vt_r, s_r, d_avg_r = _ridge_sa(X, y_target)

        # Recover original parameterization
        beta = theta.copy()
        beta[nb] = 1.0 - theta[nb]  # β₂ = 1 − δ₂
    else:
        # ── Standard Ridge (fallback for very short series) ──────────
        X[:, nb] = L_innov[start - 1 : -1]
        if max_cp >= 2:
            X[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]
        # Short series: no PLOOCV (insufficient data for meaningful weights)
        beta, loo_resid, _, Vt_r, s_r, d_avg_r = _ridge_sa(X, L_innov[start:])

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
    h_test = np.empty(m)
    L_point = np.concatenate([L_innov, np.zeros(m)])
    for j in range(m):
        ti = n_complete + j
        pred_pt = beta[0] + beta[1] * _damped_trend[j] + beta[nb] * L_point[ti - 1]
        if max_cp >= 2:
            pred_pt += beta[nb + 1] * L_point[ti - max_cp]
        L_point[ti] = pred_pt

        x_j = np.zeros(nf)
        x_j[0] = 1.0
        x_j[1] = _damped_trend[j]
        if _DIFF_TARGET and n_train >= 3:
            x_j[nb] = -L_point[ti - 1]
        else:
            x_j[nb] = L_point[ti - 1]
        if max_cp >= 2:
            x_j[nb + 1] = L_point[ti - max_cp]

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
        )


# Backward compatibility
flair_forecast = forecast
