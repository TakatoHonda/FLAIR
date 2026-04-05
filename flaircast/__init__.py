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

__version__ = "0.2.0"
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
_SHAPE_K = 2  # Number of recent periods for Shape estimation (insensitive; see paper)
_PHASE_NOISE_K = 50  # Number of recent periods for phase noise
_N_ALPHAS = 25  # Number of log-spaced GCV alphas
_ALPHA_LOG_MIN = -4  # log10 of minimum Ridge alpha
_ALPHA_LOG_MAX = 4  # log10 of maximum Ridge alpha

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
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Ridge regression with GCV soft-average over 25 log-spaced alphas.

    Returns (beta, loo_residuals, gcv_min).
    Everything from one SVD.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2, Uty = s**2, U.T @ y
    alphas = np.logspace(_ALPHA_LOG_MIN, _ALPHA_LOG_MAX, _N_ALPHAS)

    # GCV for each alpha
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
        beta += wi * (Vt.T @ (d * Uty / s))
        d_avg += wi * d

    # Predictive residuals: LOO scaled by √(1−h²) to convert from
    # training-point LOO variance σ²/(1−h) to prediction-point
    # variance σ²(1+h).  Ratio = (1+h)(1−h) = 1−h², so the
    # standard-deviation correction is √(1−h²).
    residuals = y - X @ beta
    h_avg = (U**2) @ d_avg
    loo = residuals / np.maximum(1 - h_avg, _EPS)
    loo *= np.sqrt(np.maximum(1 - h_avg * h_avg, _EPS))

    return beta, loo, gcv_min


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


def _estimate_gamma(mat: NDArray[np.floating], P: int, n_complete: int) -> float:
    """Estimate seasonal strength γ ∈ [0, 1] from rank-1 explained variance.

    γ = (r₁ − r_rand) / (1 − r_rand)

    where r₁ = σ₁² / Σσᵢ² is the fraction of variance captured by the
    first singular value, and r_rand ≈ 1/min(P, n_complete) is the expected
    ratio for a random matrix (Marchenko-Pastur baseline).

    When γ ≈ 1 the rank-1 seasonal pattern is dominant; when γ ≈ 0 the
    period-folded matrix has no low-rank structure and Shape should be
    dampened toward uniform.  This follows the same MDL principle used
    for period selection and Shape₂ gating.
    """
    if P < 2 or n_complete < _MIN_COMPLETE:
        return 1.0  # No meaningful periodicity check possible

    s = np.linalg.svd(mat, compute_uv=False)
    total = float(np.sum(s**2))
    if total < _EPS_LOG:
        return 1.0

    rank1_ratio = float(s[0] ** 2 / total)
    random_baseline = 1.0 / min(P, n_complete)
    gamma = (rank1_ratio - random_baseline) / max(1.0 - random_baseline, _EPS)
    return float(np.clip(gamma, 0.0, 1.0))


def _dampen_shape(S: NDArray[np.floating], gamma: float) -> NDArray[np.floating]:
    """Apply Shape dampening S^γ and re-normalize.

    S can be 1D (P,) or 2D (m, P).
    γ=1 returns S unchanged; γ=0 returns uniform 1/P.
    Intermediate values smoothly reduce seasonal contrast while
    preserving the relative phase ordering.
    """
    if gamma >= 1.0 - _EPS:
        return S
    S_d = np.power(np.maximum(S, _EPS_LOG), gamma)
    if S_d.ndim == 1:
        S_d /= max(float(S_d.sum()), _EPS)
    else:
        row_sums = S_d.sum(axis=1, keepdims=True)
        S_d /= np.maximum(row_sums, _EPS)
    return S_d


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
    y = np.nan_to_num(y_arr, nan=0.0)
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

    # ── Seasonal strength: dampen Shape when rank-1 structure is weak ────
    gamma = _estimate_gamma(mat, P, n_complete)
    S_forecast = _dampen_shape(S_forecast, gamma)
    S_hist = _dampen_shape(S_hist, gamma)

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
    X[:, nb] = L_innov[start - 1 : -1]
    if max_cp >= 2:
        X[:, nb + 1] = L_innov[start - max_cp : n_complete - max_cp]

    # ── One Ridge SA ─────────────────────────────────────────────────
    beta, loo_resid, _ = _ridge_sa(X, L_innov[start:])

    # ── Stochastic Level paths (recursive noise injection) ────────────
    # The same recursion as the point forecast, but with LOO residual
    # noise injected at each step.  Errors propagate through the lag
    # features exactly as the Ridge dynamics dictate — no √step, no
    # scaling formula.  Mean-reverting series naturally saturate;
    # random-walk series naturally grow as √step.
    noise_pool = loo_resid[rng.randint(0, len(loo_resid), size=(n_samples, m))]
    L_paths = np.column_stack(
        [np.tile(L_innov, (n_samples, 1)), np.zeros((n_samples, m))]
    )  # (n_samples, n_complete + m)

    for j in range(m):
        ti = n_complete + j
        pred = beta[0] + beta[1] * (ti / n_complete) + beta[nb] * L_paths[:, ti - 1]
        if max_cp >= 2:
            pred += beta[nb + 1] * L_paths[:, ti - max_cp]
        L_paths[:, ti] = pred + noise_pool[:, j]

    # Point forecast = median path (noise_pool row 0 is arbitrary, use mean)
    L_hat_all = _bc_inv(L_paths[:, n_complete : n_complete + m] + last_L, lam)

    if use_deseason:
        assert S2 is not None
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # ── Phase noise (SVD Residual Quantiles, unchanged) ─────────────
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

    y_orig = y - y_shift
    lookback = max(horizon * 2, _PHASE_NOISE_K)
    y_lo, y_hi = y_orig[-lookback:].min(), y_orig[-lookback:].max()
    y_range = max(y_hi - y_lo, _EPS_SHAPE)
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)

    return np.asarray(np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)


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
