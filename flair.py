"""FLAIR: Factored Level And Interleaved Ridge.

A single-equation time series forecasting method that separates
WHAT (level) from HOW (shape) via period-aligned matrix reshaping.

    y(phase, period) = Level(period) × Shape(phase)

Level is forecast by Ridge regression with soft-average GCV.
Shape is estimated via Dirichlet-Multinomial empirical Bayes, with
context derived from the secondary period structure (e.g., day-of-week
for hourly data). When no secondary period exists, degenerates to
the simple K-period average.

One SVD. Zero hyperparameters. No neural network.

Example:
    >>> from flair import flair_forecast
    >>> samples = flair_forecast(y, horizon=24, freq='H')
    >>> point = samples.mean(axis=0)
"""

import numpy as np
from scipy.stats import boxcox as scipy_boxcox

# ── Calendar tables ──────────────────────────────────────────────────────

FREQ_TO_PERIOD = {
    'S': 60, 'T': 60, '5T': 12, '10T': 6, '15T': 4, '10S': 6,
    'H': 24, 'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'A': 1, 'Y': 1,
}

FREQ_TO_PERIODS = {
    '10S': [6, 360], 'S': [60], '5T': [12, 288], '10T': [6, 144],
    '15T': [4, 96], 'H': [24, 168], 'D': [7, 365], 'W': [52],
    'M': [12], 'Q': [4], 'A': [], 'Y': [],
}


def _resolve_freq(freq):
    return freq.upper().replace('MIN', 'T')


def _get_period(freq):
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIOD:
        return FREQ_TO_PERIOD[f]
    for k in sorted(FREQ_TO_PERIOD, key=len, reverse=True):
        if f.endswith(k):
            return FREQ_TO_PERIOD[k]
    return 1


def _get_periods(freq):
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIODS:
        return list(FREQ_TO_PERIODS[f])
    for k in sorted(FREQ_TO_PERIODS, key=len, reverse=True):
        if f.endswith(k):
            return list(FREQ_TO_PERIODS[k])
    return []


# ── Box-Cox ──────────────────────────────────────────────────────────────

def _bc_lambda(y):
    yp = y[y > 0]
    if len(yp) < 10:
        return 1.0
    try:
        _, lam = scipy_boxcox(yp)
        return float(np.clip(lam, 0.0, 1.0))
    except Exception:
        return 1.0


def _bc(y, lam):
    y = np.maximum(y, 1e-8)
    return np.log(y) if lam == 0.0 else (y ** lam - 1) / lam


def _bc_inv(z, lam):
    if lam == 0.0:
        return np.exp(np.clip(z, -30, 30))
    return np.maximum(z * lam + 1, 1e-10) ** (1 / lam)


# ── Ridge with Soft-Average GCV ─────────────────────────────────────────

def _ridge_sa(X, y):
    """Ridge regression with GCV soft-average over 25 log-spaced alphas.

    Returns (beta, loo_residuals, gcv_min).
    Everything from one SVD.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2, Uty = s ** 2, U.T @ y
    alphas = np.logspace(-4, 4, 25)

    # GCV for each alpha
    gcv = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        d = s2 / (s2 + a)
        h = (U ** 2) @ d
        r = y - U @ (d * Uty)
        gcv[i] = np.mean((r / np.maximum(1 - h, 1e-10)) ** 2)

    # Softmax weights (temperature = gcv_min)
    gcv_min = gcv.min()
    log_w = -(gcv - gcv_min) / max(gcv_min, 1e-10)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    # Weighted-average beta and hat-matrix diagonal
    beta = np.zeros(X.shape[1])
    d_avg = np.zeros(len(s))
    for wi, a in zip(w, alphas):
        if wi < 1e-15:
            continue
        d = s2 / (s2 + a)
        beta += wi * (Vt.T @ (d * Uty / s))
        d_avg += wi * d

    # LOO residuals
    residuals = y - X @ beta
    h_avg = (U ** 2) @ d_avg
    loo = residuals / np.maximum(1 - h_avg, 1e-10)

    return beta, loo, gcv_min


# ── Fourier periods (for fallback) ──────────────────────────────────────

def _fourier_periods(calendar_periods, T):
    c = set()
    for p in calendar_periods:
        if p < 2:
            continue
        if p <= T // 2:
            c.add(p)
        if p * 2 <= T // 2:
            c.add(p * 2)
        if p // 2 >= 2:
            c.add(p // 2)
    return sorted(c)


# ── Fallback: Fourier-Lag Ridge (for non-periodic series) ───────────────

def _fallback_forecast(y, horizon, period, freq, n_samples):
    """Ridge with Fourier + lag features. Used when reshape is not possible."""
    n = len(y)
    lam = _bc_lambda(y)
    y_t = _bc(y + 1, lam)
    periods = _fourier_periods(_get_periods(freq), n)
    start = max(1, period) if period >= 2 else 1

    if n <= start + horizon:
        fc = np.full(horizon, y[-1])
        sigma = max(np.std(np.diff(y[-min(50, n):])), 1e-6) if n > 1 else 1.0
        return np.maximum(0, np.array([
            fc + np.random.normal(0, sigma, horizon) for _ in range(n_samples)
        ]))

    # Features
    t = np.arange(n, dtype=float)
    trend = t / n
    cols = [np.ones(n), trend]
    for p in periods:
        cols.append(np.cos(2 * np.pi * t / p))
        cols.append(np.sin(2 * np.pi * t / p))
    nb = len(cols)
    base = np.column_stack(cols)
    nf = nb + 1 + (1 if period >= 2 else 0)

    X = np.zeros((n - start, nf))
    X[:, :nb] = base[start:]
    X[:, nb] = y_t[start - 1 : -1]
    if period >= 2:
        X[:, nb + 1] = y_t[start - period : n - period]

    beta, loo_resid, _ = _ridge_sa(X, y_t[start:])

    # Recursive forecast
    y_ext = np.concatenate([y_t, np.zeros(horizon)])
    for h in range(horizon):
        ti = n + h
        x = np.zeros(nf)
        x[0], x[1] = 1.0, ti / n
        col = 2
        for p in periods:
            x[col] = np.cos(2 * np.pi * ti / p)
            x[col + 1] = np.sin(2 * np.pi * ti / p)
            col += 2
        x[nb] = y_ext[ti - 1]
        if period >= 2:
            x[nb + 1] = y_ext[ti - period]
        y_ext[ti] = x @ beta

    point = np.maximum(_bc_inv(y_ext[n:], lam) - 1, 0.0)
    return _conformal_samples(point, loo_resid, y_t[start:], lam, y, n_samples, horizon)


# ── Conformal sample generation ─────────────────────────────────────────

def _conformal_samples(point_fc, loo_resid, y_train, lam, y_raw, n_samples, horizon):
    fitted = y_train - loo_resid
    loo_orig = _bc_inv(y_train, lam) - _bc_inv(fitted, lam)
    loo_orig = loo_orig[np.isfinite(loo_orig)]

    if len(loo_orig) < 3:
        loo_orig = np.array([-1, 0, 1]) * max(np.mean(point_fc) * 0.05, 1e-6)

    recent = loo_orig[-min(200, len(loo_orig)):]
    drawn = np.random.choice(recent, size=(n_samples, horizon), replace=True)
    jitter = np.random.normal(0, np.std(recent) * 0.1, size=(n_samples, horizon))
    samples = np.maximum(0, point_fc[np.newaxis, :] + drawn + jitter)

    rm = np.max(y_raw[-max(horizon * 2, 50):])
    if rm > 0:
        samples = np.clip(samples, 0, rm * 3)

    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


# ── FLAIR core ───────────────────────────────────────────────────────────

def flair_forecast(y_raw, horizon, freq, n_samples=20):
    """FLAIR: Factored Level And Interleaved Ridge.

    Parameters
    ----------
    y_raw : array-like, shape (n,)
        Historical observations (non-negative).
    horizon : int
        Number of steps to forecast.
    freq : str
        Frequency string ('H', 'D', 'W', 'M', '5T', etc.).
    n_samples : int
        Number of sample paths for probabilistic forecast.

    Returns
    -------
    samples : ndarray, shape (n_samples, horizon)
        Probabilistic forecast sample paths.
    """
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    # Location shift: make all values positive for Box-Cox compatibility
    y_floor = y.min()
    y_shift = max(1 - y_floor, 1.0)  # shift so min(y) >= 1
    y = y + y_shift
    n = len(y)
    period = _get_period(freq)
    cal = _get_periods(freq)
    P = cal[0] if cal else period
    secondary = cal[1:] if len(cal) > 1 else []

    # Context limit
    max_ctx = max(period * 10, 500) if period >= 2 else 500
    if n > max_ctx:
        y = y[-max_ctx:]
        n = len(y)

    # Fallback for non-periodic or insufficient data
    n_complete = n // P if P >= 2 else 0
    if P < 2 or n_complete < 3:
        return _fallback_forecast(y, horizon, period, freq, n_samples)

    # Allow more context for Level series (up to 500 periods)
    if n_complete > 500:
        y = y[-(500 * P):]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]

    # ── Reshape ──────────────────────────────────────────────────────
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

    # ── Shape: Dirichlet-Multinomial empirical Bayes ─────────────────
    K = min(5, n_complete)
    recent = mat[:, -K:]
    totals = recent.sum(axis=0, keepdims=True)
    S_global = np.where(totals > 1e-10, recent / totals, 1.0 / P).mean(axis=1)
    S_global /= max(S_global.sum(), 1e-10)

    # ── Level: period totals ──────────────────────────────────────────
    L = mat.sum(axis=0)

    # ── Context = period position mod C (secondary / primary period) ─
    C = secondary[0] // P if (secondary and secondary[0] % P == 0 and
                               n_complete >= secondary[0] // P) else 1
    m = int(np.ceil(horizon / P))

    if C <= 1:
        S_forecast = np.tile(S_global, (m, 1))
        S_hist = np.tile(S_global, (n_complete, 1))
    else:
        # Data window: K*C recent periods → K observations per context
        K_ds = min(K * C, n_complete)
        ds_mat = mat[:, -K_ds:]
        ds_L = L[-K_ds:]
        ds_ctx = np.arange(n_complete - K_ds, n_complete) % C

        # Kappa: Dirichlet concentration from method-of-moments
        ds_totals = ds_mat.sum(axis=0, keepdims=True)
        ds_props = np.where(ds_totals > 1e-10, ds_mat / ds_totals, 1.0 / P)
        mp = ds_props.mean(axis=1)
        vp = ds_props.var(axis=1, ddof=1)
        valid = (mp > 1e-6) & (vp > 1e-10)
        kappa = max(float(np.median(
            mp[valid] * (1 - mp[valid]) / vp[valid] - 1
        )), 0.0) if valid.sum() >= 2 else 1e6

        # Dirichlet posterior mean per context
        S_ctx = np.empty((C, P))
        for c_val in range(C):
            mask = (ds_ctx == c_val)
            if mask.sum() == 0:
                S_ctx[c_val] = S_global
            else:
                S_c = (kappa * S_global + ds_mat[:, mask].sum(axis=1)) / \
                      max(kappa + ds_L[mask].sum(), 1e-10)
                S_ctx[c_val] = S_c / max(S_c.sum(), 1e-10)

        S_forecast = S_ctx[(n_complete + np.arange(m)) % C]
        S_hist = S_ctx[np.arange(n_complete) % C]

    # ── Box-Cox → NLinear (L already shifted positive) ──────────────────
    lam = _bc_lambda(L)
    L_bc = _bc(L, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # ── Cross-period features ────────────────────────────────────────
    cross_periods = [sp // P for sp in secondary if 2 <= sp // P <= n_complete // 2]
    max_cp = max(cross_periods) if cross_periods else 0
    start = max(1, max_cp) if max_cp >= 2 else 1

    if n_complete <= start + 1:
        return _fallback_forecast(y, horizon, period, freq, n_samples)

    n_train = n_complete - start
    t = np.arange(n_complete, dtype=float)
    trend = t / n_complete
    cols = [np.ones(n_complete), trend]
    for cp in cross_periods:
        cols.append(np.cos(2 * np.pi * t / cp))
        cols.append(np.sin(2 * np.pi * t / cp))
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

    # ── Forecast Level ───────────────────────────────────────────────
    L_ext = np.concatenate([L_innov, np.zeros(m)])

    for j in range(m):
        ti = n_complete + j
        x = np.zeros(nf)
        x[0], x[1] = 1.0, ti / n_complete
        col = 2
        for cp in cross_periods:
            x[col] = np.cos(2 * np.pi * ti / cp)
            x[col + 1] = np.sin(2 * np.pi * ti / cp)
            col += 2
        x[nb] = L_ext[ti - 1]
        if max_cp >= 2:
            x[nb + 1] = L_ext[ti - max_cp]
        L_ext[ti] = x @ beta

    L_hat = _bc_inv(L_ext[n_complete : n_complete + m] + last_L, lam)

    # ── Reconstruct: Level × Shape, undo shift ────────────────────────
    point_fc = (L_hat[:, np.newaxis] * S_forecast).reshape(-1)[:horizon] - y_shift

    # ── LOO Conformal (context-specific Shape for residuals) ──────────
    L_fitted = _bc_inv(L_innov[start:] - loo_resid + last_L, lam)
    loo_orig = []
    for i in range(n_train):
        for j in range(P):
            r = mat[j, start + i] - L_fitted[i] * S_hist[start + i, j]
            if np.isfinite(r):
                loo_orig.append(r)

    loo_orig = np.array(loo_orig) if loo_orig else np.array([-1, 0, 1]) * max(abs(np.mean(point_fc)) * 0.05, 1e-6)
    if len(loo_orig) < 3:
        loo_orig = np.array([-1, 0, 1]) * max(abs(np.mean(point_fc)) * 0.05, 1e-6)

    recent_loo = loo_orig[-min(200, len(loo_orig)):]
    drawn = np.random.choice(recent_loo, size=(n_samples, horizon), replace=True)
    jitter = np.random.normal(0, np.std(recent_loo) * 0.1, size=(n_samples, horizon))
    samples = point_fc[np.newaxis, :] + drawn + jitter

    y_orig = y - y_shift
    y_lo, y_hi = y_orig[-max(horizon * 2, 50):].min(), y_orig[-max(horizon * 2, 50):].max()
    y_range = max(y_hi - y_lo, 1e-6)
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)

    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
