#!/usr/bin/env python3
"""FLAIR-DS: Dirichlet Shape — context-dependent Shape via secondary period structure.

Extends V9 by replacing the fixed Shape average with Dirichlet-Multinomial
empirical Bayes shrinkage. Context is derived purely from the calendar table:

  C = secondary_period / primary_period
  context(k) = k % C

For hourly data (P=24, secondary=168): C=7, context = position within the week.
For 15T data (P=4, secondary=96): C=24, context = hour of day.

When C < 2 or data is insufficient, degenerates to V9 exactly.

Usage:
  python -u research/benchmarks/run_gift_eval_flair_ds.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

os.environ['GIFT_EVAL'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'gift-eval')

from gift_eval.data import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model import evaluate_model
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

# Import all V9 utilities
from run_gift_eval_flar9 import (
    get_period, get_periods, FREQ_TO_PERIOD, FREQ_TO_PERIODS,
    _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg,
    _multi_fourier_periods, flar_v5, flar_v9,
    Predictor, METRICS, CONFIGS, _get_sn, run_variant,
)

warnings.filterwarnings('ignore')

# =========================================================================
# FLAIR-DS: Dirichlet Shape + Sinusoidal Prior Shrinkage for Shape₂
# =========================================================================

def _compute_shape2(L, cp, n_complete):
    """Compute Shape₂ with MDL-gated sinusoidal prior shrinkage.

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
    if raw_mean < 1e-10:
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
    RSS_flat = np.sum(S2_c ** 2)
    RSS_harmonic = np.sum((S2_raw - S2_harmonic) ** 2)
    bic_flat = cp * np.log(max(RSS_flat / cp, 1e-30))
    bic_harmonic = cp * np.log(max(RSS_harmonic / cp, 1e-30)) + 2 * np.log(cp)
    S2_prior = S2_harmonic if bic_harmonic < bic_flat else np.ones(cp)

    # Bayesian shrinkage: w = data_strength / (data + prior)
    w = nc2 / (nc2 + cp)
    S2 = w * S2_raw + (1 - w) * S2_prior

    S2 = np.maximum(S2, 1e-6)
    S2 = S2 / S2.mean()
    return S2


def flair_ds(y_raw, horizon, period, freq_str, n_samples=500):
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    # Location shift: make all values positive for multiplicative decomposition
    y_floor = y.min()
    y_shift = max(1 - y_floor, 1.0)  # shift so min(y) >= 1
    y = y + y_shift
    n = len(y)

    cal = get_periods(freq_str)
    candidates = cal if cal else [period]
    candidates = [p for p in candidates if p >= 1 and n // p >= 3]
    if not candidates:
        candidates = [1]  # P=1 fallback

    # --- MDL/BIC period selection: choose P that best supports rank-1 ---
    if len(candidates) == 1:
        P = candidates[0]
    else:
        # Use the same data window for all candidates (fair comparison)
        T_max = min(n, 500 * min(candidates))
        y_sel = y[-T_max:]
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = T_max // p_cand
            if nc < 5:  # need >= K=5 periods for reliable BIC
                continue
            mat_c = y_sel[-(nc * p_cand):].reshape(nc, p_cand).T
            s = np.linalg.svd(mat_c, compute_uv=False)
            rss1 = np.sum(s[1:] ** 2)
            T = nc * p_cand
            penalty = (p_cand + nc - 1) * np.log(T)
            bic = T * np.log(max(rss1 / T, 1e-30)) + penalty
            if bic < best_bic:
                best_P, best_bic = p_cand, bic
        P = best_P

    secondary = [p for p in cal[1:] if p != P] if len(cal) > 1 else []

    n_complete = n // P
    if n_complete < 3:
        P = 1
        secondary = []
        n_complete = n
        if n_complete < 3:
            # Absolute minimum: repeat last value with noise
            fc = np.full(horizon, y[-1] - y_shift)
            sigma = max(np.std(np.diff(y[-min(50, n):])), 1e-6) if n > 1 else 1.0
            return np.clip(np.array([fc + np.random.normal(0, sigma, horizon)
                                     for _ in range(n_samples)]),
                           fc.mean() - sigma * 10, fc.mean() + sigma * 10)

    if n_complete > 500:
        y = y[-(500 * P):]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]

    # --- Reshape (P=1: trivial, Level = raw series, Shape = [1]) ---
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

    # --- Shape ---
    K = min(5, n_complete)
    recent = mat[:, -K:]
    recent_totals = recent.sum(axis=0, keepdims=True)
    proportions = np.where(recent_totals > 1e-10,
                           recent / recent_totals,
                           1.0 / P)
    S_global = proportions.mean(axis=1)
    S_global = S_global / max(S_global.sum(), 1e-10)

    # --- Level ---
    L = mat.sum(axis=0)

    # --- Dirichlet Shape context ---
    C = secondary[0] // P if (secondary and secondary[0] % P == 0 and
                               n_complete >= secondary[0] // P) else 1
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
        ds_props = np.where(ds_totals > 1e-10, ds_mat / ds_totals, 1.0 / P)
        mp = ds_props.mean(axis=1)
        vp = ds_props.var(axis=1, ddof=1)
        valid = (mp > 1e-6) & (vp > 1e-10)
        kappa = max(float(np.median(
            mp[valid] * (1 - mp[valid]) / vp[valid] - 1
        )), 0.0) if valid.sum() >= 2 else 1e6

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

    # --- Cross-periods ---
    cross_periods = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})

    max_cp = max(cross_periods) if cross_periods else 0

    # --- Shape₂: sinusoidal prior shrinkage (deseasonalize Level) ---
    cp_main = cross_periods[0] if cross_periods else 0
    S2 = None
    use_deseason = False
    if cp_main >= 2:
        S2 = _compute_shape2(L, cp_main, n_complete)
        if S2 is not None:
            use_deseason = True

    if use_deseason:
        pos = np.arange(n_complete) % cp_main
        L_work = L / np.maximum(S2[pos], 1e-10)
    else:
        L_work = L

    # --- Box-Cox on (deseasonalized) Level ---
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)

    # --- NLinear on Level ---
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # --- Features: intercept + trend + lags (no Fourier — Shape₂ handles it) ---
    start = max(1, max_cp) if max_cp >= 2 else 1

    # If not enough training data, drop cross-period lag
    if n_complete - start < 3 and max_cp >= 2:
        max_cp = 0
        start = 1

    n_train = n_complete - start
    if n_train < 2:
        # Extreme fallback: random walk (no Ridge possible)
        L_hat_raw = np.full(m, _bc_inv(last_L, lam))
        if use_deseason:
            forecast_pos = (n_complete + np.arange(m)) % cp_main
            L_hat = L_hat_raw * S2[forecast_pos]
        else:
            L_hat = L_hat_raw
        fc = (L_hat[:, np.newaxis] * S_forecast).reshape(-1)[:horizon] - y_shift
        sigma = max(np.std(np.diff(y[-min(50, n):])), 1e-6) if n > 1 else 1.0
        return np.clip(np.array([fc + np.random.normal(0, sigma, horizon)
                                 for _ in range(n_samples)]),
                       fc - sigma * 10, fc + sigma * 10)

    t = np.arange(n_complete, dtype=float)
    trend = t / float(n_complete)
    cols = [np.ones(n_complete), trend]
    nb = len(cols)
    base = np.column_stack(cols)

    n_lag = 1 + (1 if max_cp >= 2 else 0)
    nf = nb + n_lag

    X = np.zeros((n_train, nf))
    X[:, :nb] = base[start:]
    X[:, nb] = L_innov[start-1:-1]
    if max_cp >= 2:
        X[:, nb+1] = L_innov[start-max_cp:n_complete-max_cp]
    y_train = L_innov[start:]

    # --- ONE Ridge SA ---
    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # --- Stochastic Level paths (recursive noise injection) ---
    # Same recursion as point forecast, with LOO residual noise injected.
    # Errors propagate through lag features via the Ridge dynamics.
    noise_pool = loo_resid[np.random.randint(0, len(loo_resid),
                                             size=(n_samples, m))]
    L_paths = np.column_stack([
        np.tile(L_innov, (n_samples, 1)),
        np.zeros((n_samples, m))
    ])

    for j in range(m):
        ti = n_complete + j
        pred = beta[0] + beta[1] * (ti / float(n_complete)) \
             + beta[nb] * L_paths[:, ti - 1]
        if max_cp >= 2:
            pred += beta[nb + 1] * L_paths[:, ti - max_cp]
        L_paths[:, ti] = pred + noise_pool[:, j]

    L_hat_all = _bc_inv(L_paths[:, n_complete:n_complete + m] + last_L, lam)

    if use_deseason:
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # --- Phase noise (SVD Residual Quantiles) ---
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(50, n_complete)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), 1e-8)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    R_flat = R.ravel()
    raw_idx = np.random.randint(0, K_r, size=(n_samples, horizon))
    phase_noise = R_flat[phase_idx[np.newaxis, :] * K_r + raw_idx]

    # Assemble: Level_path × Shape × (1 + phase_noise)
    S_h = S_forecast[step_idx, phase_idx]

    samples = (L_hat_all[:, step_idx]
               * S_h[np.newaxis, :]
               * (1 + phase_noise)
               - y_shift)

    y_orig = y - y_shift
    y_lo, y_hi = y_orig[-max(horizon*2,50):].min(), y_orig[-max(horizon*2,50):].max()
    y_range = max(y_hi - y_lo, 1e-6)
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================================
# Evaluation
# =========================================================================

    # V9 results from previous run (avoid re-running)
V9_RESULTS = [
    {'config': 'bitbrains_fast_storage/5T/short', 'mase': 1.1302, 'crps': 0.5324, 'sn_mase': None, 'sn_crps': None, 'time': 166, 'term': 'short'},
    {'config': 'bitbrains_rnd/H/short', 'mase': 6.6517, 'crps': 0.7322, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'short'},
    {'config': 'bizitobs_l2c/5T/short', 'mase': 0.4083, 'crps': 0.1187, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'short'},
    {'config': 'electricity/H/short', 'mase': 1.1967, 'crps': 0.0870, 'sn_mase': None, 'sn_crps': None, 'time': 69, 'term': 'short'},
    {'config': 'solar/H/short', 'mase': 0.9271, 'crps': 0.4132, 'sn_mase': None, 'sn_crps': None, 'time': 13, 'term': 'short'},
    {'config': 'car_parts/M/short', 'mase': 1.1880, 'crps': 1.3447, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'short'},
    {'config': 'restaurant/D/short', 'mase': 0.7563, 'crps': 0.2915, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'short'},
    {'config': 'hospital/M/short', 'mase': 0.8907, 'crps': 0.0662, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'short'},
    {'config': 'saugeen/D/short', 'mase': 3.3422, 'crps': 0.4452, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'short'},
    {'config': 'm4_yearly/A/short', 'mase': 3.6131, 'crps': 0.1503, 'sn_mase': None, 'sn_crps': None, 'time': 18, 'term': 'short'},
    {'config': 'm4_monthly/M/short', 'mase': 1.2127, 'crps': 0.1224, 'sn_mase': None, 'sn_crps': None, 'time': 39, 'term': 'short'},
    {'config': 'm4_hourly/H/short', 'mase': 1.1969, 'crps': 0.0389, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'short'},
    {'config': 'sz_taxi/15T/short', 'mase': 0.6112, 'crps': 0.2290, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'short'},
    {'config': 'electricity/H/medium', 'mase': 1.2730, 'crps': 0.0938, 'sn_mase': None, 'sn_crps': None, 'time': 28, 'term': 'medium'},
    {'config': 'electricity/H/long', 'mase': 1.4098, 'crps': 0.1109, 'sn_mase': None, 'sn_crps': None, 'time': 18, 'term': 'long'},
    {'config': 'solar/H/medium', 'mase': 1.0006, 'crps': 0.4442, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'medium'},
    {'config': 'loop_seattle/H/medium', 'mase': 1.4605, 'crps': 0.0870, 'sn_mase': None, 'sn_crps': None, 'time': 3, 'term': 'medium'},
    {'config': 'bizitobs_l2c/5T/medium', 'mase': 0.8050, 'crps': 0.4152, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'medium'},
    {'config': 'sz_taxi/15T/medium', 'mase': 0.5926, 'crps': 0.2288, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'medium'},
]
# Fill in SN values
for r in V9_RESULTS:
    sn_m, sn_c = _get_sn(r['config'])
    r['sn_mase'] = sn_m; r['sn_crps'] = sn_c

    # DS results from previous run (K*C window, no velocity)
DS_RESULTS = [
    {'config': 'bitbrains_fast_storage/5T/short', 'mase': 1.1400, 'crps': 0.5318, 'sn_mase': None, 'sn_crps': None, 'time': 193, 'term': 'short'},
    {'config': 'bitbrains_rnd/H/short', 'mase': 6.7140, 'crps': 0.7763, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'short'},
    {'config': 'bizitobs_l2c/5T/short', 'mase': 0.4071, 'crps': 0.1192, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'short'},
    {'config': 'electricity/H/short', 'mase': 1.0991, 'crps': 0.0842, 'sn_mase': None, 'sn_crps': None, 'time': 74, 'term': 'short'},
    {'config': 'solar/H/short', 'mase': 0.9059, 'crps': 0.4035, 'sn_mase': None, 'sn_crps': None, 'time': 14, 'term': 'short'},
    {'config': 'car_parts/M/short', 'mase': 1.1891, 'crps': 1.3504, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'short'},
    {'config': 'restaurant/D/short', 'mase': 0.7524, 'crps': 0.2906, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'short'},
    {'config': 'hospital/M/short', 'mase': 0.8853, 'crps': 0.0661, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'short'},
    {'config': 'saugeen/D/short', 'mase': 3.3611, 'crps': 0.4495, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'short'},
    {'config': 'm4_yearly/A/short', 'mase': 3.6133, 'crps': 0.1502, 'sn_mase': None, 'sn_crps': None, 'time': 19, 'term': 'short'},
    {'config': 'm4_monthly/M/short', 'mase': 1.2125, 'crps': 0.1224, 'sn_mase': None, 'sn_crps': None, 'time': 42, 'term': 'short'},
    {'config': 'm4_hourly/H/short', 'mase': 1.1763, 'crps': 0.0395, 'sn_mase': None, 'sn_crps': None, 'time': 1, 'term': 'short'},
    {'config': 'sz_taxi/15T/short', 'mase': 0.5980, 'crps': 0.2256, 'sn_mase': None, 'sn_crps': None, 'time': 3, 'term': 'short'},
    {'config': 'electricity/H/medium', 'mase': 1.2374, 'crps': 0.0944, 'sn_mase': None, 'sn_crps': None, 'time': 30, 'term': 'medium'},
    {'config': 'electricity/H/long', 'mase': 1.3707, 'crps': 0.1110, 'sn_mase': None, 'sn_crps': None, 'time': 19, 'term': 'long'},
    {'config': 'solar/H/medium', 'mase': 0.9886, 'crps': 0.4397, 'sn_mase': None, 'sn_crps': None, 'time': 2, 'term': 'medium'},
    {'config': 'loop_seattle/H/medium', 'mase': 1.1920, 'crps': 0.0880, 'sn_mase': None, 'sn_crps': None, 'time': 4, 'term': 'medium'},
    {'config': 'bizitobs_l2c/5T/medium', 'mase': 0.7993, 'crps': 0.4101, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'medium'},
    {'config': 'sz_taxi/15T/medium', 'mase': 0.5883, 'crps': 0.2286, 'sn_mase': None, 'sn_crps': None, 'time': 0, 'term': 'medium'},
]
for r in DS_RESULTS:
    sn_m, sn_c = _get_sn(r['config'])
    r['sn_mase'] = sn_m; r['sn_crps'] = sn_c


if __name__ == '__main__':
    total_start = time.perf_counter()
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    all_results = {'V9': V9_RESULTS, 'DS': DS_RESULTS}

    # Run FLAIR (MDL period selection) only
    print(f"\n{'='*70}")
    print(f"  Variant: FLAIR (MDL period selection)")
    print(f"{'='*70}")
    all_results['MDL'] = run_variant('MDL', flair_ds)

    for tf, tl in [('short','SHORT'),('medium','MEDIUM'),('long','LONG')]:
        print(f"\n{'='*70}")
        print(f"SUMMARY — {tl} horizon")
        print(f"{'='*70}")
        for label, res in all_results.items():
            f = [r for r in res if r['term']==tf and r['sn_mase'] is not None]
            if f:
                rm = [r['mase']/r['sn_mase'] for r in f]
                rc = [r['crps']/r['sn_crps'] for r in f]
                print(f"  {label:4s}: relMASE={gm(rm):.4f}  relCRPS={gm(rc):.4f}  ({sum(r['time'] for r in f):.0f}s, {len(f)} configs)")

    print(f"\n{'='*70}")
    print("OVERALL")
    print(f"{'='*70}")
    for label, res in all_results.items():
        v = [r for r in res if r['sn_mase'] is not None]
        if v:
            rm = [r['mase']/r['sn_mase'] for r in v]
            rc = [r['crps']/r['sn_crps'] for r in v]
            print(f"  {label:4s}: relMASE={gm(rm):.4f}  relCRPS={gm(rc):.4f}  ({sum(r['time'] for r in v):.0f}s)")

    print(f"\n{'='*70}")
    print("DELTA MDL vs DS")
    print(f"{'='*70}")
    dsm = {r['config']: r for r in all_results.get('DS', [])}
    for r in all_results.get('MDL', []):
        ds = dsm.get(r['config'])
        if ds:
            d = r['mase'] - ds['mase']
            mk = "<<" if d<-0.005 else (">>" if d>0.005 else "==")
            print(f"  {r['config']:40s} [{r['term']:6s}] {d:+.4f} {mk}  (DS={ds['mase']:.3f} -> MDL={r['mase']:.3f})")

    print(f"\nWall time: {time.perf_counter()-total_start:.0f}s")
