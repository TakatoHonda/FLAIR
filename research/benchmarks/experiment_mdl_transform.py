#!/usr/bin/env python3
"""Experiment: MDL Transform Selection — BIC chooses {identity, Yeo-Johnson}.

Principle:
  FLAIR already uses BIC for period selection. We extend BIC to variance
  stabilization: the transform T is selected by minimizing

      BIC(T) = n × log(σ²_T) − 2 × Σ log|J_T(yᵢ)| + k_T × log(n)

  identity: k=0, J=1       → BIC_id  = n × log(var(innovations))
  YJ(λ):   k=1, J=(y+1)^(λ−1) → BIC_yj = n × log(var(YJ_innov)) − 2(λ−1)Σlog(Lᵢ+1) + log(n)

  MDL naturally selects identity when Level is already homoscedastic
  (no transform, no shift needed) and YJ when variance stabilization
  provides enough improvement to justify the extra parameter λ.

  YJ(y,λ) = BC(y+1, λ) for y≥0 — the "+1" is built into the definition,
  providing natural zero-regularization without ad hoc P×shift.

Usage:
  uv run python -u research/benchmarks/experiment_mdl_transform.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import yeojohnson, yeojohnson_normmax

os.environ['GIFT_EVAL'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'gift-eval')

from gift_eval.data import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model import evaluate_model
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

from run_gift_eval_flar9 import (
    get_period, get_periods, FREQ_TO_PERIOD, FREQ_TO_PERIODS,
    _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg,
    _multi_fourier_periods, flar_v5, flar_v9,
    Predictor, METRICS, CONFIGS, _get_sn, run_variant,
)

warnings.filterwarnings('ignore')


# =========================================================================
# Yeo-Johnson helpers
# =========================================================================

def _yj(y, lam):
    """Yeo-Johnson forward transform (vectorized)."""
    out = np.zeros_like(y, dtype=float)
    pos = y >= 0
    neg = ~pos
    if lam != 0:
        out[pos] = ((y[pos] + 1.0) ** lam - 1.0) / lam
    else:
        out[pos] = np.log(y[pos] + 1.0)
    if (2.0 - lam) != 0:
        out[neg] = -((-y[neg] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)
    else:
        out[neg] = -np.log(-y[neg] + 1.0)
    return out


def _yj_inv(z, lam):
    """Yeo-Johnson inverse transform (vectorized)."""
    out = np.zeros_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    if lam != 0:
        out[pos] = (z[pos] * lam + 1.0) ** (1.0 / lam) - 1.0
    else:
        out[pos] = np.exp(z[pos]) - 1.0
    if (2.0 - lam) != 0:
        out[neg] = 1.0 - (-(2.0 - lam) * z[neg] + 1.0) ** (1.0 / (2.0 - lam))
    else:
        out[neg] = 1.0 - np.exp(-z[neg])
    return out


def _yj_lambda(y):
    """Estimate Yeo-Johnson lambda via MLE, clipped to [0, 2]."""
    if len(y) < 10:
        return 1.0  # identity
    try:
        lam = float(yeojohnson_normmax(y))
        return float(np.clip(lam, 0.0, 2.0))
    except Exception:
        return 1.0


def _select_transform_mdl(L_work, n):
    """MDL transform selection: identity vs Yeo-Johnson.

    BIC(T) = n × log(σ²_T) − 2 × Σ log|J_T| + k_T × log(n)

    Returns: (transform_name, transformed_L, lambda, inverse_fn)
    """
    # Compute innovations (first differences) for variance comparison
    innov_raw = np.diff(L_work)
    var_id = float(np.var(innov_raw))
    bic_id = n * np.log(max(var_id, 1e-30))

    # Yeo-Johnson
    lam_yj = _yj_lambda(L_work)
    L_yj = _yj(L_work, lam_yj)
    innov_yj = np.diff(L_yj)
    var_yj = float(np.var(innov_yj))

    # Jacobian for YJ: for y >= 0, J = (y+1)^(lam-1); for y < 0, J = (-y+1)^(1-lam)
    log_jac = np.zeros(len(L_work))
    pos = L_work >= 0
    log_jac[pos] = (lam_yj - 1.0) * np.log(L_work[pos] + 1.0)
    neg = ~pos
    if neg.any():
        log_jac[neg] = (1.0 - lam_yj) * np.log(-L_work[neg] + 1.0)
    sum_log_jac = float(np.sum(log_jac))

    bic_yj = n * np.log(max(var_yj, 1e-30)) - 2.0 * sum_log_jac + np.log(n)

    if bic_id <= bic_yj:
        return 'identity', L_work, 1.0
    else:
        return 'yj', L_yj, lam_yj


# =========================================================================
# Shape₂ (verbatim from run_gift_eval_flair_ds.py)
# =========================================================================

def _compute_shape2(L, cp, n_complete):
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
    t = np.arange(cp, dtype=float)
    cos_b = np.cos(2 * np.pi * t / cp)
    sin_b = np.sin(2 * np.pi * t / cp)
    S2_c = S2_raw - 1.0
    a = 2.0 * np.mean(S2_c * cos_b)
    b = 2.0 * np.mean(S2_c * sin_b)
    S2_harmonic = 1.0 + a * cos_b + b * sin_b
    RSS_flat = np.sum(S2_c ** 2)
    RSS_harmonic = np.sum((S2_raw - S2_harmonic) ** 2)
    bic_flat = cp * np.log(max(RSS_flat / cp, 1e-30))
    bic_harmonic = cp * np.log(max(RSS_harmonic / cp, 1e-30)) + 2 * np.log(cp)
    S2_prior = S2_harmonic if bic_harmonic < bic_flat else np.ones(cp)
    w = nc2 / (nc2 + cp)
    S2 = w * S2_raw + (1 - w) * S2_prior
    S2 = np.maximum(S2, 1e-6)
    S2 = S2 / S2.mean()
    return S2


# =========================================================================
# FLAIR-MDL: MDL Transform Selection
# =========================================================================

def flair_mdl_t(y_raw, horizon, period, freq_str, n_samples=500):
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)

    # ── Conditional location shift (only for negative series) ──
    y_floor = float(y.min())
    y_shift = (1.0 - y_floor) if y_floor < 0 else 0.0
    y = y + y_shift

    n = len(y)
    cal = get_periods(freq_str)
    candidates = cal if cal else [period]
    candidates = [p for p in candidates if p >= 1 and n // p >= 3]
    if not candidates:
        candidates = [1]

    # ── MDL/BIC period selection ──
    if len(candidates) == 1:
        P = candidates[0]
    else:
        T_max = min(n, 500 * min(candidates))
        y_sel = y[-T_max:]
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = T_max // p_cand
            if nc < 5:
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

    # ── Reshape ──
    mat = y_trim.reshape(n_complete, P).T

    # ── Shape ──
    K = min(5, n_complete)
    recent = mat[:, -K:]
    recent_totals = recent.sum(axis=0, keepdims=True)
    proportions = np.where(recent_totals > 1e-10,
                           recent / recent_totals,
                           1.0 / P)
    S_global = proportions.mean(axis=1)
    S_global = S_global / max(S_global.sum(), 1e-10)

    # ── Level ──
    L = mat.sum(axis=0)

    # ── Dirichlet Shape context ──
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
        ds_props = np.where(ds_totals > 1e-10,
                            ds_mat / ds_totals,
                            1.0 / P)

        ctx_shapes = np.zeros((C, P))
        for ci in range(C):
            mask = ds_ctx == ci
            if mask.sum() >= 1:
                ctx_shapes[ci] = ds_props[:, mask].mean(axis=1)
            else:
                ctx_shapes[ci] = S_global
            ctx_shapes[ci] /= max(ctx_shapes[ci].sum(), 1e-10)

        S_hist = np.zeros((n_complete, P))
        for k in range(n_complete):
            S_hist[k] = ctx_shapes[k % C]

        S_forecast = np.zeros((m, P))
        for step in range(m):
            S_forecast[step] = ctx_shapes[(n_complete + step) % C]

    # ── Cross-period lags ──
    cross_periods = []
    for sp in secondary:
        if sp % P == 0:
            cp = sp // P
            if 2 <= cp <= n_complete // 2:
                cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})

    max_cp = max(cross_periods) if cross_periods else 0

    # ── Shape₂ ──
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

    # ══════════════════════════════════════════════════════════════════════
    # MDL TRANSFORM SELECTION: identity vs Yeo-Johnson
    # ══════════════════════════════════════════════════════════════════════
    transform, L_bc_raw, lam = _select_transform_mdl(L_work, n_complete)

    # For identity: inverse is identity. For YJ: inverse is _yj_inv.
    def _inv(z):
        if transform == 'identity':
            return z
        return _yj_inv(z, lam)

    L_bc = L_bc_raw

    # ── NLinear on Level ──
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # ── Features: intercept + trend + lags ──
    start = max(1, max_cp) if max_cp >= 2 else 1
    if n_complete - start < 3 and max_cp >= 2:
        max_cp = 0
        start = 1

    n_train = n_complete - start
    if n_train < 2:
        L_hat_raw = np.full(m, _inv(last_L))
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

    # ── ONE Ridge SA ──
    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # ── Stochastic Level paths ──
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

    # ── Inverse transform ──
    L_hat_all = _inv(L_paths[:, n_complete:n_complete + m] + last_L)

    if use_deseason:
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # ── Phase noise (SVD Residual Quantiles) ──
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(50, n_complete)
    # Robust floor: L/P (uniform expectation per phase)
    L_floor = np.maximum(np.abs(L[-K_r:]), 1e-8)[np.newaxis, :] / max(P, 1)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), L_floor)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    R_flat = R.ravel()
    raw_idx = np.random.randint(0, K_r, size=(n_samples, horizon))
    phase_noise = R_flat[phase_idx[np.newaxis, :] * K_r + raw_idx]

    # ── Assemble ──
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

def run_variant_200(label, forecast_fn):
    """run_variant with n_samples=200."""
    results = []
    for i, (ds, freq, term) in enumerate(CONFIGS, 1):
        cid = f"{ds}/{freq}/{term}"
        name_map = {
            "kdd_cup_2018": "kdd_cup_2018_with_missing",
            "car_parts": "car_parts_with_missing",
            "temperature_rain": "temperature_rain_with_missing",
            "loop_seattle": "LOOP_SEATTLE",
            "m_dense": "M_DENSE", "sz_taxi": "SZ_TAXI", "saugeen": "saugeenday",
        }
        load = name_map.get(ds, ds)
        dp = os.path.join(os.environ['GIFT_EVAL'], load)
        if os.path.isdir(os.path.join(dp, freq)):
            load = f"{load}/{freq}"
        try:
            dataset = Dataset(name=load, term=term, to_univariate=False)
        except Exception:
            print(f"  [{i:>2}] {cid} SKIP")
            continue
        period_val = get_period(dataset.freq)
        pred = Predictor(dataset.prediction_length, period_val, dataset.freq,
                         200, forecast_fn)
        t0 = time.perf_counter()
        try:
            res = evaluate_model(pred, test_data=dataset.test_data, metrics=METRICS,
                                 batch_size=5000, axis=None, mask_invalid_label=True,
                                 allow_nan_forecast=False, seasonality=None)
            mase = float(res['MASE[0.5]'].iloc[0])
            crps = float(res['mean_weighted_sum_quantile_loss'].iloc[0])
        except Exception as e:
            print(f"  [{i:>2}] {cid} ERROR: {e}")
            continue
        elapsed = time.perf_counter() - t0
        print(f"  [{i:>2}] {label:4s} {cid:40s} MASE={mase:.4f}  CRPS={crps:.4f}  ({elapsed:.0f}s)")
        results.append({'config': cid, 'mase': mase, 'crps': crps,
                        'time': elapsed, 'term': term})
    return results


if __name__ == '__main__':
    from run_gift_eval_flair_ds import flair_ds

    total_start = time.perf_counter()
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    print(f"\n{'='*70}")
    print(f"  Variant: FLAIR-DS (baseline, n_samples=200)")
    print(f"{'='*70}")
    ds_results = run_variant_200('DS', flair_ds)

    print(f"\n{'='*70}")
    print(f"  Variant: FLAIR-MDL-T (MDL Transform Selection, n_samples=200)")
    print(f"{'='*70}")
    mdl_results = run_variant_200('MDLT', flair_mdl_t)

    # ── Comparison ──
    dsm = {r['config']: r for r in ds_results}
    print(f"\n{'='*70}")
    print("DELTA MDL-T vs DS (n_samples=200)")
    print(f"{'='*70}")
    ratios_m, ratios_c = [], []
    for r in mdl_results:
        ds = dsm.get(r['config'])
        if ds:
            rm = r['mase'] / ds['mase']
            rc = r['crps'] / ds['crps']
            ratios_m.append(rm)
            ratios_c.append(rc)
            dp = (rm - 1) * 100
            mk = '++' if dp < -3 else '+' if dp < 0 else '--' if dp > 3 else '-' if dp > 0 else '='
            print(f"  {mk:2s} {r['config']:40s}  MDLT={r['mase']:.4f}  DS={ds['mase']:.4f}  ({dp:+.1f}%)")

    if ratios_m:
        gm_m = gm(ratios_m)
        gm_c = gm(ratios_c)
        wins = sum(1 for r in ratios_m if r < 1.0)
        losses = sum(1 for r in ratios_m if r > 1.0)
        print(f"\nGeo mean MASE ratio (MDLT/DS): {gm_m:.4f}  ({(gm_m-1)*100:+.1f}%)")
        print(f"Geo mean CRPS ratio (MDLT/DS): {gm_c:.4f}  ({(gm_c-1)*100:+.1f}%)")
        print(f"Wins: {wins}/{len(ratios_m)}, Losses: {losses}/{len(ratios_m)}")

    elapsed = time.perf_counter() - total_start
    print(f"\nTotal time: {elapsed:.0f}s")
