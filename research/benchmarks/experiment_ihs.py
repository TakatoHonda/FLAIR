#!/usr/bin/env python3
"""Experiment: Conditional location shift + Level-only shift for Box-Cox.

Hypothesis:
  - Unconditional y_shift inflates Level by P × y_shift → systematic bias
  - Fix: only shift raw y for negative series (Shape needs non-negative)
  - For Box-Cox: shift Level separately (1× per cycle, not P×)

Changes vs flair_ds():
  1. y_shift: unconditional max(1-y_floor, 1.0) → only when y_floor < 0
  2. L_level_shift: new, ensures L_work > 0 for Box-Cox
  3. Subtract L_level_shift after inverse Box-Cox

Usage:
  uv run python -u research/benchmarks/experiment_ihs.py
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

from run_gift_eval_flar9 import (
    get_period, get_periods, FREQ_TO_PERIOD, FREQ_TO_PERIODS,
    _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg,
    _multi_fourier_periods, flar_v5, flar_v9,
    Predictor, METRICS, CONFIGS, _get_sn, run_variant,
)

warnings.filterwarnings('ignore')


# =========================================================================
# Shape₂ (copied verbatim from run_gift_eval_flair_ds.py)
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
# FLAIR-LS: Level-only Shift (conditional y_shift + L_level_shift)
# =========================================================================

def flair_ls(y_raw, horizon, period, freq_str, n_samples=500):
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)

    # ── CHANGE 1: Conditional location shift ──
    # Only shift when raw data has negative values (Shape requires non-negative)
    # Non-negative series: y_shift = 0 → no Level inflation
    y_floor = float(y.min())
    if y_floor < 0:
        y_shift = 1.0 - y_floor   # negative series: shift to min=1
    else:
        y_shift = 0.0             # non-negative series: no shift
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
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

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

    # ── Shape₂: sinusoidal prior shrinkage ──
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

    # ── CHANGE 2: Level-only shift for Box-Cox ──
    # Floor at P (same stability as DS's y_shift=1 which gives L_floor=P)
    # but without corrupting Shape via P × y_shift bias
    L_level_shift = max(float(P) - float(L_work.min()), 0.0)
    L_work = L_work + L_level_shift

    # ── Box-Cox on (deseasonalized, shifted) Level ──
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)

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
        # ── CHANGE 3a: Subtract L_level_shift after inverse Box-Cox ──
        L_hat_raw = np.full(m, _bc_inv(last_L, lam) - L_level_shift)
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

    # ── Stochastic Level paths (recursive noise injection) ──
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

    # ── CHANGE 3b: Subtract L_level_shift after inverse Box-Cox ──
    L_hat_all = _bc_inv(L_paths[:, n_complete:n_complete + m] + last_L, lam) - L_level_shift

    if use_deseason:
        forecast_pos = (n_complete + np.arange(m)) % cp_main
        L_hat_all = L_hat_all * S2[forecast_pos][np.newaxis, :]

    # ── Phase noise (SVD Residual Quantiles) ──
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(50, n_complete)
    # CHANGE 4: Robust denominator — use L/P (uniform expectation) as floor
    # Prevents blow-up when Shape has zeros (e.g., solar at night)
    L_floor = np.maximum(np.abs(L[-K_r:]), 1e-8)[np.newaxis, :] / max(P, 1)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), L_floor)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    R_flat = R.ravel()
    raw_idx = np.random.randint(0, K_r, size=(n_samples, horizon))
    phase_noise = R_flat[phase_idx[np.newaxis, :] * K_r + raw_idx]

    # ── Assemble: Level_path × Shape × (1 + phase_noise) ──
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

# DS baseline
DS_RESULTS = [
    {'config': 'bitbrains_fast_storage/5T/short', 'mase': 1.1400, 'crps': 0.5318, 'time': 193, 'term': 'short'},
    {'config': 'bitbrains_rnd/H/short', 'mase': 6.7140, 'crps': 0.7763, 'time': 2, 'term': 'short'},
    {'config': 'bizitobs_l2c/5T/short', 'mase': 0.4071, 'crps': 0.1192, 'time': 1, 'term': 'short'},
    {'config': 'electricity/H/short', 'mase': 1.0991, 'crps': 0.0842, 'time': 74, 'term': 'short'},
    {'config': 'solar/H/short', 'mase': 0.9059, 'crps': 0.4035, 'time': 14, 'term': 'short'},
    {'config': 'car_parts/M/short', 'mase': 1.1891, 'crps': 1.3504, 'time': 2, 'term': 'short'},
    {'config': 'restaurant/D/short', 'mase': 0.7524, 'crps': 0.2906, 'time': 1, 'term': 'short'},
    {'config': 'hospital/M/short', 'mase': 0.8853, 'crps': 0.0661, 'time': 0, 'term': 'short'},
    {'config': 'saugeen/D/short', 'mase': 3.3611, 'crps': 0.4495, 'time': 0, 'term': 'short'},
    {'config': 'm4_yearly/A/short', 'mase': 3.6133, 'crps': 0.1502, 'time': 19, 'term': 'short'},
    {'config': 'm4_monthly/M/short', 'mase': 1.2125, 'crps': 0.1224, 'time': 42, 'term': 'short'},
    {'config': 'm4_hourly/H/short', 'mase': 1.1763, 'crps': 0.0395, 'time': 1, 'term': 'short'},
    {'config': 'sz_taxi/15T/short', 'mase': 0.5980, 'crps': 0.2256, 'time': 3, 'term': 'short'},
    {'config': 'electricity/H/medium', 'mase': 1.2374, 'crps': 0.0944, 'time': 30, 'term': 'medium'},
    {'config': 'electricity/H/long', 'mase': 1.3707, 'crps': 0.1110, 'time': 19, 'term': 'long'},
    {'config': 'solar/H/medium', 'mase': 0.9886, 'crps': 0.4397, 'time': 2, 'term': 'medium'},
    {'config': 'loop_seattle/H/medium', 'mase': 1.1920, 'crps': 0.0880, 'time': 4, 'term': 'medium'},
    {'config': 'bizitobs_l2c/5T/medium', 'mase': 0.7993, 'crps': 0.4101, 'time': 0, 'term': 'medium'},
    {'config': 'sz_taxi/15T/medium', 'mase': 0.5883, 'crps': 0.2286, 'time': 0, 'term': 'medium'},
]


def run_variant_200(label, forecast_fn):
    """Same as run_variant but with n_samples=200 for stable comparison."""
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
                         200, forecast_fn)  # n_samples=200
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

    # Run BOTH variants fresh with n_samples=200
    print(f"\n{'='*70}")
    print(f"  Variant: FLAIR-DS (baseline, n_samples=200)")
    print(f"{'='*70}")
    ds_fresh = run_variant_200('DS', flair_ds)

    print(f"\n{'='*70}")
    print(f"  Variant: FLAIR-LS (Level-only Shift, n_samples=200)")
    print(f"{'='*70}")
    ls_results = run_variant_200('LS', flair_ls)

    # ── Per-config comparison ──
    dsm = {r['config']: r for r in ds_fresh}
    print(f"\n{'='*70}")
    print("DELTA LS vs DS (n_samples=200, per config)")
    print(f"{'='*70}")
    ratios_m, ratios_c = [], []
    for r in ls_results:
        ds = dsm.get(r['config'])
        if ds:
            rm = r['mase'] / ds['mase']
            rc = r['crps'] / ds['crps']
            ratios_m.append(rm)
            ratios_c.append(rc)
            dp = (rm - 1) * 100
            mk = '++' if dp < -3 else '+' if dp < 0 else '--' if dp > 3 else '-' if dp > 0 else '='
            print(f"  {mk:2s} {r['config']:40s}  LS={r['mase']:.4f}  DS={ds['mase']:.4f}  ({dp:+.1f}%)")

    if ratios_m:
        gm_m = gm(ratios_m)
        gm_c = gm(ratios_c)
        wins = sum(1 for r in ratios_m if r < 1.0)
        losses = sum(1 for r in ratios_m if r > 1.0)
        print(f"\nGeo mean MASE ratio (LS/DS): {gm_m:.4f}  ({(gm_m-1)*100:+.1f}%)")
        print(f"Geo mean CRPS ratio (LS/DS): {gm_c:.4f}  ({(gm_c-1)*100:+.1f}%)")
        print(f"Wins: {wins}/{len(ratios_m)}, Losses: {losses}/{len(ratios_m)}")

    elapsed = time.perf_counter() - total_start
    print(f"\nTotal time: {elapsed:.0f}s")
