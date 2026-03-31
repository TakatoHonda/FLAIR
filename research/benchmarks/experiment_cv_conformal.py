#!/usr/bin/env python3
"""Experiment: Temporal CV Conformal vs current FLAIR-DS.

Current: LOO residuals → bootstrap Level paths + phase noise (multiplicative)
Proposed: Expanding-window CV → OOS residuals per horizon → direct conformal quantiles

Both variants run fresh with n_samples=200 for fair comparison.
CV uses n_samples=10 per split for fast median point forecasts.

Usage:
    python -u research/benchmarks/experiment_cv_conformal.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

os.environ['GIFT_EVAL'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'gift-eval')

from gift_eval.data import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model import evaluate_model
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

from run_gift_eval_flar9 import (
    get_period, get_periods, FREQ_TO_PERIOD, FREQ_TO_PERIODS,
    _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg,
    _get_sn, CONFIGS,
)
from run_gift_eval_flair_ds import _compute_shape2

warnings.filterwarnings('ignore')

N_SAMPLES = 200
CV_SAMPLES = 10        # small n_samples for fast CV point forecasts
MIN_CV_SPLITS = 3      # minimum CV splits before falling back

METRICS = [
    MASE(forecast_type=0.5),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


# =========================================================================
# FLAIR-DS core (current production logic)
# =========================================================================

def _flair_ds(y_raw, horizon, period, freq_str, n_samples):
    """Current FLAIR-DS with bootstrap Level paths + phase noise."""
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    y_floor = y.min()
    y_shift = max(1 - y_floor, 1.0)
    y = y + y_shift
    n = len(y)

    cal = get_periods(freq_str)
    candidates = cal if cal else [period]
    candidates = [p for p in candidates if p >= 1 and n // p >= 3]
    if not candidates:
        candidates = [1]

    if len(candidates) == 1:
        P = candidates[0]
    else:
        T_max = min(n, 500 * min(candidates))
        y_sel = y[-T_max:]
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = T_max // p_cand
            if nc < 5: continue
            mat_c = y_sel[-(nc * p_cand):].reshape(nc, p_cand).T
            s = np.linalg.svd(mat_c, compute_uv=False)
            rss1 = np.sum(s[1:] ** 2)
            T = nc * p_cand
            bic = T * np.log(max(rss1 / T, 1e-30)) + (p_cand + nc - 1) * np.log(T)
            if bic < best_bic:
                best_P, best_bic = p_cand, bic
        P = best_P

    secondary = [p for p in cal[1:] if p != P] if len(cal) > 1 else []

    n_complete = n // P
    if n_complete < 3:
        P = 1; secondary = []; n_complete = n
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
    mat = y_trim.reshape(n_complete, P).T

    K = min(5, n_complete)
    recent = mat[:, -K:]
    recent_totals = recent.sum(axis=0, keepdims=True)
    proportions = np.where(recent_totals > 1e-10, recent / recent_totals, 1.0 / P)
    S_global = proportions.mean(axis=1)
    S_global = S_global / max(S_global.sum(), 1e-10)

    L = mat.sum(axis=0)

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
            if mask.sum() == 0: S_ctx[c_val] = S_global
            else:
                S_c = (kappa * S_global + ds_mat[:, mask].sum(axis=1)) / \
                      max(kappa + ds_L[mask].sum(), 1e-10)
                S_ctx[c_val] = S_c / max(S_c.sum(), 1e-10)
        S_forecast = S_ctx[(n_complete + np.arange(m)) % C]
        S_hist = S_ctx[np.arange(n_complete) % C]

    cross_periods = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})
    max_cp = max(cross_periods) if cross_periods else 0

    cp_main = cross_periods[0] if cross_periods else 0
    S2 = None; use_deseason = False
    if cp_main >= 2:
        S2 = _compute_shape2(L, cp_main, n_complete)
        if S2 is not None: use_deseason = True

    if use_deseason:
        pos = np.arange(n_complete) % cp_main
        L_work = L / np.maximum(S2[pos], 1e-10)
    else:
        L_work = L

    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    start = max(1, max_cp) if max_cp >= 2 else 1
    if n_complete - start < 3 and max_cp >= 2:
        max_cp = 0; start = 1
    n_train = n_complete - start
    if n_train < 2:
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

    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # Stochastic Level paths
    noise_pool = loo_resid[np.random.randint(0, len(loo_resid), size=(n_samples, m))]
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

    # Phase noise
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(50, n_complete)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), 1e-8)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    R_flat = R.ravel()
    raw_idx = np.random.randint(0, K_r, size=(n_samples, horizon))
    phase_noise = R_flat[phase_idx[np.newaxis, :] * K_r + raw_idx]

    S_h = S_forecast[step_idx, phase_idx]
    samples = (L_hat_all[:, step_idx]
               * S_h[np.newaxis, :]
               * (1 + phase_noise)
               - y_shift)

    y_orig = y - y_shift
    y_lo, y_hi = y_orig[-max(horizon*2, 50):].min(), y_orig[-max(horizon*2, 50):].max()
    y_range = max(y_hi - y_lo, 1e-6)
    samples = np.clip(samples, y_lo - y_range, y_hi + y_range)
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================================
# CV Conformal variant
# =========================================================================

def _flair_cv_conformal(y_raw, horizon, period, freq_str, n_samples):
    """FLAIR with Temporal CV Conformal quantile calibration.

    1. Expanding-window CV → OOS residuals per horizon position
    2. Full-series FLAIR → point forecast (median)
    3. Samples = point_forecast + resampled OOS residuals (per h)
    """
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    n = len(y)

    # Determine minimum training length
    P_guess = max(period, 1)
    min_train = max(3 * P_guess, 20)

    # --- Expanding-window CV ---
    residuals_by_h = [[] for _ in range(horizon)]
    step = max(P_guess, 1)
    n_splits = 0

    for t in range(n - horizon, min_train - 1, -step):
        # Get point forecast from FLAIR on y[:t]
        try:
            cv_samples = _flair_ds(y_raw[:t], horizon, period, freq_str, CV_SAMPLES)
            fc = np.median(cv_samples, axis=0)
        except Exception:
            continue

        actual = np.asarray(y_raw[t:t + horizon], float)
        for h in range(min(horizon, len(actual))):
            r = actual[h] - fc[h]
            if np.isfinite(r):
                residuals_by_h[h].append(r)
        n_splits += 1

        # Cap at 20 splits for speed
        if n_splits >= 20:
            break

    # Check if we have enough residuals
    min_resids = min(len(r) for r in residuals_by_h) if residuals_by_h[0] else 0

    if min_resids < MIN_CV_SPLITS:
        # Fallback to current approach
        return _flair_ds(y_raw, horizon, period, freq_str, n_samples)

    # --- Full-series point forecast ---
    full_samples = _flair_ds(y_raw, horizon, period, freq_str, CV_SAMPLES)
    point_fc = np.median(full_samples, axis=0)

    # --- Generate samples from OOS residuals ---
    samples = np.zeros((n_samples, horizon))
    for h in range(horizon):
        resids = np.array(residuals_by_h[h])
        drawn = resids[np.random.randint(0, len(resids), size=n_samples)]
        samples[:, h] = point_fc[h] + drawn

    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


# =========================================================================
# Wrappers
# =========================================================================

def flair_baseline(y_raw, horizon, period, freq_str, n_samples=N_SAMPLES):
    return _flair_ds(y_raw, horizon, period, freq_str, n_samples)

def flair_cv(y_raw, horizon, period, freq_str, n_samples=N_SAMPLES):
    return _flair_cv_conformal(y_raw, horizon, period, freq_str, n_samples)


# =========================================================================
# Runner
# =========================================================================

class Predictor(RepresentablePredictor):
    def __init__(self, prediction_length, period, freq_str, n_samples, forecast_fn):
        super().__init__(prediction_length=prediction_length)
        self.period = period; self.freq_str = freq_str
        self.n_samples = n_samples; self.forecast_fn = forecast_fn
    def predict_item(self, item):
        target = item['target']
        if target.ndim == 2:
            nv, T = target.shape
            samples = np.zeros((self.n_samples, self.prediction_length, nv))
            for v in range(nv):
                samples[:,:,v] = self.forecast_fn(
                    target[v], self.prediction_length, self.period,
                    self.freq_str, self.n_samples)
            start = item['start'] + T
        else:
            samples = self.forecast_fn(
                target, self.prediction_length, self.period,
                self.freq_str, self.n_samples)
            start = item['start'] + len(target)
        return SampleForecast(samples=samples, start_date=start,
                              item_id=item.get('item_id', 'unknown'))


def run_experiment(label, forecast_fn):
    results = []
    name_map = {
        "kdd_cup_2018": "kdd_cup_2018_with_missing",
        "car_parts": "car_parts_with_missing",
        "temperature_rain": "temperature_rain_with_missing",
        "loop_seattle": "LOOP_SEATTLE",
        "m_dense": "M_DENSE", "sz_taxi": "SZ_TAXI", "saugeen": "saugeenday",
    }
    for i, (ds, freq, term) in enumerate(CONFIGS, 1):
        cid = f"{ds}/{freq}/{term}"
        load = name_map.get(ds, ds)
        dp = os.path.join(os.environ['GIFT_EVAL'], load)
        if os.path.isdir(os.path.join(dp, freq)):
            load = f"{load}/{freq}"
        try:
            dataset = Dataset(name=load, term=term, to_univariate=False)
        except Exception:
            print(f"  [{i:>2}] {cid} SKIP"); continue
        period_val = get_period(dataset.freq)
        pred = Predictor(dataset.prediction_length, period_val, dataset.freq,
                         N_SAMPLES, forecast_fn)
        t0 = time.perf_counter()
        try:
            res = evaluate_model(pred, test_data=dataset.test_data, metrics=METRICS,
                                 batch_size=5000, axis=None, mask_invalid_label=True,
                                 allow_nan_forecast=False, seasonality=None)
            mase = float(res['MASE[0.5]'].iloc[0])
            crps = float(res['mean_weighted_sum_quantile_loss'].iloc[0])
        except Exception as e:
            print(f"  [{i:>2}] {cid} ERROR: {e}"); continue
        elapsed = time.perf_counter() - t0
        sn_m, sn_c = _get_sn(cid)
        print(f"  [{i:>2}] {label:8s} {cid:40s} MASE={mase:.4f}  CRPS={crps:.4f}  ({elapsed:.0f}s)")
        results.append({'config': cid, 'mase': mase, 'crps': crps,
                        'sn_mase': sn_m, 'sn_crps': sn_c, 'time': elapsed,
                        'term': term})
    return results


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    total_start = time.perf_counter()
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    print(f"Experiment: CV Conformal vs Current FLAIR-DS (n_samples={N_SAMPLES})")
    print(f"  CV splits: up to 20, step=P, min {MIN_CV_SPLITS} residuals/h")
    print(f"  CV n_samples={CV_SAMPLES} for point forecast median")
    print(f"{'='*70}")

    all_results = {}
    for label, fn in [('BASELINE', flair_baseline), ('CV_CONF', flair_cv)]:
        print(f"\n{'='*70}")
        print(f"  Variant: {label}")
        print(f"{'='*70}")
        all_results[label] = run_experiment(label, fn)

    # Per-config delta
    print(f"\n{'='*70}")
    print("PER-CONFIG COMPARISON (CRPS: negative delta = CV_CONF better)")
    print(f"{'='*70}")
    base_map = {r['config']: r for r in all_results.get('BASELINE', [])}
    wins_crps, losses_crps = 0, 0
    for r in all_results.get('CV_CONF', []):
        b = base_map.get(r['config'])
        if b:
            d_mase = r['mase'] - b['mase']
            d_crps = r['crps'] - b['crps']
            c_sign = "<<" if d_crps < -0.002 else (">>" if d_crps > 0.002 else "==")
            if d_crps < -0.002: wins_crps += 1
            elif d_crps > 0.002: losses_crps += 1
            print(f"  {r['config']:40s} [{r['term']:6s}]  "
                  f"dMASE={d_mase:+.4f}  dCRPS={d_crps:+.4f} {c_sign}  "
                  f"({r['time']:.0f}s vs {b['time']:.0f}s)")
    print(f"\n  CRPS wins/losses: {wins_crps}/{losses_crps}")

    # Overall geometric mean (using raw values since SN may not be available)
    print(f"\n{'='*70}")
    print("OVERALL (geometric mean of raw MASE/CRPS)")
    print(f"{'='*70}")
    for label, res in all_results.items():
        if res:
            mases = [r['mase'] for r in res]
            crpss = [r['crps'] for r in res]
            print(f"  {label:8s}: gm_MASE={gm(mases):.4f}  gm_CRPS={gm(crpss):.4f}  "
                  f"(total {sum(r['time'] for r in res):.0f}s)")

    print(f"\nWall time: {time.perf_counter()-total_start:.0f}s")
