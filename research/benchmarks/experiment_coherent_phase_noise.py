#!/usr/bin/env python3
"""Experiment: Scenario-Coherent Phase Noise Sampling.

Current: each (sample, horizon) independently picks a random column from R.
Proposed: all phases within the same forecast step share the same historical
scenario (column), preserving cross-phase correlation structure.

Both variants run fresh with n_samples=200 for fair comparison.

Usage:
    python -u research/benchmarks/experiment_coherent_phase_noise.py
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

METRICS = [
    MASE(forecast_type=0.5),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]


# =========================================================================
# Shared FLAIR-DS core — only phase noise sampling differs
# =========================================================================

def _flair_ds_core(y_raw, horizon, period, freq_str, n_samples, coherent=False):
    """FLAIR-DS with configurable phase noise sampling.

    coherent=False: independent sampling (current production)
    coherent=True:  scenario-coherent sampling (proposed)
    """
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

    # MDL/BIC period selection
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

    # Shape
    K = min(5, n_complete)
    recent = mat[:, -K:]
    recent_totals = recent.sum(axis=0, keepdims=True)
    proportions = np.where(recent_totals > 1e-10, recent / recent_totals, 1.0 / P)
    S_global = proportions.mean(axis=1)
    S_global = S_global / max(S_global.sum(), 1e-10)

    # Level
    L = mat.sum(axis=0)

    # Dirichlet Shape context
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

    # Cross-periods
    cross_periods = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})
    max_cp = max(cross_periods) if cross_periods else 0

    # Shape2
    cp_main = cross_periods[0] if cross_periods else 0
    S2 = None; use_deseason = False
    if cp_main >= 2:
        S2 = _compute_shape2(L, cp_main, n_complete)
        if S2 is not None:
            use_deseason = True

    if use_deseason:
        pos = np.arange(n_complete) % cp_main
        L_work = L / np.maximum(S2[pos], 1e-10)
    else:
        L_work = L

    # Box-Cox + NLinear
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # Ridge features
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

    # Phase noise — the only difference between variants
    fitted_mat = S_hist.T * L
    E = mat - fitted_mat
    K_r = min(50, n_complete)
    R = E[:, -K_r:] / np.maximum(np.abs(fitted_mat[:, -K_r:]), 1e-8)

    step_idx = np.arange(horizon) // P
    phase_idx = np.arange(horizon) % P

    if coherent:
        # ── Scenario-coherent: sample entire columns (periods) ────────
        # All phases within the same forecast step share the same
        # historical scenario, preserving cross-phase correlation.
        col_idx = np.random.randint(0, K_r, size=(n_samples, m))
        phase_noise = R[phase_idx[np.newaxis, :], col_idx[:, step_idx]]
    else:
        # ── Independent: current production (each cell sampled independently)
        R_flat = R.ravel()
        raw_idx = np.random.randint(0, K_r, size=(n_samples, horizon))
        phase_noise = R_flat[phase_idx[np.newaxis, :] * K_r + raw_idx]

    # Assemble
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
# Two variants
# =========================================================================

def flair_independent(y_raw, horizon, period, freq_str, n_samples=N_SAMPLES):
    return _flair_ds_core(y_raw, horizon, period, freq_str, n_samples, coherent=False)

def flair_coherent(y_raw, horizon, period, freq_str, n_samples=N_SAMPLES):
    return _flair_ds_core(y_raw, horizon, period, freq_str, n_samples, coherent=True)


# =========================================================================
# Runner (n_samples=200 for fair comparison)
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

    print(f"Experiment: Scenario-Coherent Phase Noise (n_samples={N_SAMPLES})")
    print(f"{'='*70}")

    all_results = {}
    for label, fn in [('INDEP', flair_independent), ('COHERE', flair_coherent)]:
        print(f"\n{'='*70}")
        print(f"  Variant: {label}")
        print(f"{'='*70}")
        all_results[label] = run_experiment(label, fn)

    # Summary by horizon
    for tf, tl in [('short', 'SHORT'), ('medium', 'MEDIUM'), ('long', 'LONG')]:
        print(f"\n{'='*70}")
        print(f"SUMMARY — {tl} horizon")
        print(f"{'='*70}")
        for label, res in all_results.items():
            f = [r for r in res if r['term'] == tf and r['sn_mase'] is not None]
            if f:
                rm = [r['mase'] / r['sn_mase'] for r in f]
                rc = [r['crps'] / r['sn_crps'] for r in f]
                print(f"  {label:8s}: relMASE={gm(rm):.4f}  relCRPS={gm(rc):.4f}  "
                      f"({sum(r['time'] for r in f):.0f}s, {len(f)} configs)")

    # Overall
    print(f"\n{'='*70}")
    print("OVERALL")
    print(f"{'='*70}")
    for label, res in all_results.items():
        v = [r for r in res if r['sn_mase'] is not None]
        if v:
            rm = [r['mase'] / r['sn_mase'] for r in v]
            rc = [r['crps'] / r['sn_crps'] for r in v]
            print(f"  {label:8s}: relMASE={gm(rm):.4f}  relCRPS={gm(rc):.4f}  "
                  f"({sum(r['time'] for r in v):.0f}s)")

    # Per-config delta
    print(f"\n{'='*70}")
    print("DELTA: COHERENT vs INDEPENDENT (negative = coherent is better)")
    print(f"{'='*70}")
    ind_map = {r['config']: r for r in all_results.get('INDEP', [])}
    for r in all_results.get('COHERE', []):
        ind = ind_map.get(r['config'])
        if ind and ind['sn_crps'] and r['sn_crps']:
            d_mase = r['mase'] / r['sn_mase'] - ind['mase'] / ind['sn_mase']
            d_crps = r['crps'] / r['sn_crps'] - ind['crps'] / ind['sn_crps']
            m_sign = "<<" if d_mase < -0.005 else (">>" if d_mase > 0.005 else "==")
            c_sign = "<<" if d_crps < -0.005 else (">>" if d_crps > 0.005 else "==")
            print(f"  {r['config']:40s} [{r['term']:6s}]  "
                  f"dMASE={d_mase:+.4f} {m_sign}  dCRPS={d_crps:+.4f} {c_sign}")

    print(f"\nWall time: {time.perf_counter()-total_start:.0f}s")
