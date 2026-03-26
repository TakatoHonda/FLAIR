#!/usr/bin/env python3
"""FLAR v9: Level x Shape Cross-Period Ridge.

The most elegant architecture: separate WHAT (level) from HOW (shape).

  Level L(i) = total per period i → Ridge SA forecasts this (1 series)
  Shape S(j) = proportion for phase j → estimated from recent data (structural)
  Y_hat(k,j) = L_hat(k) × S(j) — multiplicative reconstruction in original space

Key advantages over V7:
  - ONE series to predict (not P stacked) → 25x more training data per model
  - Shape captures within-period dynamics structurally (not learned)
  - Multiplicative in original space → Box-Cox applies only to Level → no incoherence
  - m=ceil(H/P) recursive steps on Level (same long-horizon stability as V7)

Usage:
  python -u research/benchmarks/run_gift_eval_flar9.py
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
from scipy.stats import boxcox as scipy_boxcox

warnings.filterwarnings('ignore')

# =========================================================================
# Period tables
# =========================================================================

FREQ_TO_PERIOD = {
    'S': 60, 'T': 60, '5T': 12, '10T': 6, '15T': 4, '10S': 6,
    'H': 24, 'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'A': 1, 'Y': 1,
}
FREQ_TO_PERIODS = {
    '10S': [6, 360], 'S': [60], '5T': [12, 288], '10T': [6, 144],
    '15T': [4, 96], 'H': [24, 168], 'D': [7, 365], 'W': [52],
    'M': [12], 'Q': [4], 'A': [], 'Y': [],
}

def get_period(f):
    f = f.upper().replace('MIN', 'T')
    if f in FREQ_TO_PERIOD: return FREQ_TO_PERIOD[f]
    for k in sorted(FREQ_TO_PERIOD, key=len, reverse=True):
        if f.endswith(k): return FREQ_TO_PERIOD[k]
    return 1

def get_periods(f):
    f = f.upper().replace('MIN', 'T')
    if f in FREQ_TO_PERIODS: return list(FREQ_TO_PERIODS[f])
    for k in sorted(FREQ_TO_PERIODS, key=len, reverse=True):
        if f.endswith(k): return list(FREQ_TO_PERIODS[k])
    return []

# =========================================================================
# Box-Cox (applied to Level series only)
# =========================================================================

def _bc_lambda(y):
    yp = y[y > 0]
    if len(yp) < 10: return 1.0
    try: _, l = scipy_boxcox(yp); return float(np.clip(l, 0.0, 1.0))
    except: return 1.0

def _bc(y, l):
    y = np.maximum(y, 1e-8)
    return np.log(y) if l == 0.0 else (y**l - 1)/l

def _bc_inv(z, l):
    if l == 0.0: return np.exp(np.clip(z, -30, 30))
    return np.maximum(z*l + 1, 1e-10)**(1/l)

# =========================================================================
# Ridge with soft-average GCV
# =========================================================================

def _ridge_gcv_loo_softavg(X, y):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s2 = s**2; Uty = U.T @ y
    alphas = [10**la for la in np.linspace(-4, 4, 25)]
    gcv_scores = []
    for a in alphas:
        d = s2 / (s2 + a)
        H = (U**2) @ d
        r = y - U @ (d * Uty)
        gcv_scores.append(np.mean((r / np.maximum(1 - H, 1e-10))**2))
    gcv_arr = np.array(gcv_scores)
    gcv_min = gcv_arr.min()
    log_w = -(gcv_arr - gcv_min) / max(gcv_min, 1e-10)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()
    beta_avg = np.zeros(X.shape[1])
    d_avg = np.zeros(len(s))
    for wi, a in zip(w, alphas):
        if wi < 1e-15: continue
        d = s2 / (s2 + a)
        beta_avg += wi * (Vt.T @ (d * Uty / s))
        d_avg += wi * d
    r = y - X @ beta_avg
    H_avg = (U**2) @ d_avg
    loo = r / np.maximum(1 - H_avg, 1e-10)
    return beta_avg, loo, gcv_min

# =========================================================================
# V5 baseline (for fallback and comparison)
# =========================================================================

def _multi_fourier_periods(calendar_periods, T):
    c = set()
    for p in calendar_periods:
        if p < 2: continue
        if p <= T // 2: c.add(p)
        if p * 2 <= T // 2: c.add(p * 2)
        if p // 2 >= 2: c.add(p // 2)
    return sorted(c)

def flar_v5(y_raw, horizon, period, freq_str, n_samples=20):
    y = np.maximum(np.nan_to_num(np.asarray(y_raw, float), nan=0.0), 0.0)
    n = len(y)
    max_ctx = max(period*10, 500) if period >= 2 else 500
    if n > max_ctx: y = y[-max_ctx:]; n = len(y)
    lam = _bc_lambda(y)
    y_t = _bc(y+1, lam)
    cal = get_periods(freq_str)
    periods = _multi_fourier_periods(cal, n)
    start = max(1, period) if period >= 2 else 1
    if n <= start + horizon:
        fc = np.full(horizon, y[-1])
        s = max(np.std(np.diff(y[-min(50,n):])), 1e-6) if n>1 else 1.0
        return np.maximum(0, np.array([fc+np.random.normal(0,s,horizon) for _ in range(n_samples)]))
    t = np.arange(n, dtype=float); trend = t / float(n)
    cols = [np.ones(n), trend]
    for p in periods:
        cols.append(np.cos(2*np.pi*t/p)); cols.append(np.sin(2*np.pi*t/p))
    nb = len(cols); base = np.column_stack(cols)
    nl = 1 + (1 if period >= 2 else 0); nf = nb + nl
    X = np.zeros((n-start, nf)); X[:, :nb] = base[start:]
    X[:, nb] = y_t[start-1:-1]
    if period >= 2: X[:, nb+1] = y_t[start-period:n-period]
    yt = y_t[start:]
    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, yt)
    y_ext = np.concatenate([y_t, np.zeros(horizon)]); fc = np.zeros(horizon)
    for h in range(horizon):
        ti = n+h; x = np.zeros(nf); x[0]=1.0; x[1]=ti/float(n); col=2
        for p in periods:
            x[col]=np.cos(2*np.pi*ti/p); x[col+1]=np.sin(2*np.pi*ti/p); col+=2
        x[nb]=y_ext[ti-1]
        if period>=2: x[nb+1]=y_ext[ti-period]
        pred=x@beta; fc[h]=pred; y_ext[ti]=pred
    point_fc = np.maximum(_bc_inv(fc, lam)-1, 0.0)
    fitted_t = yt - loo_resid
    lo = _bc_inv(yt, lam) - _bc_inv(fitted_t, lam)
    lo = lo[np.isfinite(lo)]
    if len(lo)<3: lo = np.array([-1,0,1])*max(np.mean(point_fc)*0.05, 1e-6)
    rec = lo[-min(200,len(lo)):]
    drawn = np.random.choice(rec, size=(n_samples, horizon), replace=True)
    jitter = np.random.normal(0, np.std(rec)*0.1, size=(n_samples, horizon))
    samples = np.maximum(0, point_fc[np.newaxis,:] + drawn + jitter)
    rm = np.max(y[-max(horizon*2,50):])
    if rm>0: samples = np.clip(samples, 0, rm*3)
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

# =========================================================================
# V9: Level x Shape Cross-Period Ridge
# =========================================================================

def flar_v9(y_raw, horizon, period, freq_str, n_samples=20):
    y = np.maximum(np.nan_to_num(np.asarray(y_raw, float), nan=0.0), 0.0)
    n = len(y)

    cal = get_periods(freq_str)
    P = cal[0] if cal else period
    secondary = cal[1:] if len(cal) > 1 else []

    # Need >= 3 complete primary periods
    n_complete = n // P if P >= 2 else 0
    if P < 2 or n_complete < 3:
        return flar_v5(y_raw, horizon, period, freq_str, n_samples)

    # --- Context: use more data for Level (up to 500 periods) ---
    max_level_ctx = 500
    if n_complete > max_level_ctx:
        usable = max_level_ctx * P
        y = y[-usable:]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]

    # --- Reshape ---
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

    # --- Shape: proportions from last K periods ---
    K = min(5, n_complete)
    recent = mat[:, -K:]  # (P, K)
    recent_totals = recent.sum(axis=0, keepdims=True)  # (1, K)
    S = np.where(recent_totals > 1e-10,
                 recent / recent_totals,
                 1.0 / P).mean(axis=1)  # (P,)
    S = S / max(S.sum(), 1e-10)  # normalize to sum=1

    # --- Level: period totals ---
    L = mat.sum(axis=0)  # (n_complete,)

    # --- Box-Cox on Level ---
    lam = _bc_lambda(L)
    L_bc = _bc(L + 1, lam)

    # --- NLinear on Level ---
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # --- Cross-period features for Level ---
    cross_periods = []
    for sp in secondary:
        cp = sp // P
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)

    max_cp = max(cross_periods) if cross_periods else 0
    start = max(1, max_cp) if max_cp >= 2 else 1

    if n_complete <= start + 1:
        return flar_v5(y_raw, horizon, period, freq_str, n_samples)

    n_train = n_complete - start
    t = np.arange(n_complete, dtype=float)
    trend = t / float(n_complete)
    cols = [np.ones(n_complete), trend]
    for cp in cross_periods:
        cols.append(np.cos(2*np.pi*t/cp))
        cols.append(np.sin(2*np.pi*t/cp))
    nb = len(cols)
    base = np.column_stack(cols)

    n_lag = 1 + (1 if max_cp >= 2 else 0)
    nf = nb + n_lag

    X = np.zeros((n_train, nf))
    X[:, :nb] = base[start:]
    X[:, nb] = L_innov[start-1:-1]  # lag-1
    if max_cp >= 2:
        X[:, nb+1] = L_innov[start-max_cp:n_complete-max_cp]
    y_train = L_innov[start:]

    # --- ONE Ridge SA ---
    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # --- Forecast Level for m periods ---
    m = int(np.ceil(horizon / P))
    L_innov_ext = np.concatenate([L_innov, np.zeros(m)])
    L_hat_innov = np.zeros(m)

    for j in range(m):
        ti = n_complete + j
        x = np.zeros(nf)
        x[0] = 1.0
        x[1] = ti / float(n_complete)
        col = 2
        for cp in cross_periods:
            x[col] = np.cos(2*np.pi*ti/cp)
            x[col+1] = np.sin(2*np.pi*ti/cp)
            col += 2
        x[nb] = L_innov_ext[ti-1]
        if max_cp >= 2: x[nb+1] = L_innov_ext[ti-max_cp]
        pred = x @ beta
        L_hat_innov[j] = pred
        L_innov_ext[ti] = pred

    # --- Inverse: NLinear → Box-Cox inverse → Level in original space ---
    L_hat_bc = L_hat_innov + last_L
    L_hat = np.maximum(_bc_inv(L_hat_bc, lam) - 1, 0.0)

    # --- Reconstruct: Level x Shape ---
    # phase_fc(k, j) = L_hat(k) * S(j)
    phase_fc = L_hat[:, np.newaxis] * S[np.newaxis, :]  # (m, P)
    fc = phase_fc.reshape(-1)[:horizon]
    point_fc = np.maximum(fc, 0.0)

    # --- LOO Conformal ---
    # LOO residuals on Level → distribute by Shape for per-phase residuals
    L_fitted_innov = y_train - loo_resid
    L_fitted_bc = L_fitted_innov + last_L
    L_actual = np.maximum(_bc_inv(L_bc[start:], lam) - 1, 0.0)
    L_fitted = np.maximum(_bc_inv(L_fitted_bc, lam) - 1, 0.0)

    # Per-phase residuals: compare actual phase values with Level_hat × Shape
    loo_orig_list = []
    for i in range(n_train):
        period_idx = start + i
        for j in range(P):
            actual_val = mat[j, period_idx]
            predicted_val = L_fitted[i] * S[j]
            resid = actual_val - predicted_val
            if np.isfinite(resid):
                loo_orig_list.append(resid)

    loo_orig = np.array(loo_orig_list) if loo_orig_list else \
               np.array([-1,0,1]) * max(np.mean(point_fc)*0.05, 1e-6)
    if len(loo_orig) < 3:
        loo_orig = np.array([-1,0,1]) * max(np.mean(point_fc)*0.05, 1e-6)

    recent_loo = loo_orig[-min(200, len(loo_orig)):]
    drawn = np.random.choice(recent_loo, size=(n_samples, horizon), replace=True)
    jitter = np.random.normal(0, np.std(recent_loo)*0.1, size=(n_samples, horizon))
    samples = np.maximum(0, point_fc[np.newaxis,:] + drawn + jitter)
    rm = np.max(y[-max(horizon*2,50):])
    if rm > 0: samples = np.clip(samples, 0, rm*3)
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

# =========================================================================
# Evaluation
# =========================================================================

METRICS = [
    MASE(forecast_type=0.5),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

CONFIGS = [
    ("bitbrains_fast_storage", "5T", "short"),
    ("bitbrains_rnd", "H", "short"),
    ("bizitobs_l2c", "5T", "short"),
    ("electricity", "H", "short"),
    ("solar", "H", "short"),
    ("car_parts", "M", "short"),
    ("restaurant", "D", "short"),
    ("hospital", "M", "short"),
    ("saugeen", "D", "short"),
    ("m4_yearly", "A", "short"),
    ("m4_monthly", "M", "short"),
    ("m4_hourly", "H", "short"),
    ("sz_taxi", "15T", "short"),
    ("electricity", "H", "medium"),
    ("electricity", "H", "long"),
    ("solar", "H", "medium"),
    ("loop_seattle", "H", "medium"),
    ("bizitobs_l2c", "5T", "medium"),
    ("sz_taxi", "15T", "medium"),
]

_SN_PATH = '/tmp/gift-eval/results/seasonal_naive/all_results.csv'
_SN_DF = pd.read_csv(_SN_PATH) if os.path.exists(_SN_PATH) else None
def _get_sn(cid):
    if _SN_DF is None: return None, None
    row = _SN_DF[_SN_DF['dataset'] == cid]
    if row.empty: return None, None
    return (row['eval_metrics/MASE[0.5]'].iloc[0],
            row['eval_metrics/mean_weighted_sum_quantile_loss'].iloc[0])

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

def run_variant(label, forecast_fn):
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
        if os.path.isdir(os.path.join(dp, freq)): load = f"{load}/{freq}"
        try:
            dataset = Dataset(name=load, term=term, to_univariate=False)
        except: print(f"  [{i:>2}] {cid} SKIP"); continue
        period = get_period(dataset.freq)
        pred = Predictor(dataset.prediction_length, period, dataset.freq, 20, forecast_fn)
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
        print(f"  [{i:>2}] {label:4s} {cid:40s} MASE={mase:.4f}  CRPS={crps:.4f}  ({elapsed:.0f}s)")
        results.append({'config': cid, 'mase': mase, 'crps': crps,
                        'sn_mase': sn_m, 'sn_crps': sn_c, 'time': elapsed,
                        'term': term})
    return results

if __name__ == '__main__':
    total_start = time.perf_counter()
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    all_results = {}
    for label, fn in [('V5', flar_v5), ('V9', flar_v9)]:
        print(f"\n{'='*70}")
        print(f"  Variant: {label}")
        print(f"{'='*70}")
        all_results[label] = run_variant(label, fn)

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
    print("DELTA V9 vs V5")
    print(f"{'='*70}")
    v5m = {r['config']: r for r in all_results.get('V5', [])}
    ds, dm = [], []
    for r in all_results.get('V9', []):
        v5 = v5m.get(r['config'])
        if v5:
            d = r['mase'] - v5['mase']
            mk = "<<" if d<-0.005 else (">>" if d>0.005 else "  ")
            print(f"  {r['config']:40s} [{r['term']:6s}] {d:+.4f} {mk}  (V5={v5['mase']:.3f} → V9={r['mase']:.3f})")
            (ds if r['term']=='short' else dm).append(d)
    if ds: print(f"\n  SHORT:   avg={np.mean(ds):+.4f}  wins={sum(1 for d in ds if d<-1e-6)}/{len(ds)}")
    if dm: print(f"  MED/LONG: avg={np.mean(dm):+.4f}  wins={sum(1 for d in dm if d<-1e-6)}/{len(dm)}")

    print(f"\nWall time: {time.perf_counter()-total_start:.0f}s")
