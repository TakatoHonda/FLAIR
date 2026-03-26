#!/usr/bin/env python3
"""Visualize the FLAIR pipeline: Raw → Reshape → Level×Shape → Forecast.

Produces a multi-panel figure for selected GIFT-Eval datasets showing
each step of the FLAIR decomposition and forecasting process.

Usage:
    uv run python research/figures/visualize_flair_pipeline.py

Output:
    docs/flair_tech_report/fig_pipeline_demo.png
"""

import os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['GIFT_EVAL'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'gift-eval')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks'))

from gift_eval.data import Dataset
from run_gift_eval_flar9 import get_period, get_periods, _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Hiragino Sans'

OUTPUT_PATH = "docs/flair_tech_report/fig_pipeline_demo.png"

# Datasets to visualize: (name, freq, term, series_idx, display_name)
VIZ_CONFIGS = [
    ("electricity", "H", "short", 0, "Electricity (Hourly, P=24)"),
    ("sz_taxi", "15T", "short", 0, "SZ Taxi (15-min, P=4)"),
    ("restaurant", "D", "short", 0, "Restaurant (Daily, P=7)"),
]


def extract_flair_internals(y_raw, horizon, period, freq_str):
    """Run FLAIR pipeline and return all intermediate values for visualization."""
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    y_orig = y.copy()
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
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = min(n // p_cand, 500)
            mat_c = y[-(nc * p_cand):].reshape(nc, p_cand).T
            s = np.linalg.svd(mat_c, compute_uv=False)
            rss1 = np.sum(s[1:] ** 2)
            penalty = (p_cand + nc - 1) * np.log(nc * p_cand)
            bic = nc * p_cand * np.log(max(rss1 / (nc * p_cand), 1e-30)) + penalty
            if bic < best_bic:
                best_P, best_bic = p_cand, bic
        P = best_P

    secondary = [p for p in cal[1:] if p != P] if len(cal) > 1 else []

    n_complete = n // P
    if n_complete > 500:
        y = y[-(500 * P):]
        y_orig = y_orig[-(500 * P):]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

    # SVD
    U, s_vals, Vt = np.linalg.svd(mat, full_matrices=False)

    # Shape
    K = min(5, n_complete)
    recent = mat[:, -K:]
    recent_totals = recent.sum(axis=0, keepdims=True)
    proportions = np.where(recent_totals > 1e-10, recent / recent_totals, 1.0 / P)
    S = proportions.mean(axis=1)
    S = S / max(S.sum(), 1e-10)

    # Level
    L = mat.sum(axis=0)

    # Box-Cox + NLinear on Level
    lam = _bc_lambda(L)
    L_bc = _bc(L, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # Features
    cross_periods = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})
    max_cp = max(cross_periods) if cross_periods else 0
    start = max(1, max_cp) if max_cp >= 2 else 1

    while n_complete - start < 3 and cross_periods:
        cross_periods.pop()
        max_cp = max(cross_periods) if cross_periods else 0
        start = max(1, max_cp) if max_cp >= 2 else 1

    n_train = n_complete - start
    t = np.arange(n_complete, dtype=float)
    trend = t / float(n_complete)
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
    X[:, nb] = L_innov[start - 1:-1]
    if max_cp >= 2:
        X[:, nb + 1] = L_innov[start - max_cp:n_complete - max_cp]
    y_train = L_innov[start:]

    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # Forecast Level
    m = int(np.ceil(horizon / P))
    L_innov_ext = np.concatenate([L_innov, np.zeros(m)])
    for j in range(m):
        ti = n_complete + j
        x = np.zeros(nf)
        x[0] = 1.0
        x[1] = ti / float(n_complete)
        col = 2
        for cp in cross_periods:
            x[col] = np.cos(2 * np.pi * ti / cp)
            x[col + 1] = np.sin(2 * np.pi * ti / cp)
            col += 2
        x[nb] = L_innov_ext[ti - 1]
        if max_cp >= 2:
            x[nb + 1] = L_innov_ext[ti - max_cp]
        L_innov_ext[ti] = x @ beta

    L_hat = _bc_inv(L_innov_ext[n_complete:n_complete + m] + last_L, lam)
    fc = (L_hat[:, np.newaxis] * S[np.newaxis, :]).reshape(-1)[:horizon] - y_shift

    # Fitted Level
    L_fitted = _bc_inv(y_train - loo_resid + last_L, lam)

    return {
        'y_orig': y_orig,
        'y_shift': y_shift,
        'P': P,
        'n_complete': n_complete,
        'mat': mat,  # (P, n_complete) - shifted
        'mat_orig': mat - y_shift,  # original scale
        's_vals': s_vals,
        'S': S,
        'L': L - y_shift * P,  # original scale
        'L_hat': L_hat - y_shift * P,
        'L_fitted': L_fitted - y_shift * P,
        'start': start,
        'horizon': horizon,
        'fc': fc,
        'secondary': secondary,
        'cross_periods': cross_periods,
    }


def plot_one_dataset(fig, outer_gs, row, name, freq, term, series_idx, title):
    """Plot the FLAIR pipeline for one dataset in a row of subplots."""
    name_map = {
        'hierarchical_sales': 'favorita_sales', 'covid_deaths': 'covid_deaths_dataset',
        'kdd_cup_2018': 'kdd_cup', 'loop_seattle': 'loop_seattle_speed',
        'jena_weather': 'jena_weather_dataset',
    }
    load = name_map.get(name, name)
    dp = os.path.join(os.environ['GIFT_EVAL'], load)
    if os.path.isdir(os.path.join(dp, freq)):
        load = f"{load}/{freq}"
    dataset = Dataset(name=load, term=term, to_univariate=False)
    period = get_period(dataset.freq)

    # Get the first univariate series
    test_data = list(dataset.test_data)
    inp, label = test_data[series_idx]
    y_hist = np.asarray(inp['target'], float)
    y_true = np.asarray(label['target'], float)
    if y_hist.ndim > 1:
        y_hist = y_hist[0]
        y_true = y_true[0]
    horizon = dataset.prediction_length

    info = extract_flair_internals(y_hist, horizon, period, str(dataset.freq))

    # Create 5 subplots in this row
    inner = outer_gs.subgridspec(1, 5, width_ratios=[2.5, 2, 1, 2, 2.5], wspace=0.35)

    # --- Panel 1: Raw time series (last N periods + forecast) ---
    ax1 = fig.add_subplot(inner[0])
    n_show = min(info['n_complete'], 30) * info['P']
    y_show = info['y_orig'][-n_show:] - info['y_shift']
    t_hist = np.arange(len(y_show))
    t_fc = np.arange(len(y_show), len(y_show) + horizon)

    ax1.plot(t_hist, y_show, color='#2c3e50', linewidth=0.6, alpha=0.8)
    ax1.plot(t_fc, y_true, color='#2c3e50', linewidth=0.6, alpha=0.4, label='actual')
    ax1.plot(t_fc, info['fc'], color='#c0392b', linewidth=1.2, label='FLAIR')
    ax1.fill_between(t_fc, y_true, info['fc'], alpha=0.15, color='#c0392b')
    ax1.axvline(len(y_show) - 0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax1.set_title(f'(a) Raw + Forecast', fontsize=8, fontweight='bold')
    ax1.tick_params(labelsize=6)
    ax1.legend(fontsize=5, loc='upper left')
    if row == 0:
        ax1.set_title(f'(a) Raw + Forecast', fontsize=8, fontweight='bold')

    # --- Panel 2: Reshaped matrix heatmap ---
    ax2 = fig.add_subplot(inner[1])
    n_cols_show = min(info['n_complete'], 40)
    mat_show = info['mat_orig'][:, -n_cols_show:]
    im = ax2.imshow(mat_show, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_title(f'(b) Reshape (P={info["P"]})', fontsize=8, fontweight='bold')
    ax2.set_ylabel('phase', fontsize=6)
    ax2.set_xlabel('period', fontsize=6)
    ax2.tick_params(labelsize=5)

    # --- Panel 3: Shape vector ---
    ax3 = fig.add_subplot(inner[2])
    phases = np.arange(info['P'])
    ax3.barh(phases, info['S'], color='#e74c3c', alpha=0.8, height=0.7)
    ax3.set_ylim(info['P'] - 0.5, -0.5)
    ax3.set_title('(c) Shape', fontsize=8, fontweight='bold')
    ax3.set_ylabel('phase', fontsize=6)
    ax3.tick_params(labelsize=5)
    ax3.set_xlabel('proportion', fontsize=6)

    # --- Panel 4: Level series + forecast ---
    ax4 = fig.add_subplot(inner[3])
    L = info['L']
    n_L_show = min(len(L), 40)
    t_L = np.arange(n_L_show)
    ax4.plot(t_L, L[-n_L_show:], color='#2980b9', linewidth=1.0, label='Level (hist)')

    # Fitted values
    fit_start = max(0, n_L_show - len(info['L_fitted']))
    if fit_start < n_L_show and len(info['L_fitted']) > 0:
        n_fit_show = min(len(info['L_fitted']), n_L_show)
        t_fit = np.arange(n_L_show - n_fit_show, n_L_show)
        ax4.plot(t_fit, info['L_fitted'][-n_fit_show:], color='#27ae60',
                 linewidth=0.8, ls='--', alpha=0.7, label='Fitted')

    m = len(info['L_hat'])
    t_fc_L = np.arange(n_L_show, n_L_show + m)
    ax4.plot(t_fc_L, info['L_hat'], color='#c0392b', linewidth=1.5,
             marker='o', markersize=3, label='Forecast')
    ax4.axvline(n_L_show - 0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax4.set_title('(d) Level Series', fontsize=8, fontweight='bold')
    ax4.tick_params(labelsize=5)
    ax4.legend(fontsize=5, loc='upper left')

    # --- Panel 5: Reconstructed forecast detail ---
    ax5 = fig.add_subplot(inner[4])
    n_context = min(3 * info['P'], len(info['y_orig']))
    y_context = info['y_orig'][-n_context:] - info['y_shift']
    t_ctx = np.arange(len(y_context))
    t_fc2 = np.arange(len(y_context), len(y_context) + horizon)

    ax5.plot(t_ctx, y_context, color='#2c3e50', linewidth=0.7, alpha=0.7, label='history')
    ax5.plot(t_fc2, y_true, color='#7f8c8d', linewidth=0.8, ls='--', alpha=0.7, label='actual')
    ax5.plot(t_fc2, info['fc'], color='#c0392b', linewidth=1.2, label='Level×Shape')
    ax5.axvline(len(y_context) - 0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax5.set_title('(e) Level×Shape → Forecast', fontsize=8, fontweight='bold')
    ax5.tick_params(labelsize=5)
    ax5.legend(fontsize=5, loc='upper left')

    # Row label
    fig.text(0.01, outer_gs.get_position(fig).y0 +
             (outer_gs.get_position(fig).y1 - outer_gs.get_position(fig).y0) / 2,
             title, fontsize=9, fontweight='bold', rotation=90,
             va='center', ha='center')


def main():
    n_rows = len(VIZ_CONFIGS)
    fig = plt.figure(figsize=(18, 3.5 * n_rows + 0.5))
    outer = gridspec.GridSpec(n_rows, 1, hspace=0.45, left=0.08, right=0.97,
                              top=0.94, bottom=0.05)

    for i, (name, freq, term, sidx, title) in enumerate(VIZ_CONFIGS):
        print(f"Processing {name}/{freq}/{term}...")
        plot_one_dataset(fig, outer[i], i, name, freq, term, sidx, title)

    fig.suptitle('FLAIR Pipeline: Raw → Reshape → Shape × Level → Forecast',
                 fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()
