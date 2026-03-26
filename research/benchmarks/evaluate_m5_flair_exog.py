#!/usr/bin/env python3
"""M5 FLAIR with External Features (Stage 2).

Extends FLAIR's Level Ridge with calendar/price covariates
while keeping the closed-form GCV-SA solution (zero hyperparams).

Usage:
    uv run python -u research/benchmarks/evaluate_m5_flair_exog.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flar9 import (
    _bc_lambda, _bc, _bc_inv, _ridge_gcv_loo_softavg,
)
from evaluate_m5_wrmsse import (
    calc_rmsse, calc_wrmsse_level, seasonal_naive_forecast,
    croston_tsb_forecast,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M5_DIR = os.path.join(PROJECT_ROOT, 'data', 'm5-forecasting-accuracy')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', '17_m5_flair')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 28
P = 7  # weekly period


def flair_exog(y_raw, horizon, price_hist, price_fc, cal_hist, cal_fc):
    """FLAIR with external features injected into Level Ridge.

    Args:
        y_raw: 1D array of sales (length n)
        horizon: forecast horizon
        price_hist: (n,) normalized price history (or None)
        price_fc: (horizon,) normalized price forecast (or None)
        cal_hist: (n, n_cal_features) calendar features for history
        cal_fc: (horizon, n_cal_features) calendar features for forecast
    """
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    y_floor = y.min()
    y_shift = max(1 - y_floor, 1.0)
    y = y + y_shift
    n = len(y)

    n_complete = n // P
    if n_complete < 3:
        fc = np.full(horizon, y[-1] - y_shift)
        return fc

    if n_complete > 500:
        trim = 500 * P
        y = y[-trim:]
        if price_hist is not None:
            price_hist = price_hist[-trim:]
        cal_hist = cal_hist[-trim:]
        n = len(y)
        n_complete = n // P

    usable = n_complete * P
    y_trim = y[-usable:]
    mat = y_trim.reshape(n_complete, P).T  # (P, n_complete)

    # Shape: K=5 average proportions
    K = min(5, n_complete)
    recent = mat[:, -K:]
    totals = recent.sum(axis=0, keepdims=True)
    props = np.where(totals > 1e-10, recent / totals, 1.0 / P)
    S = props.mean(axis=1)
    S = S / max(S.sum(), 1e-10)

    # Level
    L = mat.sum(axis=0)

    # Box-Cox + NLinear
    lam = _bc_lambda(L)
    L_bc = _bc(L, lam)
    last_L = L_bc[-1]
    L_innov = L_bc - last_L

    # --- Build feature matrix with external covariates ---
    # Aggregate calendar/price features to period level (sum or mean)
    cal_trim = cal_hist[-usable:]
    n_cal = cal_trim.shape[1]

    # Aggregate cal features per period (mean within each period)
    cal_periods = cal_trim.reshape(n_complete, P, n_cal).mean(axis=1)  # (n_complete, n_cal)

    # Aggregate price per period (mean)
    if price_hist is not None:
        price_trim = price_hist[-usable:]
        price_periods = price_trim.reshape(n_complete, P).mean(axis=1)  # (n_complete,)
        # Price momentum: diff of price
        price_mom = np.concatenate([[0], np.diff(price_periods)])
    else:
        price_periods = None

    # Cross-period Fourier (cp=52 for yearly)
    t = np.arange(n_complete, dtype=float)
    cp = 52
    cols = [np.ones(n_complete), t / n_complete]  # intercept + trend
    if n_complete > 2 * cp:
        cols.append(np.cos(2 * np.pi * t / cp))
        cols.append(np.sin(2 * np.pi * t / cp))

    # Calendar features
    for j in range(n_cal):
        cols.append(cal_periods[:, j])

    # Price features
    if price_periods is not None:
        cols.append(price_periods)
        cols.append(price_mom)

    nb = len(cols)
    base = np.column_stack(cols)

    # Lags
    start = max(1, cp if n_complete > 2 * cp else 1)
    n_lag = 1 + (1 if n_complete > 2 * cp else 0)
    nf = nb + n_lag
    n_train = n_complete - start

    if n_train < 3:
        L_hat = np.full(int(np.ceil(horizon / P)), _bc_inv(last_L, lam))
        fc = (L_hat[:, None] * S[None, :]).reshape(-1)[:horizon] - y_shift
        return np.maximum(0, fc)

    X = np.zeros((n_train, nf))
    X[:, :nb] = base[start:]
    X[:, nb] = L_innov[start - 1:-1]  # lag-1
    if n_lag > 1:
        X[:, nb + 1] = L_innov[start - cp:n_complete - cp]  # lag-cp
    y_train = L_innov[start:]

    beta, loo_resid, _ = _ridge_gcv_loo_softavg(X, y_train)

    # --- Forecast ---
    m = int(np.ceil(horizon / P))

    # Prepare future calendar/price features
    cal_fc_periods = cal_fc[:horizon].reshape(-1, P, n_cal)
    # Pad if needed
    n_fc_periods = cal_fc_periods.shape[0]
    if n_fc_periods < m:
        pad = np.tile(cal_fc_periods[-1:], (m - n_fc_periods, 1, 1))
        cal_fc_periods = np.concatenate([cal_fc_periods, pad])
    cal_fc_agg = cal_fc_periods[:m].mean(axis=1)  # (m, n_cal)

    if price_fc is not None and price_periods is not None:
        price_fc_periods = price_fc[:horizon].reshape(-1, P).mean(axis=1)
        if len(price_fc_periods) < m:
            price_fc_periods = np.concatenate([
                price_fc_periods, np.full(m - len(price_fc_periods), price_fc_periods[-1])])
        price_fc_mom = np.concatenate([[price_fc_periods[0] - price_periods[-1]],
                                        np.diff(price_fc_periods)])
    else:
        price_fc_periods = None

    L_innov_ext = np.concatenate([L_innov, np.zeros(m)])
    for j in range(m):
        ti = n_complete + j
        x = np.zeros(nf)
        # Base features
        x[0] = 1.0
        x[1] = ti / n_complete
        col = 2
        if n_complete > 2 * cp:
            x[col] = np.cos(2 * np.pi * ti / cp)
            x[col + 1] = np.sin(2 * np.pi * ti / cp)
            col += 2
        # Calendar
        for c in range(n_cal):
            x[col] = cal_fc_agg[j, c]
            col += 1
        # Price
        if price_fc_periods is not None:
            x[col] = price_fc_periods[j]
            x[col + 1] = price_fc_mom[j]
            col += 2
        # Lags
        x[nb] = L_innov_ext[ti - 1]
        if n_lag > 1:
            x[nb + 1] = L_innov_ext[ti - cp]
        L_innov_ext[ti] = x @ beta

    L_hat = _bc_inv(L_innov_ext[n_complete:n_complete + m] + last_L, lam)
    fc = (L_hat[:, None] * S[None, :]).reshape(-1)[:horizon] - y_shift
    return np.maximum(0, fc)


def build_calendar_features(calendar_df, n_total):
    """Build calendar feature matrix (n_total, n_features) for d_1..d_n_total."""
    # Map d_1..d_n_total to calendar rows
    cal = calendar_df.copy()
    cal['d_num'] = cal['d'].str.replace('d_', '').astype(int)
    cal = cal.sort_values('d_num')

    features = []
    # Month sin/cos (yearly seasonality)
    month = cal['month'].values[:n_total].astype(float)
    features.append(np.sin(2 * np.pi * month / 12))
    features.append(np.cos(2 * np.pi * month / 12))

    # Day of month (normalized)
    dom = cal['date'].apply(lambda x: int(x.split('-')[2]) if isinstance(x, str) else 15).values[:n_total]
    features.append(dom / 31.0)

    # SNAP flags (one per state)
    for col in ['snap_CA', 'snap_TX', 'snap_WI']:
        features.append(cal[col].values[:n_total].astype(float))

    # Event binary (any event)
    event = (~cal['event_name_1'].isna()).astype(float).values[:n_total]
    features.append(event)

    return np.column_stack(features)  # (n_total, 7)


def build_price_features(prices_df, sales_df, calendar_df, n_days):
    """Build per-series normalized price matrix via vectorized merge.

    Returns: (n_series, n_days) normalized price
    """
    print("  Building price matrix (vectorized)...")
    t0 = time.perf_counter()

    cal = calendar_df[['d', 'wm_yr_wk']].copy()
    cal['d_num'] = cal['d'].str.replace('d_', '').astype(int)
    cal = cal[cal['d_num'] <= n_days].sort_values('d_num')
    wk_arr = cal['wm_yr_wk'].values  # (n_days,)

    # Assign row index to each series
    meta = sales_df[['item_id', 'store_id']].reset_index()
    meta.columns = ['series_idx', 'item_id', 'store_id']

    # Merge prices with series index
    pr = prices_df.merge(meta, on=['item_id', 'store_id'], how='inner')

    # Build wk → column mapping (only weeks in our range)
    unique_wks = sorted(set(wk_arr))
    wk_to_col = {wk: i for i, wk in enumerate(unique_wks)}
    n_wks = len(unique_wks)

    # Build (n_series, n_wks) sparse price matrix
    n_series = len(sales_df)
    wk_price = np.full((n_series, n_wks), np.nan)
    pr_wk_col = pr['wm_yr_wk'].map(wk_to_col)
    valid = pr_wk_col.notna()
    wk_price[pr.loc[valid, 'series_idx'].values, pr_wk_col[valid].values.astype(int)] = \
        pr.loc[valid, 'sell_price'].values

    # Forward-fill NaN (item wasn't sold that week → use last known price)
    for i in range(1, n_wks):
        mask = np.isnan(wk_price[:, i])
        wk_price[mask, i] = wk_price[mask, i - 1]
    # Back-fill remaining NaN
    for i in range(n_wks - 2, -1, -1):
        mask = np.isnan(wk_price[:, i])
        wk_price[mask, i] = wk_price[mask, i + 1]
    # Any still NaN → 1.0
    wk_price = np.nan_to_num(wk_price, nan=1.0)

    # Expand to daily: (n_series, n_days)
    day_wk_idx = np.array([wk_to_col.get(w, 0) for w in wk_arr])
    price_mat = wk_price[:, day_wk_idx]  # (n_series, n_days)

    # Normalize per series
    maxp = price_mat.max(axis=1, keepdims=True)
    maxp[maxp < 1e-6] = 1.0
    price_mat = price_mat / maxp

    print(f"  Price matrix: ({n_series}, {n_days}) built in {time.perf_counter()-t0:.1f}s")
    return price_mat


def main():
    t0 = time.perf_counter()

    # Load data
    sales = pd.read_csv(os.path.join(M5_DIR, 'sales_train_evaluation.csv'))
    calendar = pd.read_csv(os.path.join(M5_DIR, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(M5_DIR, 'sell_prices.csv'))

    train_cols = [f'd_{i}' for i in range(1, 1914)]
    test_cols = [f'd_{i}' for i in range(1914, 1942)]
    train_data = sales[train_cols].values.astype(float)
    test_data = sales[test_cols].values.astype(float)
    n_series = len(train_data)

    # Revenue weights
    last28 = sales[[f'd_{i}' for i in range(1886, 1914)]].values.astype(float).sum(axis=1)
    lp = prices.groupby(['item_id', 'store_id'])['sell_price'].last().reset_index()
    sp = sales[['item_id', 'store_id']].merge(lp, on=['item_id', 'store_id'], how='left')
    revenue = last28 * sp['sell_price'].fillna(1.0).values

    print(f"{n_series} series loaded")

    # Build calendar features for all days (train + test)
    print("Building calendar features...")
    n_total = 1941
    cal_all = build_calendar_features(calendar, n_total)  # (1941, 7)
    cal_hist = cal_all[:1913]  # (1913, 7)
    cal_fc = cal_all[1913:1941]  # (28, 7)
    n_cal = cal_all.shape[1]
    print(f"  {n_cal} calendar features")

    # Build price matrix (vectorized)
    price_mat = build_price_features(prices, sales, calendar, 1941)
    price_hist = price_mat[:, :1913]  # (n_series, 1913)
    price_fc = price_mat[:, 1913:1941]  # (n_series, 28)

    # --- Run methods ---
    sn_preds = np.zeros((n_series, HORIZON))
    flair_preds = np.zeros((n_series, HORIZON))
    exog_preds = np.zeros((n_series, HORIZON))
    route_counts = {'flair_exog': 0, 'croston': 0}
    errors = 0

    print(f"\nRunning on {n_series} series...")
    t1 = time.perf_counter()

    # Determine which state each series belongs to (for SNAP routing)
    state_map = sales['state_id'].values
    snap_col_map = {'CA': 3, 'TX': 4, 'WI': 5}  # column indices in cal_all

    for i in range(n_series):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  [{i+1:>6}/{n_series}] ({time.perf_counter()-t1:.0f}s) routes={route_counts}")

        y = train_data[i]
        sn_preds[i] = seasonal_naive_forecast(y, HORIZON, 7)

        # Classify
        recent = y[-365:]
        zr = np.mean(recent == 0)

        if zr > 0.3:
            # Sparse: Croston TSB + seasonal overlay + price adjustment
            fc = croston_tsb_forecast(y, HORIZON)
            tail = y[-7:]
            seasonal = tail - tail.mean()
            base_fc = np.maximum(0, fc + np.tile(seasonal, 4)[:HORIZON] * 0.3)
            flair_preds[i] = base_fc

            # Price-adjusted: if price drops → demand up, price rises → demand down
            p_hist_i = price_hist[i]
            p_fc_i = price_fc[i]
            # Recent average normalized price (last 28 days)
            recent_price = p_hist_i[-28:].mean()
            if recent_price > 1e-6:
                price_ratio = recent_price / np.maximum(p_fc_i, 1e-6)
                # Damped elasticity: sqrt to avoid overreaction
                price_adj = np.sqrt(np.clip(price_ratio, 0.5, 2.0))
                exog_preds[i] = np.maximum(0, base_fc * price_adj)
            else:
                exog_preds[i] = base_fc
            route_counts['croston'] += 1
        else:
            route_counts['flair_exog'] += 1
            # Dense: FLAIR with calendar exog
            # Select relevant SNAP column for this series' state
            state = state_map[i]
            snap_idx = snap_col_map.get(state, 3)

            # Build per-series calendar: month_sin, month_cos, dom, snap_this_state, event
            cal_h = np.column_stack([cal_hist[:, 0], cal_hist[:, 1], cal_hist[:, 2],
                                      cal_hist[:, snap_idx], cal_hist[:, 6]])
            cal_f = np.column_stack([cal_fc[:, 0], cal_fc[:, 1], cal_fc[:, 2],
                                      cal_fc[:, snap_idx], cal_fc[:, 6]])

            try:
                exog_preds[i] = flair_exog(
                    y, HORIZON,
                    price_hist[i], price_fc[i],
                    cal_h, cal_f)
            except Exception:
                exog_preds[i] = sn_preds[i]
                errors += 1

            # Also run plain FLAIR for comparison
            try:
                from run_gift_eval_flair_ds import flair_ds
                flair_preds[i] = np.maximum(0, flair_ds(y, HORIZON, 7, 'D', 20).mean(axis=0))
            except Exception:
                flair_preds[i] = sn_preds[i]

    elapsed = time.perf_counter() - t1
    print(f"  Done in {elapsed:.0f}s ({elapsed/60:.1f}min), {errors} errors, routes={route_counts}")

    # --- WRMSSE ---
    print("\nComputing WRMSSE...")
    methods = {
        'SN': sn_preds,
        'FLAIR-Routed': flair_preds,
        'FLAIR-Exog+Price': exog_preds,
    }

    sn_rmsse = np.array([calc_rmsse(test_data[i], sn_preds[i], train_data[i]) for i in range(n_series)])
    sn_w = calc_wrmsse_level(sn_rmsse, revenue)

    for name, preds in methods.items():
        r = np.array([calc_rmsse(test_data[i], preds[i], train_data[i]) for i in range(n_series)])
        w = calc_wrmsse_level(r, revenue)
        mr = np.nanmean(r)
        imp = (sn_w - w) / sn_w * 100
        wins = int(np.sum(r < sn_rmsse)) if name != 'SN' else 0
        print(f"  {name:16s}: WRMSSE={w:.5f}  mean={mr:.4f}  vs_SN={imp:+.1f}%  wins={wins}")

    print(f"\nKaggle: 1st=0.521, Theta=1.001")
    print(f"Total: {time.perf_counter()-t0:.0f}s")


if __name__ == '__main__':
    main()
