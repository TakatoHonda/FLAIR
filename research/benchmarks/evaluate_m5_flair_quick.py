#!/usr/bin/env python3
"""Quick FLAIR evaluation on M5 (Level 12 item-level WRMSSE).

Runs FLAIR-DS on all 30,490 M5 bottom-level series and computes WRMSSE.
Compares with SeasonalNaive and Kaggle top scores.

Usage:
    uv run python -u research/benchmarks/evaluate_m5_flair_quick.py

Output:
    results/17_m5_flair/m5_flair_quick_report.md
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flair_ds import flair_ds
from evaluate_m5_wrmsse import (
    calc_rmsse, calc_wrmsse_level, seasonal_naive_forecast,
    croston_tsb_forecast, drift_seasonal_forecast,
)

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
M5_DIR = os.path.join(PROJECT_ROOT, 'data', 'm5-forecasting-accuracy')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', '17_m5_flair')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 28
KAGGLE_SCORES = [
    ("1st DRFAM", 0.52060),
    ("2nd Anderer", 0.52816),
    ("3rd Jeon&Seong", 0.53200),
    ("SeasonalNaive", None),
    ("FLAIR-DS", None),
]


def load_m5_data():
    """Load M5 sales data and compute revenue weights."""
    print("Loading M5 data...")
    sales = pd.read_csv(os.path.join(M5_DIR, 'sales_train_evaluation.csv'))
    calendar = pd.read_csv(os.path.join(M5_DIR, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(M5_DIR, 'sell_prices.csv'))

    # Extract day columns
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    n_days = len(day_cols)
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    # Train = d_1..d_1913, Test = d_1914..d_1941
    train_cols = [f'd_{i}' for i in range(1, 1914)]
    test_cols = [f'd_{i}' for i in range(1914, 1942)]

    train_data = sales[train_cols].values.astype(float)
    test_data = sales[test_cols].values.astype(float)

    # Revenue weights: last 28 days of train × latest price (vectorized)
    last28_cols = [f'd_{i}' for i in range(1886, 1914)]
    last28_sales = sales[last28_cols].values.astype(float)
    total_last28 = last28_sales.sum(axis=1)

    # Get latest price per item-store via merge
    latest_prices = prices.groupby(['item_id', 'store_id'])['sell_price'].last().reset_index()
    sales_with_price = sales[['item_id', 'store_id']].merge(
        latest_prices, on=['item_id', 'store_id'], how='left')
    price_vec = sales_with_price['sell_price'].fillna(1.0).values
    revenue = total_last28 * price_vec

    print(f"  {len(sales)} series, {train_data.shape[1]} train days, {test_data.shape[1]} test days")
    return train_data, test_data, revenue, sales[id_cols]


def flair_routed(y_train, horizon, period=7, freq_str='D', n_samples=20):
    """FLAIR with intermittency-aware routing.

    Routes sparse/lumpy series to specialized methods:
    - zero_ratio > 0.3 → Croston TSB (intermittent demand specialist)
    - CV of non-zero > 1.5 → Drift + Seasonal (lumpy demand)
    - otherwise → FLAIR-DS (periodic structure exploitation)
    """
    y = np.asarray(y_train, float)
    recent = y[-min(365, len(y)):]  # classify on recent year

    zero_ratio = np.mean(recent == 0)
    nz = recent[recent > 0]

    if zero_ratio > 0.3:
        # Intermittent: Croston TSB
        fc = croston_tsb_forecast(y, horizon)
        # Add seasonal overlay from last period
        tail = y[-period:]
        seasonal = tail - tail.mean()
        seasonal_fc = np.tile(seasonal, (horizon // period) + 1)[:horizon]
        return np.maximum(0, fc + seasonal_fc * 0.3)

    if len(nz) > 1 and nz.mean() > 0 and (nz.std() / nz.mean()) > 1.5:
        # Lumpy: drift + seasonal
        return drift_seasonal_forecast(y, horizon, period)

    # Dense: FLAIR-DS
    samples = flair_ds(y, horizon, period, freq_str, n_samples)
    return np.maximum(0, samples.mean(axis=0))


def main():
    t0 = time.perf_counter()
    train_data, test_data, revenue, id_df = load_m5_data()
    n_series = len(train_data)

    # --- Forecasts: FLAIR-DS, FLAIR-Routed, SN ---
    print(f"\nRunning on {n_series} series (H={HORIZON})...")
    flair_preds = np.zeros((n_series, HORIZON))
    routed_preds = np.zeros((n_series, HORIZON))
    sn_preds = np.zeros((n_series, HORIZON))
    route_counts = {'flair': 0, 'croston': 0, 'drift': 0}
    errors = 0

    t_start = time.perf_counter()
    for i in range(n_series):
        if (i + 1) % 5000 == 0 or i == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1:>6}/{n_series}] ({elapsed:.0f}s) routes: {route_counts}")

        y_train = train_data[i]
        sn_preds[i] = seasonal_naive_forecast(y_train, HORIZON, period=7)

        # Pure FLAIR-DS
        try:
            samples = flair_ds(y_train, HORIZON, period=7, freq_str='D', n_samples=20)
            flair_preds[i] = np.maximum(0, samples.mean(axis=0))
        except Exception:
            flair_preds[i] = sn_preds[i]
            errors += 1

        # FLAIR-Routed
        try:
            recent = y_train[-min(365, len(y_train)):]
            zero_ratio = np.mean(recent == 0)
            nz = recent[recent > 0]
            if zero_ratio > 0.3:
                route_counts['croston'] += 1
            elif len(nz) > 1 and nz.mean() > 0 and (nz.std() / nz.mean()) > 1.5:
                route_counts['drift'] += 1
            else:
                route_counts['flair'] += 1
            routed_preds[i] = flair_routed(y_train, HORIZON)
        except Exception:
            routed_preds[i] = sn_preds[i]

    t_total = time.perf_counter() - t_start
    print(f"  Done in {t_total:.0f}s ({t_total/60:.1f}min), {errors} errors")
    print(f"  Routes: {route_counts}")

    # --- Compute WRMSSE (Level 12: item-store level) ---
    print("\nComputing WRMSSE (Level 12)...")

    all_preds = {'SN': sn_preds, 'FLAIR-DS': flair_preds, 'FLAIR-Routed': routed_preds}
    rmsse = {}
    wrmsse = {}
    for name, preds in all_preds.items():
        rmsse[name] = np.array([calc_rmsse(test_data[i], preds[i], train_data[i])
                                for i in range(n_series)])
        wrmsse[name] = calc_wrmsse_level(rmsse[name], revenue)

    total_time = time.perf_counter() - t0

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"M5 FLAIR Evaluation (with Intermittency Router)")
    print(f"{'='*60}")
    print(f"\nRouting breakdown: {route_counts}")
    print(f"\nLevel 12 (Item-Store) WRMSSE:")
    sn_w = wrmsse['SN']
    for name in ['SN', 'FLAIR-DS', 'FLAIR-Routed']:
        imp = (sn_w - wrmsse[name]) / sn_w * 100
        mean_r = np.nanmean(rmsse[name])
        wins_vs_sn = int(np.sum(rmsse[name] < rmsse['SN'])) if name != 'SN' else 0
        print(f"  {name:15s}: WRMSSE={wrmsse[name]:.5f}  mean_RMSSE={mean_r:.4f}  vs_SN={imp:+.1f}%  wins={wins_vs_sn}")

    print(f"\nKaggle reference: 1st=0.52060, 2nd=0.52816, 3rd=0.53200")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    report_path = os.path.join(OUTPUT_DIR, 'm5_flair_routed_report.md')
    with open(report_path, 'w') as f:
        f.write(f"# M5 FLAIR Evaluation\n\n")
        f.write(f"## Routes: {route_counts}\n\n")
        f.write(f"| Method | WRMSSE | mean RMSSE | vs SN |\n")
        f.write(f"|--------|--------|------------|-------|\n")
        for name in ['SN', 'FLAIR-DS', 'FLAIR-Routed']:
            imp = (sn_w - wrmsse[name]) / sn_w * 100
            f.write(f"| {name} | {wrmsse[name]:.5f} | {np.nanmean(rmsse[name]):.4f} | {imp:+.1f}% |\n")
    print(f"\nSaved: {report_path}")


if __name__ == '__main__':
    main()
