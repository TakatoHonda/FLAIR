#!/usr/bin/env python3
"""M5 Hierarchical FLAIR: Top-Down Disaggregation.

Forecasts at store×dept level (70 series) where FLAIR is strong,
then disaggregates to item level using recent sales proportions.

Usage:
    uv run python -u research/benchmarks/evaluate_m5_flair_hierarchical.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flair_ds import flair_ds
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


def load_m5():
    print("Loading M5 data...")
    sales = pd.read_csv(os.path.join(M5_DIR, 'sales_train_evaluation.csv'))
    prices = pd.read_csv(os.path.join(M5_DIR, 'sell_prices.csv'))

    train_cols = [f'd_{i}' for i in range(1, 1914)]
    test_cols = [f'd_{i}' for i in range(1914, 1942)]
    train = sales[train_cols].values.astype(float)
    test = sales[test_cols].values.astype(float)

    # Revenue weights
    last28 = sales[[f'd_{i}' for i in range(1886, 1914)]].values.astype(float).sum(axis=1)
    lp = prices.groupby(['item_id', 'store_id'])['sell_price'].last().reset_index()
    sp = sales[['item_id', 'store_id']].merge(lp, on=['item_id', 'store_id'], how='left')
    revenue = last28 * sp['sell_price'].fillna(1.0).values

    meta = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
    print(f"  {len(sales)} series")
    return train, test, revenue, meta


def main():
    t0 = time.perf_counter()
    train, test, revenue, meta = load_m5()
    n = len(train)

    # --- Method 1: Bottom-up FLAIR-Routed (baseline from Stage 1) ---
    print("\n[1/4] Bottom-up FLAIR-Routed...")
    t1 = time.perf_counter()
    bu_preds = np.zeros((n, HORIZON))
    for i in range(n):
        y = train[i]
        recent = y[-365:]
        zr = np.mean(recent == 0)
        if zr > 0.3:
            fc = croston_tsb_forecast(y, HORIZON)
            tail = y[-7:]
            seasonal = tail - tail.mean()
            bu_preds[i] = np.maximum(0, fc + np.tile(seasonal, 4)[:HORIZON] * 0.3)
        else:
            try:
                bu_preds[i] = np.maximum(0, flair_ds(y, HORIZON, 7, 'D', 20).mean(axis=0))
            except Exception:
                bu_preds[i] = seasonal_naive_forecast(y, HORIZON, 7)
    print(f"  Done in {time.perf_counter()-t1:.0f}s")

    # --- Method 2: Top-Down from Store×Dept (70 groups) ---
    print("\n[2/4] Top-Down FLAIR (Store×Dept → Item)...")
    t2 = time.perf_counter()
    td_preds = np.zeros((n, HORIZON))

    groups = meta.groupby(['store_id', 'dept_id'])
    n_groups = groups.ngroups
    print(f"  {n_groups} store×dept groups")

    for g_idx, ((store, dept), g_df) in enumerate(groups):
        if (g_idx + 1) % 10 == 0:
            print(f"  [{g_idx+1}/{n_groups}] {store}/{dept} ({time.perf_counter()-t2:.0f}s)")

        idx = g_df.index.values
        # Aggregate: sum all items in this store×dept
        agg_train = train[idx].sum(axis=0)
        agg_test = test[idx].sum(axis=0)

        # FLAIR forecast on aggregated series
        try:
            agg_fc = np.maximum(0, flair_ds(agg_train, HORIZON, 7, 'D', 20).mean(axis=0))
        except Exception:
            agg_fc = seasonal_naive_forecast(agg_train, HORIZON, 7)

        # Disaggregate: item proportions from last 28 days (day-of-week aware)
        last28 = train[idx][:, -28:]  # (n_items, 28)
        # Use weekly proportions: for each forecast day, use the matching DOW proportion
        for h in range(HORIZON):
            dow = h % 7  # day-of-week in forecast
            # Average proportion from last 4 matching DOWs
            dow_sales = last28[:, dow::7]  # shape: (n_items, 4)
            item_totals = dow_sales.sum(axis=1)  # per-item total for this DOW
            group_total = item_totals.sum()
            if group_total > 0:
                props = item_totals / group_total
            else:
                props = np.ones(len(idx)) / len(idx)
            td_preds[idx, h] = agg_fc[h] * props

    td_preds = np.maximum(0, td_preds)
    print(f"  Done in {time.perf_counter()-t2:.0f}s")

    # --- Method 3: Hybrid (Top-Down for sparse, FLAIR for dense) ---
    print("\n[3/4] Hybrid (TD for sparse, FLAIR for dense)...")
    t3 = time.perf_counter()
    hybrid_preds = np.zeros((n, HORIZON))
    n_td, n_flair = 0, 0

    for i in range(n):
        recent = train[i][-365:]
        zr = np.mean(recent == 0)
        if zr > 0.3:
            # Sparse: use top-down allocation
            hybrid_preds[i] = td_preds[i]
            n_td += 1
        else:
            # Dense: use FLAIR directly
            hybrid_preds[i] = bu_preds[i]
            n_flair += 1

    print(f"  TD: {n_td}, FLAIR: {n_flair}")
    print(f"  Done in {time.perf_counter()-t3:.0f}s")

    # --- Method 4: SN baseline ---
    print("\n[4/4] SeasonalNaive baseline...")
    sn_preds = np.zeros((n, HORIZON))
    for i in range(n):
        sn_preds[i] = seasonal_naive_forecast(train[i], HORIZON, 7)

    # --- WRMSSE ---
    print("\nComputing WRMSSE...")
    methods = {
        'SN': sn_preds,
        'FLAIR-Routed': bu_preds,
        'FLAIR-TopDown': td_preds,
        'FLAIR-Hybrid': hybrid_preds,
    }

    sn_w = None
    for name, preds in methods.items():
        r = np.array([calc_rmsse(test[i], preds[i], train[i]) for i in range(n)])
        w = calc_wrmsse_level(r, revenue)
        mr = np.nanmean(r)
        wins = int(np.sum(r < np.array([calc_rmsse(test[i], sn_preds[i], train[i])
                                         for i in range(n)]))) if name != 'SN' else 0
        if name == 'SN':
            sn_w = w
        imp = (sn_w - w) / sn_w * 100
        print(f"  {name:16s}: WRMSSE={w:.5f}  mean={mr:.4f}  vs_SN={imp:+.1f}%  wins={wins}")

    total = time.perf_counter() - t0
    print(f"\nKaggle: 1st=0.521, Theta=1.001")
    print(f"Total: {total:.0f}s ({total/60:.1f}min)")


if __name__ == '__main__':
    main()
