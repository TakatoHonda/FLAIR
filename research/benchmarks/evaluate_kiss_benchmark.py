#!/usr/bin/env python3
"""KISSデータセット ベース予測器ベンチマーク.

M5で検証したWP-MO-Theta手法を、KISSデータセット3種で追加検証する。
- nishimatsu: 日次小売売上 (多商品・間欠需要)
- restaurant: 日次レストラン売上 (8店舗)
- 小売出荷実績: 日次出荷 (7商品)

実行方法:
    python research/benchmarks/evaluate_kiss_benchmark.py

出力 (results/14_kiss_benchmark/):
    kiss_benchmark_report.md
    kiss_benchmark_comparison.png
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from statsmodels.tsa.seasonal import MSTL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from collections import Counter

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Hiragino Sans'
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
KISS_DIR = os.path.join(PROJECT_ROOT, 'data', 'kiss')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', '14_kiss_benchmark')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Forecasting functions (same as M5 evaluation)
# ============================================================
def mstl_ets_forecast(y_train, horizon, periods=None, base_period=7):
    y_train = np.asarray(y_train, float)
    y_work = np.maximum(y_train, 1e-6)
    if periods is None:
        periods = [base_period] + _get_mstl_long_periods(base_period, len(y_work))
    periods = [p for p in periods if 2 <= p <= len(y_work) // 2]
    if not periods:
        periods = [base_period] if len(y_work) >= base_period * 2 else [max(2, len(y_work) // 4)]
    try:
        mstl = MSTL(pd.Series(y_work), periods=periods).fit()
        trend = mstl.trend.values
        seasonal_sum = np.zeros_like(y_work)
        for col in mstl.seasonal.columns:
            seasonal_sum += mstl.seasonal[col].values
    except Exception:
        trend = y_work.copy()
        seasonal_sum = np.zeros_like(y_work)
    try:
        ets = ExponentialSmoothing(trend, trend="add", seasonal=None).fit(optimized=True)
        trend_fc = ets.forecast(horizon)
    except Exception:
        trend_fc = np.full(horizon, trend[-1])
    max_period = max(periods)
    seasonal_tail = seasonal_sum[-max_period:]
    seasonal_fc = np.tile(seasonal_tail, (horizon // max_period) + 1)[:horizon]
    return np.maximum(0.0, trend_fc + seasonal_fc)


def seasonal_naive_forecast(y_train, horizon, period=7):
    tail = y_train[-period:]
    return np.tile(tail, (horizon // period) + 1)[:horizon]


def theta_forecast(y_train, horizon, period=7):
    y_train = np.asarray(y_train, float)
    y_work = np.maximum(y_train, 1e-6)
    try:
        model = ThetaModel(pd.Series(y_work), period=period).fit()
        fc = model.forecast(horizon)
        return np.maximum(0.0, fc.values)
    except Exception:
        return seasonal_naive_forecast(y_train, horizon, period)


def hw_seasonal_forecast(y_train, horizon, period=7):
    y_train = np.asarray(y_train, float)
    y_work = np.maximum(y_train, 1e-6)
    try:
        model = ExponentialSmoothing(
            y_work, trend='add', seasonal='add',
            seasonal_periods=period).fit(optimized=True)
        fc = model.forecast(horizon)
        return np.maximum(0.0, fc)
    except Exception:
        return seasonal_naive_forecast(y_train, horizon, period)


def _get_mstl_long_periods(base_period, n_obs):
    """Thetaのbase_periodより長い周期で、データ長が1.5倍以上あるものを返す.

    日次(base_period=7): Thetaが7日を乗法処理 → MSTLは14,28,91,365日を分離
    週次(base_period=4): Thetaが4週を乗法処理 → MSTLは13,52週を分離
    """
    candidates = {
        7:   [91, 365],            # 四半期, 年次 (14/28はTheta period=7の高調波なので除外)
        4:   [13, 52],             # 四半期, 年次
        12:  [52],                 # 年次 (月次base)
        52:  [],
        365: [],
    }
    raw = candidates.get(base_period, [])
    min_factor = 1.5  # 1.5周期分のデータがあれば抽出可能
    return [p for p in raw if n_obs >= p * min_factor]


def mstl_theta_forecast(y_train, horizon, periods=None, base_period=7):
    """MSTL乗法分解(log1p空間) + Theta乗法短周期 (split approach).

    log1p空間でMSTLを適用 = 原空間で乗法的に長周期季節性を分離。
    短周期(週次等)はThetaの内蔵乗法分解に任せる。
    全ての分解が乗法的に統一されるため、加法/乗法の不整合が解消される。
    """
    y_train = np.asarray(y_train, float)
    y_work = np.maximum(y_train, 0.0)
    long_periods = _get_mstl_long_periods(base_period, len(y_work))

    # log1p空間でMSTL → 乗法的に長周期季節性を分離
    long_mult = np.ones_like(y_work)  # 乗法的季節因子 (1.0 = 季節性なし)
    max_long = 0
    if long_periods:
        try:
            log_y = np.log1p(y_work)
            mstl = MSTL(pd.Series(log_y), periods=long_periods).fit()
            log_seasonal = np.zeros_like(log_y)
            for col in mstl.seasonal.columns:
                log_seasonal += mstl.seasonal[col].values
            long_mult = np.exp(log_seasonal)  # log空間の加法 → 原空間の乗法
            max_long = max(long_periods)
        except Exception:
            pass

    # 乗法的に長期季節性を除去 → Thetaがbase_periodを乗法処理
    y_delong = np.maximum(y_work / long_mult, 1e-6)
    try:
        model = ThetaModel(pd.Series(y_delong), period=base_period).fit()
        fc_delong = model.forecast(horizon).values
    except Exception:
        fc_delong = seasonal_naive_forecast(y_delong, horizon, base_period)

    # 乗法的に長期季節性を再付加
    if max_long > 0:
        long_mult_tail = long_mult[-max_long:]
        long_mult_fc = np.tile(long_mult_tail, (horizon // max_long) + 1)[:horizon]
    else:
        long_mult_fc = np.ones(horizon)

    return np.maximum(0.0, fc_delong * long_mult_fc)



def calc_rmsse(y_true, y_pred, y_train):
    mse_forecast = np.mean((y_true - y_pred) ** 2)
    naive_diffs = y_train[1:] - y_train[:-1]
    mse_naive = np.mean(naive_diffs ** 2)
    if mse_naive < 1e-12:
        return 0.0
    return np.sqrt(mse_forecast / mse_naive)


# ============================================================
# WP-MO-Theta pipeline
# ============================================================
def run_wpmo_theta(train_matrix, horizon, n_clusters=None, period=7,
                   base_forecaster='theta', half_life=90,
                   instab_window=90, instab_percentile=80):
    """WP-MO clustering + base forecaster + MO disaggregation.

    Returns: predictions (N, horizon), cluster_labels
    """
    N, T = train_matrix.shape
    if n_clusters is None:
        n_clusters = max(2, min(40, N // 5))

    forecaster_fn = {
        'theta': lambda y, h: theta_forecast(y, h, period),
        'mstl_ets': lambda y, h: mstl_ets_forecast(y, h, base_period=period),
        'mstl_theta': lambda y, h: mstl_theta_forecast(y, h, base_period=period),
        'hw_seasonal': lambda y, h: hw_seasonal_forecast(y, h, period),
        'seasonal_naive': lambda y, h: seasonal_naive_forecast(y, h, period),
    }[base_forecaster]

    # Weekly profile PCA
    n_weeks = T // period
    if n_weeks < 3:
        # Too short for PCA, use single cluster
        agg = train_matrix.sum(axis=0)
        fc = forecaster_fn(agg, horizon)
        lam = np.log(2.0) / half_life
        decay_w = np.exp(-lam * np.arange(T - 1, -1, -1))
        decay_w /= decay_w.sum()
        sw = (train_matrix * decay_w[np.newaxis, :]).sum(axis=1)
        total = sw.sum()
        shares = sw / total if total > 1e-12 else np.ones(N) / N
        pred = shares[:, np.newaxis] * fc[np.newaxis, :]
        return pred, np.zeros(N, dtype=int)

    weekly_train = train_matrix[:, :n_weeks * period].reshape(N, n_weeks, period).sum(axis=2)
    weekly_log = np.log1p(weekly_train)
    mu = weekly_log.mean(axis=1, keepdims=True)
    sigma = weekly_log.std(axis=1, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    weekly_z = (weekly_log - mu) / sigma
    n_pca = min(min(20, n_weeks), N)
    pca = PCA(n_components=n_pca, random_state=SEED)
    F_wp = pca.fit_transform(weekly_z)

    # Clustering
    n_sub = min(3000, N)
    if N <= n_sub:
        sub_idx = np.arange(N)
    else:
        sub_idx = np.random.choice(N, size=n_sub, replace=False)
        sub_idx.sort()

    actual_clusters = min(n_clusters, len(sub_idx))
    Z = linkage(F_wp[sub_idx], method='ward')
    sub_labels = fcluster(Z, t=actual_clusters, criterion='maxclust')

    if N <= n_sub:
        labels = sub_labels
    else:
        clf = NearestCentroid()
        clf.fit(F_wp[sub_idx], sub_labels)
        labels = clf.predict(F_wp)

    # Decay-weighted shares
    lam = np.log(2.0) / half_life
    decay_w = np.exp(-lam * np.arange(T - 1, -1, -1))
    decay_w /= decay_w.sum()
    series_weighted = (train_matrix * decay_w[np.newaxis, :]).sum(axis=1)

    # Instability detection
    recent_start = max(0, T - instab_window)
    series_recent = train_matrix[:, recent_start:].mean(axis=1)
    series_full = train_matrix.mean(axis=1)

    # Cluster forecasts + MO disaggregation
    pred = np.zeros((N, horizon))
    cluster_ids = sorted(np.unique(labels))
    for c_id in cluster_ids:
        mask = labels == c_id
        agg = train_matrix[mask].sum(axis=0)
        fc = forecaster_fn(agg, horizon)
        cw = series_weighted[mask]
        ct = cw.sum()
        shares = cw / ct if ct > 1e-12 else np.ones(mask.sum()) / mask.sum()
        pred[mask] = shares[:, np.newaxis] * fc[np.newaxis, :]

    # Risky series → seasonal naive fallback
    instab = np.zeros(N)
    for c_id in cluster_ids:
        mask = labels == c_id
        idx = np.where(mask)[0]
        sf = series_full[idx] / max(series_full[idx].sum(), 1e-12)
        sr = series_recent[idx] / max(series_recent[idx].sum(), 1e-12)
        instab[idx] = np.abs(sf - sr) / np.maximum(sf, 1e-6)
    if N > 1:
        risky_th = np.percentile(instab, instab_percentile)
        risky = instab >= risky_th
        for i in np.where(risky)[0]:
            pred[i] = seasonal_naive_forecast(train_matrix[i], horizon, period)

    return pred, labels


def evaluate_dataset(name, train_matrix, test_matrix, horizon, period=7,
                     n_clusters=None, series_names=None):
    """Evaluate multiple methods on a single dataset."""
    N, T = train_matrix.shape
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  Series: {N}, Train days: {T}, Test days: {horizon}, Period: {period}")
    print(f"{'='*60}")

    results = {}

    # 1. Seasonal Naive
    t0 = time.perf_counter()
    naive_pred = np.zeros((N, horizon))
    for i in range(N):
        naive_pred[i] = seasonal_naive_forecast(train_matrix[i], horizon, period)
    rmsse_arr = np.array([calc_rmsse(test_matrix[i], naive_pred[i], train_matrix[i]) for i in range(N)])
    results['Seasonal Naive'] = float(np.mean(rmsse_arr))
    print(f"  Seasonal Naive:  RMSSE={results['Seasonal Naive']:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # 2. MSTL+ETS (individual)
    t0 = time.perf_counter()
    mstl_pred = np.zeros((N, horizon))
    for i in range(N):
        mstl_pred[i] = mstl_ets_forecast(train_matrix[i], horizon, base_period=period)
    rmsse_arr = np.array([calc_rmsse(test_matrix[i], mstl_pred[i], train_matrix[i]) for i in range(N)])
    results['MSTL+ETS'] = float(np.mean(rmsse_arr))
    print(f"  MSTL+ETS:        RMSSE={results['MSTL+ETS']:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # 3. Theta (individual)
    t0 = time.perf_counter()
    theta_pred = np.zeros((N, horizon))
    for i in range(N):
        theta_pred[i] = theta_forecast(train_matrix[i], horizon, period)
    rmsse_arr = np.array([calc_rmsse(test_matrix[i], theta_pred[i], train_matrix[i]) for i in range(N)])
    results['Theta'] = float(np.mean(rmsse_arr))
    print(f"  Theta:           RMSSE={results['Theta']:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # 4. Holt-Winters (individual)
    t0 = time.perf_counter()
    hw_pred = np.zeros((N, horizon))
    for i in range(N):
        hw_pred[i] = hw_seasonal_forecast(train_matrix[i], horizon, period)
    rmsse_arr = np.array([calc_rmsse(test_matrix[i], hw_pred[i], train_matrix[i]) for i in range(N)])
    results['Holt-Winters'] = float(np.mean(rmsse_arr))
    print(f"  Holt-Winters:    RMSSE={results['Holt-Winters']:.4f}  ({time.perf_counter()-t0:.1f}s")

    # 5. MSTL+Theta split (long-seasonal MSTL + short-seasonal Theta)
    t0 = time.perf_counter()
    mstl_theta_pred = np.zeros((N, horizon))
    for i in range(N):
        mstl_theta_pred[i] = mstl_theta_forecast(train_matrix[i], horizon, base_period=period)
    rmsse_arr = np.array([calc_rmsse(test_matrix[i], mstl_theta_pred[i], train_matrix[i]) for i in range(N)])
    results['MSTL+Theta'] = float(np.mean(rmsse_arr))
    print(f"  MSTL+Theta:      RMSSE={results['MSTL+Theta']:.4f}  ({time.perf_counter()-t0:.1f}s)")

    # WP-MO variants (only if enough series for clustering)
    if N >= 5:
        # 5. WP-MO (MSTL+ETS base)
        t0 = time.perf_counter()
        wpmo_pred, labels = run_wpmo_theta(
            train_matrix, horizon, n_clusters=n_clusters,
            period=period, base_forecaster='mstl_ets')
        rmsse_arr = np.array([calc_rmsse(test_matrix[i], wpmo_pred[i], train_matrix[i]) for i in range(N)])
        results['WP-MO'] = float(np.mean(rmsse_arr))
        n_cl = len(np.unique(labels))
        print(f"  WP-MO:           RMSSE={results['WP-MO']:.4f}  (clusters={n_cl}, {time.perf_counter()-t0:.1f}s)")

        # 6. WP-MO-Theta
        t0 = time.perf_counter()
        wpmo_theta_pred, labels_t = run_wpmo_theta(
            train_matrix, horizon, n_clusters=n_clusters,
            period=period, base_forecaster='theta')
        rmsse_arr = np.array([calc_rmsse(test_matrix[i], wpmo_theta_pred[i], train_matrix[i]) for i in range(N)])
        results['WP-MO-Theta'] = float(np.mean(rmsse_arr))
        print(f"  WP-MO-Theta:     RMSSE={results['WP-MO-Theta']:.4f}  ({time.perf_counter()-t0:.1f}s)")

        # 7. WP-MO-HW
        t0 = time.perf_counter()
        wpmo_hw_pred, _ = run_wpmo_theta(
            train_matrix, horizon, n_clusters=n_clusters,
            period=period, base_forecaster='hw_seasonal')
        rmsse_arr = np.array([calc_rmsse(test_matrix[i], wpmo_hw_pred[i], train_matrix[i]) for i in range(N)])
        results['WP-MO-HW'] = float(np.mean(rmsse_arr))
        print(f"  WP-MO-HW:       RMSSE={results['WP-MO-HW']:.4f}  ({time.perf_counter()-t0:.1f}s)")

        # 8. WP-MO-MSTL+Theta
        t0 = time.perf_counter()
        wpmo_mt_pred, _ = run_wpmo_theta(
            train_matrix, horizon, n_clusters=n_clusters,
            period=period, base_forecaster='mstl_theta')
        rmsse_arr = np.array([calc_rmsse(test_matrix[i], wpmo_mt_pred[i], train_matrix[i]) for i in range(N)])
        results['WP-MO-MSTL+Theta'] = float(np.mean(rmsse_arr))
        print(f"  WP-MO-MSTL+Theta: RMSSE={results['WP-MO-MSTL+Theta']:.4f}  ({time.perf_counter()-t0:.1f}s)")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    total_start = time.perf_counter()

    color_map = {
        'Seasonal Naive': '#95a5a6', 'MSTL+ETS': '#3498db',
        'Theta': '#2ecc71', 'Holt-Winters': '#e74c3c',
        'MSTL+Theta': '#27ae60',
        'WP-MO': '#9b59b6', 'WP-MO-Theta': '#d35400', 'WP-MO-HW': '#e67e22',
        'WP-MO-MSTL+Theta': '#c0392b',
    }

    # ================================================================
    # Load all datasets once, then evaluate at multiple horizons
    # ================================================================

    # --- Restaurant ---
    print("Loading restaurant data...")
    df_rest = pd.read_csv(os.path.join(KISS_DIR, 'restaurant_detailed_data.csv'))
    df_rest['日付'] = pd.to_datetime(df_rest['日付'])
    stores = sorted(df_rest['店舗'].unique())
    pivot_rest = df_rest.pivot_table(
        index='店舗', columns='日付', values='売上(円)', fill_value=0)
    pivot_rest = pivot_rest.loc[stores]
    rest_matrix = pivot_rest.values.astype(np.float64)
    print(f"  Restaurant: {rest_matrix.shape[0]} stores × {rest_matrix.shape[1]} days")

    # --- 小売出荷 ---
    print("Loading 小売出荷実績...")
    df_retail = pd.read_csv(os.path.join(KISS_DIR, '【小売】出荷実績サンプル .csv'))
    df_retail['売上日'] = pd.to_datetime(df_retail['売上日'])
    df_retail['品名CD'] = df_retail['品名CD'].str.strip()
    df_retail = df_retail[df_retail['売上日'] <= '2025-03-18'].copy()
    products_retail = sorted(df_retail['品名CD'].unique())
    all_dates_retail = pd.date_range(
        df_retail['売上日'].min(), df_retail['売上日'].max(), freq='D')
    retail_matrix = np.zeros((len(products_retail), len(all_dates_retail)))
    date_to_idx_r = {d: i for i, d in enumerate(all_dates_retail)}
    for _, row in df_retail.iterrows():
        pi = products_retail.index(row['品名CD'])
        di = date_to_idx_r.get(row['売上日'])
        if di is not None:
            retail_matrix[pi, di] = row['数量']
    print(f"  Retail: {retail_matrix.shape[0]} products × {retail_matrix.shape[1]} days")

    # --- Nishimatsu ---
    print("Loading nishimatsu data...")
    df_nishi = pd.read_csv(os.path.join(KISS_DIR, 'nishimatsu_until_2024_07_weekly.csv'))
    df_nishi['週_開始日'] = pd.to_datetime(df_nishi['週_開始日'])
    # Use full date range, aggregate to ISO weeks
    df_nishi['iso_week'] = df_nishi['週_開始日'].dt.isocalendar().week.astype(int)
    df_nishi['iso_year'] = df_nishi['週_開始日'].dt.isocalendar().year.astype(int)
    df_nishi['year_week'] = df_nishi['iso_year'] * 100 + df_nishi['iso_week']
    weekly_agg = df_nishi.groupby(
        ['商品名_商品CD', 'year_week'])['週次_売上個数'].sum().reset_index()
    all_yw = sorted(weekly_agg['year_week'].unique())
    n_weeks_total = len(all_yw)
    yw_to_idx = {yw: i for i, yw in enumerate(all_yw)}
    # Select products with >= 50% week coverage
    product_coverage = weekly_agg.groupby('商品名_商品CD')['year_week'].nunique()
    min_coverage = int(n_weeks_total * 0.5)
    good_products = product_coverage[product_coverage >= min_coverage].index.tolist()
    if len(good_products) > 500:
        vol = weekly_agg[weekly_agg['商品名_商品CD'].isin(good_products)].groupby(
            '商品名_商品CD')['週次_売上個数'].sum()
        good_products = vol.nlargest(500).index.tolist()
    nishi_matrix = np.zeros((len(good_products), n_weeks_total))
    product_to_idx = {p: i for i, p in enumerate(good_products)}
    for _, row in weekly_agg[weekly_agg['商品名_商品CD'].isin(good_products)].iterrows():
        pi = product_to_idx[row['商品名_商品CD']]
        wi = yw_to_idx[row['year_week']]
        nishi_matrix[pi, wi] = row['週次_売上個数']
    print(f"  Nishimatsu: {nishi_matrix.shape[0]} products × {nishi_matrix.shape[1]} weeks")

    # ================================================================
    # Multi-horizon evaluation
    # ================================================================
    # Dataset configs: (name, matrix, period, horizons, n_clusters)
    datasets = [
        ("レストラン (8店舗)", rest_matrix, 7, [14, 28, 56, 84], 4),
        ("小売出荷 (7商品)", retail_matrix, 7, [14, 28, 56, 84], 3),
        ("西松屋 (500商品)", nishi_matrix, 4, [4, 8, 12, 16], min(40, len(good_products) // 3)),
    ]

    # results_by_ds[ds_name][horizon] = {method: rmsse}
    results_by_ds = {}

    for ds_name, full_matrix, period, horizons, n_cl in datasets:
        results_by_ds[ds_name] = {}
        total_T = full_matrix.shape[1]

        for h in horizons:
            if total_T <= h + period * 4:
                print(f"\n  SKIP {ds_name} h={h}: not enough data (T={total_T})")
                continue
            train = full_matrix[:, :-h]
            test = full_matrix[:, -h:]
            res = evaluate_dataset(
                f'{ds_name} h={h}', train, test, h,
                period=period, n_clusters=n_cl)
            results_by_ds[ds_name][h] = res

    # ================================================================
    # Visualization: multi-horizon line charts
    # ================================================================
    print("\n" + "=" * 60)
    print("Generating multi-horizon visualizations...")
    print("=" * 60)

    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 6))
    if n_ds == 1:
        axes = [axes]

    for ax, (ds_name, _, period, horizons, _) in zip(axes, datasets):
        ds_results = results_by_ds.get(ds_name, {})
        if not ds_results:
            continue

        valid_horizons = sorted(ds_results.keys())
        all_methods = set()
        for h_res in ds_results.values():
            all_methods.update(h_res.keys())
        all_methods = sorted(all_methods)

        for method in all_methods:
            vals = []
            hs = []
            for h in valid_horizons:
                if method in ds_results[h]:
                    vals.append(ds_results[h][method])
                    hs.append(h)
            if vals:
                color = color_map.get(method, '#555555')
                lw = 2.5 if method in ('Theta', 'WP-MO-Theta') else 1.5
                marker = 'o' if method in ('Theta', 'WP-MO-Theta') else 's'
                ax.plot(hs, vals, marker=marker, linewidth=lw, label=method,
                        color=color, markersize=6)

        unit = '週' if period == 4 else '日'
        ax.set_xlabel(f'予測ホライズン ({unit})', fontsize=11)
        ax.set_ylabel('RMSSE', fontsize=11)
        ax.set_title(ds_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xticks(valid_horizons)

    plt.suptitle('予測ホライズン別 RMSSE比較', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kiss_multi_horizon.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: kiss_multi_horizon.png")

    # --- Bar chart: longest horizon only ---
    fig, axes2 = plt.subplots(1, n_ds, figsize=(7 * n_ds, 7))
    if n_ds == 1:
        axes2 = [axes2]

    for ax, (ds_name, _, _, horizons, _) in zip(axes2, datasets):
        ds_results = results_by_ds.get(ds_name, {})
        if not ds_results:
            continue
        longest_h = max(ds_results.keys())
        res = ds_results[longest_h]
        sorted_methods = sorted(res.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_methods]
        vals = [x[1] for x in sorted_methods]
        bar_colors = [color_map.get(n, '#555555') for n in names]
        bars = ax.barh(range(len(names)), vals, color=bar_colors,
                       edgecolor='black', alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('RMSSE (lower is better)')
        ax.set_title(f'{ds_name} (h={longest_h})', fontsize=12, fontweight='bold')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kiss_benchmark_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: kiss_benchmark_comparison.png")

    # ================================================================
    # Report
    # ================================================================
    print("\nGenerating report...")
    lines = []
    lines.append("# KISSデータセット 多ホライズン予測ベンチマーク\n")
    lines.append(f"**評価日**: 2026-03-18")
    lines.append(f"**目的**: 短期〜長期予測で手法の優劣がどう変化するかを検証\n")
    lines.append("---\n")

    lines.append("## 1. 実験条件\n")
    lines.append("| データセット | 系列数 | 粒度 | 訓練長 | テスト期間 | 季節周期 |")
    lines.append("| --- | ---: | --- | ---: | --- | ---: |")
    for ds_name, full_mat, period, horizons, _ in datasets:
        T = full_mat.shape[1]
        N = full_mat.shape[0]
        unit = '週' if period == 4 else '日'
        h_str = ', '.join(f'{h}{unit}' for h in horizons)
        lines.append(f"| {ds_name} | {N} | {'週次' if period == 4 else '日次'} "
                     f"| {T}{unit} | {h_str} | {period} |")
    lines.append("")

    lines.append("---\n")
    lines.append("## 2. ホライズン別結果\n")

    for ds_name, _, period, _, _ in datasets:
        ds_results = results_by_ds.get(ds_name, {})
        if not ds_results:
            continue
        lines.append(f"### {ds_name}\n")
        valid_horizons = sorted(ds_results.keys())
        all_methods = set()
        for h_res in ds_results.values():
            all_methods.update(h_res.keys())
        all_methods = sorted(all_methods)

        unit = '週' if period == 4 else '日'
        header = "| 手法 | " + " | ".join(f"h={h}{unit}" for h in valid_horizons) + " |"
        sep = "| --- | " + " | ".join(["---:"] * len(valid_horizons)) + " |"
        lines.append(header)
        lines.append(sep)
        for method in all_methods:
            vals = []
            for h in valid_horizons:
                v = ds_results[h].get(method, float('nan'))
                # Bold the best per horizon
                all_vals_h = [ds_results[h].get(m, float('inf')) for m in all_methods]
                is_best = (v == min(all_vals_h))
                cell = f"**{v:.4f}**" if is_best else f"{v:.4f}"
                vals.append(cell)
            lines.append(f"| {method} | " + " | ".join(vals) + " |")
        lines.append("")

    lines.append("![ホライズン別比較](./kiss_multi_horizon.png)\n")
    lines.append("![最長ホライズン比較](./kiss_benchmark_comparison.png)\n")

    lines.append("---\n")
    lines.append("## 3. 考察\n")

    # Summarize: which method wins at each horizon
    lines.append("### ホライズン別ベスト手法\n")
    for ds_name, _, period, _, _ in datasets:
        ds_results = results_by_ds.get(ds_name, {})
        if not ds_results:
            continue
        lines.append(f"**{ds_name}**:\n")
        unit = '週' if period == 4 else '日'
        for h in sorted(ds_results.keys()):
            res = ds_results[h]
            best = min(res, key=res.get)
            lines.append(f"- h={h}{unit}: {best} ({res[best]:.4f})")
        lines.append("")

    # Cross-dataset summary
    lines.append("### 短期 vs 長期の傾向\n")
    lines.append("- 短期(h=14日/4週): シンプルな手法(Theta/Naive)が有利な傾向")
    lines.append("- 長期(h=84日/16週): 季節性モデル(HW/MSTL+ETS)やクラスタリング手法が相対的に改善するか検証")
    lines.append("- WP-MOクラスタリングの効果は系列数に大きく依存\n")

    lines.append("---\n")
    total_elapsed = time.perf_counter() - total_start
    lines.append(f"## 4. 実行時間\n")
    lines.append(f"- 総実行時間: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)\n")

    report_path = os.path.join(OUTPUT_DIR, 'kiss_benchmark_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {report_path}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*60}")
    print("Done!")
