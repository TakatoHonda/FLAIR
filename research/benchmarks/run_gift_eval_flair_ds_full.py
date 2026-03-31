#!/usr/bin/env python3
"""FLAIR-DS: Full GIFT-Eval benchmark (97 configs).

Dirichlet Shape with secondary-period context and K*C recency window.

Usage:
    uv run python -u research/benchmarks/run_gift_eval_flair_ds_full.py
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
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
    MeanWeightedSumQuantileLoss,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flair_ds import flair_ds
from run_gift_eval_flar9 import get_period

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'results', '15_gift_eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "FLAIR_MDL_coherent"
N_SAMPLES = 200

METRICS = [
    MSE(forecast_type="mean"), MSE(forecast_type=0.5),
    MAE(forecast_type=0.5), MASE(forecast_type=0.5),
    MAPE(forecast_type=0.5), SMAPE(forecast_type=0.5), MSIS(),
    RMSE(forecast_type="mean"), NRMSE(forecast_type="mean"),
    ND(forecast_type=0.5),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
]

DATASET_CONFIGS = [
    ("bitbrains_fast_storage", "5T", ["short", "medium", "long"]),
    ("bitbrains_fast_storage", "H", ["short"]),
    ("bitbrains_rnd", "5T", ["short", "medium", "long"]),
    ("bitbrains_rnd", "H", ["short"]),
    ("bizitobs_application", "10S", ["short", "medium", "long"]),
    ("bizitobs_l2c", "5T", ["short", "medium", "long"]),
    ("bizitobs_l2c", "H", ["short", "medium", "long"]),
    ("bizitobs_service", "10S", ["short", "medium", "long"]),
    ("car_parts", "M", ["short"]),
    ("covid_deaths", "D", ["short"]),
    ("electricity", "15T", ["short", "medium", "long"]),
    ("electricity", "D", ["short"]),
    ("electricity", "H", ["short", "medium", "long"]),
    ("electricity", "W", ["short"]),
    ("ett1", "15T", ["short", "medium", "long"]),
    ("ett1", "D", ["short"]),
    ("ett1", "H", ["short", "medium", "long"]),
    ("ett1", "W", ["short"]),
    ("ett2", "15T", ["short", "medium", "long"]),
    ("ett2", "D", ["short"]),
    ("ett2", "H", ["short", "medium", "long"]),
    ("ett2", "W", ["short"]),
    ("hierarchical_sales", "D", ["short"]),
    ("hierarchical_sales", "W", ["short"]),
    ("hospital", "M", ["short"]),
    ("jena_weather", "10T", ["short", "medium", "long"]),
    ("jena_weather", "D", ["short"]),
    ("jena_weather", "H", ["short", "medium", "long"]),
    ("kdd_cup_2018", "D", ["short"]),
    ("kdd_cup_2018", "H", ["short", "medium", "long"]),
    ("loop_seattle", "5T", ["short", "medium", "long"]),
    ("loop_seattle", "D", ["short"]),
    ("loop_seattle", "H", ["short", "medium", "long"]),
    ("m4_daily", "D", ["short"]),
    ("m4_hourly", "H", ["short"]),
    ("m4_monthly", "M", ["short"]),
    ("m4_quarterly", "Q", ["short"]),
    ("m4_weekly", "W", ["short"]),
    ("m4_yearly", "A", ["short"]),
    ("m_dense", "D", ["short"]),
    ("m_dense", "H", ["short", "medium", "long"]),
    ("restaurant", "D", ["short"]),
    ("saugeen", "D", ["short"]),
    ("saugeen", "M", ["short"]),
    ("saugeen", "W", ["short"]),
    ("solar", "10T", ["short", "medium", "long"]),
    ("solar", "D", ["short"]),
    ("solar", "H", ["short", "medium", "long"]),
    ("solar", "W", ["short"]),
    ("sz_taxi", "15T", ["short", "medium", "long"]),
    ("sz_taxi", "H", ["short"]),
    ("temperature_rain", "D", ["short"]),
    ("us_births", "D", ["short"]),
    ("us_births", "M", ["short"]),
    ("us_births", "W", ["short"]),
]

NAME_MAP = {
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
    "temperature_rain": "temperature_rain_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE", "sz_taxi": "SZ_TAXI", "saugeen": "saugeenday",
}


class DSPredictor(RepresentablePredictor):
    def __init__(self, prediction_length, period, freq_str, n_samples=200):
        super().__init__(prediction_length=prediction_length)
        self.period = period; self.freq_str = freq_str; self.n_samples = n_samples

    def predict_item(self, item):
        # Deterministic per-series seed (hash() is randomized per process;
        # use a stable hash instead for cross-run reproducibility)
        import hashlib
        sid = int(hashlib.md5(str(item.get('item_id', '')).encode()).hexdigest()[:8], 16)
        np.random.seed(sid)
        target = item['target']
        if target.ndim == 2:
            nv, T = target.shape
            samples = np.zeros((self.n_samples, self.prediction_length, nv))
            for v in range(nv):
                samples[:,:,v] = flair_ds(
                    target[v], self.prediction_length, self.period,
                    self.freq_str, self.n_samples)
            start = item['start'] + T
        else:
            samples = flair_ds(
                target, self.prediction_length, self.period,
                self.freq_str, self.n_samples)
            start = item['start'] + len(target)
        return SampleForecast(samples=samples, start_date=start,
                              item_id=item.get('item_id', 'unknown'))


def evaluate_config(ds_name, freq, term):
    load_name = NAME_MAP.get(ds_name, ds_name)
    ds_path = os.path.join(os.environ['GIFT_EVAL'], load_name)
    if os.path.isdir(os.path.join(ds_path, freq)):
        load_name = f"{load_name}/{freq}"
    try:
        ds = Dataset(name=load_name, term=term, to_univariate=False)
    except Exception as e:
        print(f"SKIP: {e}"); return None

    period = get_period(ds.freq)
    pred = DSPredictor(ds.prediction_length, period, ds.freq, N_SAMPLES)
    try:
        res_df = evaluate_model(
            pred, test_data=ds.test_data, metrics=METRICS,
            batch_size=5000, axis=None, mask_invalid_label=True,
            allow_nan_forecast=False, seasonality=None)
        row = {f"eval_metrics/{c}": round(float(res_df[c].iloc[0]), 6)
               for c in res_df.columns}
        row['dataset'] = f"{ds_name}/{freq}/{term}"
        row['model'] = MODEL_NAME
        return row
    except Exception as e:
        print(f"EVAL ERROR: {e}"); return None


if __name__ == '__main__':
    total_start = time.perf_counter()
    n_total = sum(len(terms) for _, _, terms in DATASET_CONFIGS)
    results = []
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    print(f"{'='*70}")
    print(f"FLAIR-DS Full Benchmark: {n_total} configs, N_SAMPLES={N_SAMPLES}")
    print(f"{'='*70}")

    done = 0
    for ds_name, freq, terms in DATASET_CONFIGS:
        for term in terms:
            done += 1
            cid = f"{ds_name}/{freq}/{term}"
            t0 = time.perf_counter()
            print(f"[{done:>3}/{n_total}] {cid}...", end=" ", flush=True)
            res = evaluate_config(ds_name, freq, term)
            elapsed = time.perf_counter() - t0
            if res is not None:
                results.append(res)
                mase = res.get('eval_metrics/MASE[0.5]', float('nan'))
                crps = res.get('eval_metrics/mean_weighted_sum_quantile_loss', float('nan'))
                print(f"MASE={mase:.3f}  CRPS={crps:.3f}  ({elapsed:.1f}s)")
            else:
                print(f"SKIPPED ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f'all_results_{MODEL_NAME}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # Relative to Seasonal Naive
    sn_path = '/tmp/gift-eval/results/seasonal_naive/all_results.csv'
    if os.path.exists(sn_path):
        sn_df = pd.read_csv(sn_path)
        merged = df.merge(sn_df, on='dataset', suffixes=('_ds', '_sn'), how='inner')
        if len(merged) > 0:
            rel_mase = merged['eval_metrics/MASE[0.5]_ds'] / merged['eval_metrics/MASE[0.5]_sn']
            rel_crps = (merged['eval_metrics/mean_weighted_sum_quantile_loss_ds'] /
                       merged['eval_metrics/mean_weighted_sum_quantile_loss_sn'])
            print(f"\n{'='*70}")
            print(f"FLAIR-DS: relMASE={gm(rel_mase):.4f}  relCRPS={gm(rel_crps):.4f}  ({len(merged)} configs)")

            # Compare with V2 (previous best)
            v2_path = os.path.join(OUTPUT_DIR, 'all_results_flair_v2.csv')
            if os.path.exists(v2_path):
                v2_df = pd.read_csv(v2_path)
                v2m = v2_df.merge(sn_df, on='dataset', suffixes=('_v2', '_sn'), how='inner')
                if len(v2m) > 0:
                    rv2 = gm(v2m['eval_metrics/MASE[0.5]_v2'] / v2m['eval_metrics/MASE[0.5]_sn'])
                    cv2 = gm(v2m['eval_metrics/mean_weighted_sum_quantile_loss_v2'] / v2m['eval_metrics/mean_weighted_sum_quantile_loss_sn'])
                    print(f"FLAIR-V2: relMASE={rv2:.4f}  relCRPS={cv2:.4f}  (previous best)")

            # Per-horizon
            for term in ['short', 'medium', 'long']:
                t_configs = [f"{ds}/{freq}/{term}" for ds, freq, terms in DATASET_CONFIGS if term in terms]
                tm = merged[merged['dataset'].isin(t_configs)]
                if len(tm) > 0:
                    rm = gm(tm['eval_metrics/MASE[0.5]_ds'] / tm['eval_metrics/MASE[0.5]_sn'])
                    print(f"  {term:8s}: relMASE={rm:.4f}  ({len(tm)} configs)")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nWall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
