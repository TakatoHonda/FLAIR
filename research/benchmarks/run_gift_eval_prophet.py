#!/usr/bin/env python3
"""Prophet on GIFT-EVAL: Full 97 config benchmark.

Usage:
    python -u research/benchmarks/run_gift_eval_prophet.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import logging

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
from prophet import Prophet

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet.forecaster').setLevel(logging.ERROR)
logging.disable(logging.INFO)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'results', '15_gift_eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "Prophet"
N_SAMPLES = 20

FREQ_TO_PERIOD = {
    'S': 60, 'T': 60, '5T': 12, '10T': 6, '15T': 4, '10S': 6,
    'H': 24, 'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'A': 1, 'Y': 1,
}

# Prophet freq strings
FREQ_TO_PROPHET_FREQ = {
    '10S': '10s', 'S': 's', '5T': '5min', '10T': '10min', '15T': '15min',
    'H': 'h', 'D': 'D', 'W': 'W', 'M': 'MS', 'Q': 'QS', 'A': 'YS', 'Y': 'YS',
}

def get_period(f):
    f = f.upper().replace('MIN', 'T')
    if f in FREQ_TO_PERIOD: return FREQ_TO_PERIOD[f]
    for k in sorted(FREQ_TO_PERIOD, key=len, reverse=True):
        if f.endswith(k): return FREQ_TO_PERIOD[k]
    return 1

def get_prophet_freq(f):
    f = f.upper().replace('MIN', 'T')
    if f in FREQ_TO_PROPHET_FREQ: return FREQ_TO_PROPHET_FREQ[f]
    for k in sorted(FREQ_TO_PROPHET_FREQ, key=len, reverse=True):
        if f.endswith(k): return FREQ_TO_PROPHET_FREQ[k]
    return 'D'


def prophet_forecast(y_raw, horizon, period, freq_str, n_samples=20):
    """Prophet forecast with sample generation."""
    y = np.nan_to_num(np.asarray(y_raw, float), nan=0.0)
    y = np.maximum(y, 0.0)
    n = len(y)

    # Context limit (Prophet is slow on long series)
    max_ctx = max(period * 10, 500) if period >= 2 else 500
    max_ctx = min(max_ctx, 2000)  # hard cap for speed
    if n > max_ctx:
        y = y[-max_ctx:]
        n = len(y)

    pf = get_prophet_freq(freq_str)

    try:
        # Build dataframe
        dates = pd.date_range(start='2020-01-01', periods=n, freq=pf)
        df = pd.DataFrame({'ds': dates, 'y': y})

        # Fit Prophet (suppress output)
        m = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            uncertainty_samples=n_samples,
        )
        m.fit(df)

        # Predict
        future = m.make_future_dataframe(periods=horizon, freq=pf)
        forecast = m.predict(future)
        fc = forecast.tail(horizon)

        point_fc = np.maximum(fc['yhat'].values, 0.0)

        # Generate samples from prediction intervals
        yhat_lower = fc['yhat_lower'].values
        yhat_upper = fc['yhat_upper'].values
        sigma = np.maximum((yhat_upper - yhat_lower) / (2 * 1.28), 1e-6)

        samples = np.zeros((n_samples, horizon))
        for s in range(n_samples):
            samples[s] = np.maximum(0, point_fc + np.random.normal(0, sigma))

        rm = np.max(y[-max(horizon*2, 50):])
        if rm > 0:
            samples = np.clip(samples, 0, rm * 3)

    except Exception:
        # Fallback: seasonal naive
        if period >= 2:
            tail = y[-period:]
            point_fc = np.tile(tail, (horizon // period) + 1)[:horizon]
        else:
            point_fc = np.full(horizon, y[-1])
        sigma = max(np.std(np.diff(y[-min(50, n):])), 1e-6) if n > 1 else 1.0
        samples = np.array([np.maximum(0, point_fc + np.random.normal(0, sigma, horizon))
                           for _ in range(n_samples)])

    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)


class ProphetPredictor(RepresentablePredictor):
    def __init__(self, prediction_length, period, freq_str, n_samples=20):
        super().__init__(prediction_length=prediction_length)
        self.period = period
        self.freq_str = freq_str
        self.n_samples = n_samples

    def predict_item(self, item):
        target = item['target']
        if target.ndim == 2:
            nv, T = target.shape
            samples = np.zeros((self.n_samples, self.prediction_length, nv))
            for v in range(nv):
                samples[:,:,v] = prophet_forecast(
                    target[v], self.prediction_length, self.period,
                    self.freq_str, self.n_samples)
            start = item['start'] + T
        else:
            samples = prophet_forecast(
                target, self.prediction_length, self.period,
                self.freq_str, self.n_samples)
            start = item['start'] + len(target)
        return SampleForecast(samples=samples, start_date=start,
                              item_id=item.get('item_id', 'unknown'))


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

DOMAIN_MAP = {
    "bitbrains_fast_storage": "Web/CloudOps", "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps", "bizitobs_l2c": "Web/CloudOps",
    "bizitobs_service": "Web/CloudOps", "car_parts": "Sales",
    "covid_deaths": "Healthcare", "electricity": "Energy",
    "ett1": "Energy", "ett2": "Energy", "hierarchical_sales": "Sales",
    "hospital": "Healthcare", "jena_weather": "Nature",
    "kdd_cup_2018": "Nature", "loop_seattle": "Transport",
    "m4_daily": "Econ/Fin", "m4_hourly": "Econ/Fin",
    "m4_monthly": "Econ/Fin", "m4_quarterly": "Econ/Fin",
    "m4_weekly": "Econ/Fin", "m4_yearly": "Econ/Fin",
    "m_dense": "Transport", "restaurant": "Sales",
    "saugeen": "Nature", "solar": "Energy", "sz_taxi": "Transport",
    "temperature_rain": "Nature", "us_births": "Healthcare",
}

NAME_MAP = {
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
    "temperature_rain": "temperature_rain_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE", "sz_taxi": "SZ_TAXI", "saugeen": "saugeenday",
}


def evaluate_config(ds_name, freq, term):
    load_name = NAME_MAP.get(ds_name, ds_name)
    ds_path = os.path.join(os.environ['GIFT_EVAL'], load_name)
    if os.path.isdir(os.path.join(ds_path, freq)):
        load_name = f"{load_name}/{freq}"
    try:
        ds = Dataset(name=load_name, term=term, to_univariate=False)
    except Exception as e:
        print(f"SKIP: {e}"); return None, 0

    period = get_period(ds.freq)
    pred = ProphetPredictor(ds.prediction_length, period, ds.freq, N_SAMPLES)
    try:
        res_df = evaluate_model(
            pred, test_data=ds.test_data, metrics=METRICS,
            batch_size=5000, axis=None, mask_invalid_label=True,
            allow_nan_forecast=False, seasonality=None)
        row = {f"eval_metrics/{c}": round(float(res_df[c].iloc[0]), 6)
               for c in res_df.columns}
        row['dataset'] = f"{ds_name}/{freq}/{term}"
        row['model'] = MODEL_NAME
        row['domain'] = DOMAIN_MAP.get(ds_name, 'Unknown')
        first_item = next(iter(ds.test_data.input))
        target = first_item['target']
        row['num_variates'] = float(target.shape[0]) if target.ndim == 2 else 1.0
        return row, 0
    except Exception as e:
        print(f"EVAL ERROR: {e}"); return None, 0


if __name__ == '__main__':
    total_start = time.perf_counter()
    n_total = sum(len(terms) for _, _, terms in DATASET_CONFIGS)
    results = []
    timings = []
    gm = lambda v: np.exp(np.mean(np.log(np.clip(v, 1e-10, None))))

    print(f"{'='*70}")
    print(f"Prophet Full Benchmark: {n_total} configs")
    print(f"{'='*70}")

    done = 0
    for ds_name, freq, terms in DATASET_CONFIGS:
        for term in terms:
            done += 1
            cid = f"{ds_name}/{freq}/{term}"
            t0 = time.perf_counter()
            print(f"[{done:>3}/{n_total}] {cid}...", end=" ", flush=True)
            res, _ = evaluate_config(ds_name, freq, term)
            elapsed = time.perf_counter() - t0
            if res is not None:
                res['time_seconds'] = round(elapsed, 1)
                results.append(res)
                mase = res.get('eval_metrics/MASE[0.5]', float('nan'))
                crps = res.get('eval_metrics/mean_weighted_sum_quantile_loss', float('nan'))
                print(f"MASE={mase:.3f}  CRPS={crps:.3f}  ({elapsed:.1f}s)")
            else:
                print(f"SKIPPED ({elapsed:.1f}s)")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, 'all_results_prophet.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows)")

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    mase_vals = df['eval_metrics/MASE[0.5]'].dropna()
    crps_vals = df['eval_metrics/mean_weighted_sum_quantile_loss'].dropna()
    total_time = df['time_seconds'].sum()
    print(f"  Prophet:  MASE_gm={gm(mase_vals):.4f}  CRPS_gm={gm(crps_vals):.4f}  total_time={total_time:.0f}s")

    # Relative to Seasonal Naive
    sn_path = '/tmp/gift-eval/results/seasonal_naive/all_results.csv'
    if os.path.exists(sn_path):
        sn_df = pd.read_csv(sn_path)
        merged = df.merge(sn_df, on='dataset', suffixes=('_prophet', '_sn'), how='inner')
        if len(merged) > 0:
            rel_mase = merged['eval_metrics/MASE[0.5]_prophet'] / merged['eval_metrics/MASE[0.5]_sn']
            rel_crps = (merged['eval_metrics/mean_weighted_sum_quantile_loss_prophet'] /
                       merged['eval_metrics/mean_weighted_sum_quantile_loss_sn'])
            print(f"  Prophet:  relMASE={gm(rel_mase):.4f}  relCRPS={gm(rel_crps):.4f}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\nWall time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
