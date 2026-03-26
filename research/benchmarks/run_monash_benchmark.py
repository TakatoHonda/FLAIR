#!/usr/bin/env python3
"""FLAIR evaluation on Chronos benchmark datasets (Monash + others).

Reproduces the exact evaluation protocol from:
  Chronos: Learning the Language of Time Series (Ansari et al., 2024)

Protocol:
  - Each dataset has an `offset` (negative) that defines train/test split
  - prediction_length steps are held out as the test set
  - Metrics: MASE (median forecast) and WQL (weighted quantile loss)
  - Quantiles: [0.1, 0.2, ..., 0.9]

Usage:
    uv run python -u research/benchmarks/run_monash_benchmark.py
    uv run python -u research/benchmarks/run_monash_benchmark.py --subset zero-shot
    uv run python -u research/benchmarks/run_monash_benchmark.py --datasets monash_tourism_monthly monash_hospital
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flair_ds import flair_ds
from run_gift_eval_flar9 import get_period, get_periods

from datasets import load_dataset
from gluonts.dataset.split import split
from gluonts.model import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss

warnings.filterwarnings('ignore')

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_SAMPLES = 200

# =========================================================================
# Dataset configurations (from Chronos zero-shot.yaml and in-domain.yaml)
# =========================================================================

ZERO_SHOT_DATASETS = [
    {"name": "monash_traffic", "prediction_length": 24, "offset": -24},
    {"name": "monash_australian_electricity", "prediction_length": 48, "offset": -48},
    {"name": "ercot", "prediction_length": 24, "offset": -24},
    {"name": "exchange_rate", "prediction_length": 30, "offset": -30},
    {"name": "nn5", "prediction_length": 56, "offset": -56},
    {"name": "monash_nn5_weekly", "prediction_length": 8, "offset": -8},
    {"name": "monash_weather", "prediction_length": 30, "offset": -30},
    {"name": "monash_covid_deaths", "prediction_length": 30, "offset": -30},
    {"name": "monash_fred_md", "prediction_length": 12, "offset": -12},
    {"name": "m4_quarterly", "prediction_length": 8, "offset": -8},
    {"name": "m4_yearly", "prediction_length": 6, "offset": -6},
    {"name": "dominick", "prediction_length": 8, "offset": -8},
    {"name": "m5", "prediction_length": 28, "offset": -28},
    {"name": "monash_tourism_monthly", "prediction_length": 24, "offset": -24},
    {"name": "monash_tourism_quarterly", "prediction_length": 8, "offset": -8},
    {"name": "monash_tourism_yearly", "prediction_length": 4, "offset": -4},
    {"name": "monash_car_parts", "prediction_length": 12, "offset": -12},
    {"name": "monash_hospital", "prediction_length": 12, "offset": -12},
    {"name": "monash_cif_2016", "prediction_length": 12, "offset": -12},
    {"name": "monash_m1_yearly", "prediction_length": 6, "offset": -6},
    {"name": "monash_m1_quarterly", "prediction_length": 8, "offset": -8},
    {"name": "monash_m1_monthly", "prediction_length": 18, "offset": -18},
    {"name": "monash_m3_monthly", "prediction_length": 18, "offset": -18},
    {"name": "monash_m3_yearly", "prediction_length": 6, "offset": -6},
    {"name": "monash_m3_quarterly", "prediction_length": 8, "offset": -8},
]

IN_DOMAIN_DATASETS = [
    {"name": "electricity_15min", "prediction_length": 24, "offset": -24},
    {"name": "monash_electricity_hourly", "prediction_length": 24, "offset": -24},
    {"name": "monash_electricity_weekly", "prediction_length": 8, "offset": -8},
    {"name": "monash_kdd_cup_2018", "prediction_length": 48, "offset": -48},
    {"name": "m4_daily", "prediction_length": 14, "offset": -14},
    {"name": "m4_hourly", "prediction_length": 48, "offset": -48},
    {"name": "m4_monthly", "prediction_length": 18, "offset": -18},
    {"name": "m4_weekly", "prediction_length": 13, "offset": -13},
    {"name": "monash_pedestrian_counts", "prediction_length": 48, "offset": -48},
    {"name": "taxi_30min", "prediction_length": 48, "offset": -48},
    {"name": "uber_tlc_hourly", "prediction_length": 24, "offset": -24},
    {"name": "uber_tlc_daily", "prediction_length": 7, "offset": -7},
    {"name": "monash_rideshare", "prediction_length": 24, "offset": -24},
    {"name": "monash_temperature_rain", "prediction_length": 30, "offset": -30},
    {"name": "monash_london_smart_meters", "prediction_length": 48, "offset": -48},
]

# Freq inference from dataset name
DATASET_FREQ = {
    "monash_traffic": "H", "monash_australian_electricity": "30T",
    "ercot": "H", "exchange_rate": "D",
    "nn5": "D", "monash_nn5_weekly": "W",
    "monash_weather": "D", "monash_covid_deaths": "D",
    "monash_fred_md": "M", "m4_quarterly": "Q", "m4_yearly": "A",
    "dominick": "W", "m5": "D",
    "monash_tourism_monthly": "M", "monash_tourism_quarterly": "Q",
    "monash_tourism_yearly": "A",
    "monash_car_parts": "M", "monash_hospital": "M",
    "monash_cif_2016": "M",
    "monash_m1_yearly": "A", "monash_m1_quarterly": "Q",
    "monash_m1_monthly": "M",
    "monash_m3_monthly": "M", "monash_m3_yearly": "A",
    "monash_m3_quarterly": "Q",
    # In-domain
    "electricity_15min": "15T", "monash_electricity_hourly": "H",
    "monash_electricity_weekly": "W", "monash_kdd_cup_2018": "H",
    "m4_daily": "D", "m4_hourly": "H", "m4_monthly": "M", "m4_weekly": "W",
    "monash_pedestrian_counts": "H", "taxi_30min": "30T",
    "uber_tlc_hourly": "H", "uber_tlc_daily": "D",
    "monash_rideshare": "H", "monash_temperature_rain": "D",
    "monash_london_smart_meters": "30T",
}

# Chronos paper results for comparison (Table 2, zero-shot, Chronos-Small T5)
# WQL and MASE from the paper
CHRONOS_SMALL_RESULTS = {
    # dataset: (WQL, MASE) — from Chronos paper Table 2
    "monash_traffic": (0.116, None),
    "monash_australian_electricity": (0.069, None),
    "exchange_rate": (0.012, None),
    "nn5": (0.069, None),
    "monash_nn5_weekly": (0.044, None),
    "monash_weather": (0.232, None),
    "monash_covid_deaths": (0.137, None),
    "monash_fred_md": (0.072, None),
    "m4_quarterly": (0.087, None),
    "m4_yearly": (0.120, None),
    "monash_tourism_monthly": (0.107, None),
    "monash_tourism_quarterly": (0.083, None),
    "monash_tourism_yearly": (0.108, None),
    "monash_car_parts": (0.478, None),
    "monash_hospital": (0.065, None),
    "monash_cif_2016": (0.042, None),
    "monash_m1_yearly": (0.126, None),
    "monash_m1_quarterly": (0.085, None),
    "monash_m1_monthly": (0.070, None),
    "monash_m3_monthly": (0.081, None),
    "monash_m3_yearly": (0.118, None),
    "monash_m3_quarterly": (0.077, None),
}


def infer_freq(timestamps):
    """Infer frequency from timestamp list."""
    if not timestamps or len(timestamps) < 2:
        return "D"
    ts = pd.to_datetime(timestamps[:min(20, len(timestamps))])
    freq = pd.infer_freq(ts)
    if freq:
        return freq
    # Fallback: compute median diff
    diffs = np.diff(ts.values).astype('timedelta64[s]').astype(float)
    median_s = np.median(diffs)
    if median_s < 120:
        return "T"
    elif median_s < 7200:
        return "H"
    elif median_s < 172800:
        return "D"
    elif median_s < 1209600:
        return "W"
    elif median_s < 5184000:
        return "M"
    elif median_s < 15552000:
        return "Q"
    return "A"


def load_chronos_dataset(name):
    """Load dataset from autogluon/chronos_datasets HuggingFace hub."""
    hf_repo = "autogluon/chronos_datasets"
    try:
        ds = load_dataset(hf_repo, name, trust_remote_code=True, split="train")
    except Exception:
        # Try extra repo
        ds = load_dataset("autogluon/chronos_datasets_extra", name,
                          trust_remote_code=True, split="train")

    # Infer freq from first series or use known mapping
    freq = DATASET_FREQ.get(name)
    if not freq:
        freq = infer_freq(ds[0].get("timestamp"))

    # Convert to list of dicts for GluonTS
    gts_entries = []
    for row in ds:
        target = np.array(row["target"], dtype=np.float64)
        if row.get("timestamp") and len(row["timestamp"]) > 0:
            start = pd.Timestamp(row["timestamp"][0])
        else:
            start = pd.Timestamp("2000-01-01")
        gts_entries.append({
            "start": start,
            "target": target,
            "item_id": str(row.get("id", "")),
        })

    return gts_entries, freq


def flair_forecast(train_target, prediction_length, freq_str, item_id=""):
    """Generate FLAIR forecast samples for a single series."""
    import hashlib
    sid = int(hashlib.md5(str(item_id).encode()).hexdigest()[:8], 16)
    np.random.seed(sid)
    period = get_period(freq_str)
    samples = flair_ds(train_target, prediction_length, period, freq_str,
                       n_samples=N_SAMPLES)
    return samples


def evaluate_dataset(config):
    """Evaluate FLAIR on a single dataset."""
    name = config["name"]
    prediction_length = config["prediction_length"]
    offset = config["offset"]

    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  prediction_length={prediction_length}, offset={offset}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    # Load data
    try:
        entries, freq = load_chronos_dataset(name)
    except Exception as e:
        print(f"  SKIP (load failed): {e}")
        return None

    n_series = len(entries)
    print(f"  Loaded {n_series} series, freq={freq}")

    # Split: use offset to separate train/test
    # offset is negative, e.g. -24 means last 24 steps are test
    forecasts = []
    test_targets = []
    train_targets = []

    for entry in entries:
        target = entry["target"]
        if len(target) < abs(offset) + max(prediction_length, 3):
            # Skip series too short
            continue
        train = target[:offset]  # offset is negative
        test = target[offset:][:prediction_length]
        if len(test) < prediction_length:
            continue
        # Forward-fill NaN in training data
        train_mask = np.isnan(train)
        if train_mask.any() and not train_mask.all():
            for j in range(1, len(train)):
                if train_mask[j] and not train_mask[j-1]:
                    train[j] = train[j-1]
                    train_mask[j] = False
        train_targets.append(train)
        test_targets.append(test)

        # Generate forecast
        samples = flair_forecast(train, prediction_length, freq,
                                 item_id=entry.get("item_id", str(len(forecasts))))
        start_date = entry["start"] + len(train) * pd.tseries.frequencies.to_offset(
            pd.tseries.frequencies.to_offset(freq) or pd.DateOffset(days=1))
        forecasts.append({
            "samples": samples,
            "target": test,
            "train": train,
        })

    if not forecasts:
        print(f"  SKIP (no valid series)")
        return None

    n_eval = len(forecasts)
    print(f"  Evaluating {n_eval} / {n_series} series...")

    # Compute MASE and WQL manually (to avoid GluonTS version issues)
    # WQL: dataset-level aggregate (sum all QL / sum all |target|) per quantile, then avg
    # MASE: mean across series
    mase_values = []
    # For dataset-level WQL
    total_ql_per_q = {q: 0.0 for q in QUANTILES}
    total_abs_target = 0.0

    period = get_period(freq)
    if period < 2:
        period = 1

    for fc in forecasts:
        samples = fc["samples"]  # (N_SAMPLES, H)
        target = fc["target"]    # (H,)
        train = fc["train"]

        # --- MASE ---
        median_fc = np.median(samples, axis=0)
        target_clean = np.nan_to_num(target, nan=0.0)
        mae = np.mean(np.abs(target_clean - median_fc))

        # Seasonal error (naive seasonal forecast) — handle NaN
        train_clean = np.nan_to_num(train, nan=0.0)
        n_train = len(train_clean)
        if n_train > period:
            naive_errors = np.abs(train_clean[period:] - train_clean[:-period])
            valid = (train_clean[period:] != 0) | (train_clean[:-period] != 0)
            seasonal_error = np.mean(naive_errors[valid]) if valid.sum() > 0 \
                else np.mean(naive_errors)
        else:
            seasonal_error = np.mean(np.abs(np.diff(train_clean))) \
                if n_train > 1 else 1.0

        if seasonal_error < 1e-10 or not np.isfinite(seasonal_error):
            seasonal_error = 1.0
        mase_val = mae / seasonal_error
        if np.isfinite(mase_val):
            mase_values.append(mase_val)

        # --- WQL accumulation (dataset-level, exclude NaN targets) ---
        valid_test = ~np.isnan(fc["target"])
        total_abs_target += np.sum(np.abs(target_clean[valid_test]))
        for q in QUANTILES:
            q_fc = np.quantile(samples, q, axis=0)
            ql = 2 * np.abs((target_clean - q_fc) * ((q_fc >= target_clean) - q))
            total_ql_per_q[q] += np.sum(ql[valid_test])

    elapsed = time.perf_counter() - t0
    mase_mean = float(np.mean(mase_values)) if mase_values else float('nan')

    # WQL: mean across quantiles of (sum_ql / sum_abs_target)
    if total_abs_target > 1e-10:
        wql_per_q = [total_ql_per_q[q] / total_abs_target for q in QUANTILES]
        wql_mean = float(np.mean(wql_per_q))
    else:
        wql_mean = float('nan')

    print(f"  MASE={mase_mean:.4f}, WQL={wql_mean:.4f} ({elapsed:.1f}s, {n_eval} series)")

    return {
        "dataset": name,
        "n_series": n_eval,
        "mase": mase_mean,
        "wql": wql_mean,
        "time_s": elapsed,
        "freq": freq,
        "prediction_length": prediction_length,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["zero-shot", "in-domain", "all"],
                        default="zero-shot")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to evaluate")
    args = parser.parse_args()

    if args.datasets:
        all_configs = ZERO_SHOT_DATASETS + IN_DOMAIN_DATASETS
        configs = [c for c in all_configs if c["name"] in args.datasets]
    elif args.subset == "zero-shot":
        configs = ZERO_SHOT_DATASETS
    elif args.subset == "in-domain":
        configs = IN_DOMAIN_DATASETS
    else:
        configs = ZERO_SHOT_DATASETS + IN_DOMAIN_DATASETS

    print(f"FLAIR Monash Benchmark — {len(configs)} datasets")
    print(f"N_SAMPLES={N_SAMPLES}")

    results = []
    total_t0 = time.perf_counter()

    for config in configs:
        try:
            result = evaluate_dataset(config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR on {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY — FLAIR-MDL on Chronos benchmark")
    print(f"{'='*70}")
    print(f"{'Dataset':40s} {'MASE':>8s} {'WQL':>8s} {'Series':>8s} {'Time':>6s}")
    print("-" * 70)
    for r in results:
        print(f"{r['dataset']:40s} {r['mase']:8.4f} {r['wql']:8.4f} "
              f"{r['n_series']:8d} {r['time_s']:6.1f}s")

    if results:
        agg_mase = np.mean([r["mase"] for r in results])
        agg_wql = np.mean([r["wql"] for r in results])
        gm_mase = np.exp(np.mean(np.log([max(r["mase"], 1e-10) for r in results])))
        gm_wql = np.exp(np.mean(np.log([max(r["wql"], 1e-10) for r in results])))
        print("-" * 70)
        print(f"{'Arithmetic Mean':40s} {agg_mase:8.4f} {agg_wql:8.4f}")
        print(f"{'Geometric Mean':40s} {gm_mase:8.4f} {gm_wql:8.4f}")

        # Compare with Chronos-Small where available
        print(f"\n{'='*70}")
        print(f"COMPARISON vs Chronos-Small (zero-shot)")
        print(f"{'='*70}")
        print(f"{'Dataset':40s} {'FLAIR WQL':>10s} {'Chronos':>10s} {'Delta':>8s}")
        print("-" * 70)
        wins, losses = 0, 0
        flair_wqls, chronos_wqls = [], []
        for r in results:
            cs = CHRONOS_SMALL_RESULTS.get(r["dataset"])
            if cs and cs[0] is not None:
                c_wql = cs[0]
                delta = r["wql"] - c_wql
                marker = "WIN" if delta < 0 else "LOSE"
                if delta < 0:
                    wins += 1
                else:
                    losses += 1
                flair_wqls.append(r["wql"])
                chronos_wqls.append(c_wql)
                print(f"{r['dataset']:40s} {r['wql']:10.4f} {c_wql:10.4f} "
                      f"{delta:+8.4f} {marker}")
        if flair_wqls:
            gm_f = np.exp(np.mean(np.log(flair_wqls)))
            gm_c = np.exp(np.mean(np.log(chronos_wqls)))
            print("-" * 70)
            print(f"{'Geometric Mean':40s} {gm_f:10.4f} {gm_c:10.4f} "
                  f"{gm_f - gm_c:+8.4f}")
            print(f"  FLAIR wins: {wins}, Chronos-Small wins: {losses}")

    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "results", "18_monash_benchmark")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "flair_monash_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(f"Total time: {time.perf_counter() - total_t0:.0f}s")


if __name__ == "__main__":
    main()
