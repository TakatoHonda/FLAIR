#!/usr/bin/env python3
"""Long-term Time Series Forecasting Benchmark — Paper-quality evaluation.

Exact protocol from PatchTST/iTransformer/DLinear papers:
  - Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic, Exchange
  - Horizons: 96, 192, 336, 720
  - Metrics: MSE, MAE (on StandardScaler-normalized data)
  - Channel-independent: each variable forecast univariately
  - Sliding window stride=1 over test set

Usage:
    uv run python -u research/benchmarks/run_longterm_benchmark.py
    uv run python -u research/benchmarks/run_longterm_benchmark.py --datasets ETTh1 ETTh2
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gift_eval_flair_ds import flair_ds
from run_gift_eval_flar9 import get_period

warnings.filterwarnings('ignore')

DATA_DIR = "/tmp/lh_data"
HORIZONS = [96, 192, 336, 720]

# Standard splits (from Time-Series-Library data_loader.py)
DATASETS = {
    "ETTh1":    {"freq": "H",   "split": (8640, 11520, 14400)},
    "ETTh2":    {"freq": "H",   "split": (8640, 11520, 14400)},
    "ETTm1":    {"freq": "15T", "split": (34560, 46080, 57600)},
    "ETTm2":    {"freq": "15T", "split": (34560, 46080, 57600)},
    "Weather":  {"freq": "10T", "split": "ratio"},
    "ECL":      {"freq": "H",   "split": "ratio"},
    "Traffic":  {"freq": "H",   "split": "ratio"},
    "Exchange": {"freq": "D",   "split": "ratio"},
}


ETT_CSV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "data", "long_term_forecast")


def load_all_data():
    """Load all datasets, return dict of {name: (array, train_end, test_start)}."""
    from datasetsforecast.long_horizon import LongHorizon

    loaded = {}
    for ds_name, cfg in DATASETS.items():
        print(f"  Loading {ds_name}...", end=" ", flush=True)
        try:
            # ETT: load from raw CSV to get all 7 variables
            if ds_name.startswith("ETT"):
                csv_path = os.path.join(ETT_CSV_DIR, f"{ds_name}.csv")
                df = pd.read_csv(csv_path)
                # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
                value_cols = [c for c in df.columns if c != "date"]
                data = df[value_cols].values.T.astype(float)  # (7, T)
            elif ds_name == "Traffic":
                csv_path = os.path.join(DATA_DIR, "longhorizon/datasets/traffic/M/df_y.csv")
                Y = pd.read_csv(csv_path)
                uids = Y["unique_id"].unique()
                series_list = [Y[Y["unique_id"] == uid]["y"].values.astype(float) for uid in uids]
                min_len = min(len(s) for s in series_list)
                data = np.array([s[:min_len] for s in series_list])
            else:
                Y, _, _ = LongHorizon.load(DATA_DIR, group=ds_name)
                uids = Y["unique_id"].unique()
                series_list = [Y[Y["unique_id"] == uid]["y"].values.astype(float) for uid in uids]
                min_len = min(len(s) for s in series_list)
                data = np.array([s[:min_len] for s in series_list])

            # Compute split points
            T = data.shape[1]
            if cfg["split"] == "ratio":
                train_end = int(T * 0.7)
                val_end = T - int(T * 0.2)
            else:
                train_end, val_end, _ = cfg["split"]

            loaded[ds_name] = {
                "data": data,
                "train_end": train_end,
                "test_start": val_end,
                "freq": cfg["freq"],
                "n_vars": data.shape[0],
                "T": T,
            }
            print(f"vars={data.shape[0]}, T={T}, test=[{val_end}:{T}]")

        except Exception as e:
            print(f"SKIP ({e})")

    return loaded


def evaluate_dataset(ds_info, ds_name, horizons, max_vars=None):
    """Evaluate FLAIR on one dataset, all horizons."""
    data = ds_info["data"]       # (n_vars, T)
    train_end = ds_info["train_end"]
    test_start = ds_info["test_start"]
    freq = ds_info["freq"]
    n_vars, T = data.shape

    # Normalize: fit on training data
    train_data = data[:, :train_end]
    mean = train_data.mean(axis=1, keepdims=True)  # (n_vars, 1)
    std = train_data.std(axis=1, keepdims=True)     # (n_vars, 1)
    std = np.where(std < 1e-8, 1.0, std)
    data_norm = (data - mean) / std

    # Limit variables for large datasets
    if max_vars and n_vars > max_vars:
        var_idx = np.linspace(0, n_vars - 1, max_vars, dtype=int)
    else:
        var_idx = np.arange(n_vars)

    period = get_period(freq)
    results = []

    for H in horizons:
        t0 = time.perf_counter()
        n_windows = T - test_start - H + 1
        if n_windows <= 0:
            print(f"  H={H:>3d}  SKIP (test too short)")
            continue

        # For large test sets, use strided windows (every H steps) for speed
        # but ensure minimum 30 windows for stability
        if n_windows > 500:
            stride = max(1, n_windows // 200)
        else:
            stride = 1
        window_starts = list(range(test_start, test_start + n_windows, stride))

        all_mse, all_mae = [], []

        for vi in var_idx:
            y_norm = data_norm[vi]
            var_mse, var_mae = [], []

            for ws in window_starts:
                context = y_norm[:ws]
                actual = y_norm[ws:ws + H]

                samples = flair_ds(context, H, period, freq, n_samples=20)
                pred = np.median(samples, axis=0)

                var_mse.append(float(np.mean((pred - actual) ** 2)))
                var_mae.append(float(np.mean(np.abs(pred - actual))))

            all_mse.append(np.mean(var_mse))
            all_mae.append(np.mean(var_mae))

        elapsed = time.perf_counter() - t0
        avg_mse = float(np.mean(all_mse))
        avg_mae = float(np.mean(all_mae))

        results.append({
            "dataset": ds_name,
            "horizon": H,
            "mse": round(avg_mse, 4),
            "mae": round(avg_mae, 4),
            "n_vars": len(var_idx),
            "n_windows": len(window_starts),
            "stride": stride,
            "time_s": round(elapsed, 1),
        })
        print(f"  H={H:>3d}  MSE={avg_mse:.4f}  MAE={avg_mae:.4f}  "
              f"({len(var_idx)} vars, {len(window_starts)} wins, {elapsed:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    total_t0 = time.perf_counter()
    print("Loading datasets...")
    all_data = load_all_data()

    ds_names = args.datasets if args.datasets else list(DATASETS.keys())

    all_results = []
    for ds_name in ds_names:
        if ds_name not in all_data:
            print(f"\n  {ds_name}: not loaded, skipping")
            continue
        info = all_data[ds_name]
        print(f"\n{'='*60}")
        print(f"  {ds_name}: {info['n_vars']} vars, T={info['T']}, "
              f"test=[{info['test_start']}:{info['T']}]")
        print(f"{'='*60}")

        # Limit vars for very large datasets
        max_vars = 50 if info["n_vars"] > 100 else None
        results = evaluate_dataset(info, ds_name, HORIZONS, max_vars=max_vars)
        all_results.extend(results)

    # Save
    df = pd.DataFrame(all_results)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "results", "19_longterm_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "flair_longterm_results.csv")
    df.to_csv(csv_path, index=False)

    total_elapsed = time.perf_counter() - total_t0

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"FLAIR Long-term Forecasting Benchmark")
    print(f"{'='*70}")
    print(f"{'Dataset':<12s} {'H':>4s} {'MSE':>8s} {'MAE':>8s}  "
          f"{'iTransformer':>13s} {'PatchTST':>10s} {'DLinear':>9s}")
    print(f"{'-'*70}")

    # Baseline numbers from iTransformer paper (ICLR 2024)
    baselines = {
        ("ETTh1", 96):  (0.386, 0.414, 0.386),
        ("ETTh1", 192): (0.441, 0.460, 0.437),
        ("ETTh1", 336): (0.487, 0.501, 0.481),
        ("ETTh1", 720): (0.503, 0.500, 0.519),
        ("ETTh2", 96):  (0.297, 0.302, 0.333),
        ("ETTh2", 192): (0.380, 0.388, 0.477),
        ("ETTh2", 336): (0.428, 0.426, 0.594),
        ("ETTh2", 720): (0.427, 0.431, 0.831),
        ("ETTm1", 96):  (0.334, 0.329, 0.345),
        ("ETTm1", 192): (0.377, 0.367, 0.380),
        ("ETTm1", 336): (0.426, 0.399, 0.413),
        ("ETTm1", 720): (0.491, 0.454, 0.474),
        ("ETTm2", 96):  (0.180, 0.175, 0.193),
        ("ETTm2", 192): (0.250, 0.241, 0.284),
        ("ETTm2", 336): (0.311, 0.305, 0.369),
        ("ETTm2", 720): (0.412, 0.402, 0.554),
        ("Weather", 96):  (0.174, 0.177, 0.196),
        ("Weather", 192): (0.221, 0.225, 0.237),
        ("Weather", 336): (0.278, 0.278, 0.283),
        ("Weather", 720): (0.358, 0.354, 0.345),
        ("ECL", 96):  (0.148, 0.181, 0.197),
        ("ECL", 192): (0.162, 0.188, 0.196),
        ("ECL", 336): (0.178, 0.204, 0.209),
        ("ECL", 720): (0.225, 0.246, 0.245),
        ("Traffic", 96):  (0.395, 0.462, 0.650),
        ("Traffic", 192): (0.417, 0.466, 0.598),
        ("Traffic", 336): (0.433, 0.482, 0.605),
        ("Traffic", 720): (0.467, 0.514, 0.645),
        ("Exchange", 96):  (0.086, 0.088, 0.088),
        ("Exchange", 192): (0.177, 0.176, 0.176),
        ("Exchange", 336): (0.331, 0.301, 0.313),
        ("Exchange", 720): (0.847, 0.901, 0.839),
    }

    for _, r in df.iterrows():
        key = (r["dataset"], r["horizon"])
        if key in baselines:
            it, pt, dl = baselines[key]
            print(f"{r['dataset']:<12s} {r['horizon']:>4.0f} {r['mse']:>8.4f} {r['mae']:>8.4f}  "
                  f"{it:>13.3f} {pt:>10.3f} {dl:>9.3f}")
        else:
            print(f"{r['dataset']:<12s} {r['horizon']:>4.0f} {r['mse']:>8.4f} {r['mae']:>8.4f}")

    print(f"\nSaved: {csv_path}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
