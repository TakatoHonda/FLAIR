"""Robustness check: passing pure noise as exog must NOT degrade FLAIR.

Use the same Jena Climate setup that showed -15.5% improvement with real
exog, but replace the exog with random Gaussian noise of the same shape.
The naive Ridge approach should give a result close to vanilla FLAIR
(small drift, no catastrophic regression).

This is the "user passes a useless covariate by mistake" test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from flaircast import forecast  # noqa: E402


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 24) -> float:
    n = len(y_train)
    if n <= m:
        return float("nan")
    naive_err = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / max(naive_err, 1e-12))


def main() -> None:
    here = Path(__file__).parent
    df = pd.read_csv(here / "jena_climate_2009_2016.csv")
    df["dt"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("dt").sort_index()
    target_col = "T (degC)"
    hourly = df[[target_col]].resample("1h").mean().dropna()
    year = hourly[(hourly.index >= "2014-01-01") & (hourly.index < "2015-01-01")]
    y_all = year[target_col].astype(float).to_numpy()

    train_len = 24 * 90
    horizon = 48
    n_origins = 24
    n_total = len(y_all)
    origin_step = (n_total - train_len - horizon) // n_origins
    origins = [train_len + i * origin_step for i in range(n_origins)]

    print(f"Jena 2014, train_len={train_len}h, horizon={horizon}h, origins={n_origins}")
    print("Comparing: vanilla FLAIR  vs  FLAIR + random-noise exog (3 cols)\n")

    rng_seed = 42
    n_samples = 200
    noise_rng = np.random.RandomState(7)

    rows = []
    for origin in origins:
        y_hist = y_all[origin - train_len : origin]
        y_true = y_all[origin : origin + horizon]
        # Random noise exog of the same shape as the real one
        X_hist_noise = noise_rng.randn(train_len, 3)
        X_future_noise = noise_rng.randn(horizon, 3)

        s_no = forecast(y_hist, horizon, "H", n_samples=n_samples, seed=rng_seed)
        p_no = np.median(s_no, axis=0)
        s_n = forecast(
            y_hist,
            horizon,
            "H",
            n_samples=n_samples,
            seed=rng_seed,
            X_hist=X_hist_noise,
            X_future=X_future_noise,
        )
        p_n = np.median(s_n, axis=0)

        rows.append(
            dict(
                origin=origin,
                date=str(year.index[origin].date()),
                mase_no=mase(y_true, p_no, y_hist, m=24),
                mase_noise=mase(y_true, p_n, y_hist, m=24),
                mae_no=float(np.mean(np.abs(y_true - p_no))),
                mae_noise=float(np.mean(np.abs(y_true - p_n))),
            )
        )

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Summary: vanilla vs noise-exog ===")
    print(f"  MASE  vanilla: {res['mase_no'].mean():.4f}   noise: {res['mase_noise'].mean():.4f}   "
          f"Δ%: {(res['mase_noise'].mean() / res['mase_no'].mean() - 1) * 100:+.2f}%")
    print(f"  MAE   vanilla: {res['mae_no'].mean():.4f}    noise: {res['mae_noise'].mean():.4f}    "
          f"Δ%: {(res['mae_noise'].mean() / res['mae_no'].mean() - 1) * 100:+.2f}%")
    worst = (res["mase_noise"] / res["mase_no"] - 1).max()
    print(f"  Worst single-origin MASE inflation: {worst * 100:+.2f}%")
    print(f"\n  → Acceptable if mean Δ% is well within ±5% and worst case is bounded")


if __name__ == "__main__":
    main()
