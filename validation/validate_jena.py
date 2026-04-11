"""Validate exog implementation on Jena Climate (10-min, 8 years).

We resample to hourly, then run rolling-origin evaluation:
- Target: T (degC) — air temperature
- Exog: p (pressure), rh (humidity), wv (wind velocity)

Pressure is a meteorologically meaningful leading indicator for temperature
(via barometric tendency).  Humidity and wind are correlated but not
deterministic — a partially-informative exog set.

Period: 24 (daily seasonality), forecast horizon: 48 hours.
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
    if naive_err < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / naive_err)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


def main() -> None:
    here = Path(__file__).parent
    print("Loading Jena Climate (10-min)...")
    df = pd.read_csv(here / "jena_climate_2009_2016.csv")
    df["dt"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("dt").sort_index()
    print(f"  raw rows: {len(df)}, span: {df.index[0]} → {df.index[-1]}")

    # Resample to hourly (mean over 6 ten-minute intervals)
    target_col = "T (degC)"
    exog_cols = ["p (mbar)", "rh (%)", "wv (m/s)"]
    cols = [target_col] + exog_cols
    hourly = df[cols].resample("1h").mean().dropna()
    print(f"  hourly rows: {len(hourly)}")

    # Use a single year (2014) for speed
    year = hourly[(hourly.index >= "2014-01-01") & (hourly.index < "2015-01-01")]
    print(f"  using 2014: {len(year)} hourly rows")

    y_all = year[target_col].astype(float).to_numpy()
    X_all = year[exog_cols].astype(float).to_numpy()
    # Replace any spurious -9999 sentinels with NaN (Jena occasionally has these)
    X_all = np.where(X_all < -100, np.nan, X_all)
    print(f"  exog cols: {exog_cols}, exog NaN frac: {np.isnan(X_all).mean():.4f}")

    # Rolling-origin evaluation
    train_len = 24 * 90  # 90 days history (longer for stable exog coefs)
    horizon = 48  # 48-hour forecast
    n_origins = 24
    n_total = len(y_all)
    origin_step = (n_total - train_len - horizon) // n_origins
    origins = [train_len + i * origin_step for i in range(n_origins)]

    print(f"\nRolling-origin: train_len={train_len}h, horizon={horizon}h, n_origins={n_origins}")

    rng_seed = 42
    n_samples = 200

    rows = []
    for origin in origins:
        y_hist = y_all[origin - train_len : origin]
        y_true = y_all[origin : origin + horizon]
        X_hist = X_all[origin - train_len : origin]
        X_future = X_all[origin : origin + horizon]

        s_no = forecast(y_hist, horizon, "H", n_samples=n_samples, seed=rng_seed)
        p_no = np.median(s_no, axis=0)

        s_ex = forecast(
            y_hist,
            horizon,
            "H",
            n_samples=n_samples,
            seed=rng_seed,
            X_hist=X_hist,
            X_future=X_future,
        )
        p_ex = np.median(s_ex, axis=0)

        rows.append(
            dict(
                origin=origin,
                date=str(year.index[origin].date()),
                mase_no=mase(y_true, p_no, y_hist, m=24),
                mase_ex=mase(y_true, p_ex, y_hist, m=24),
                mae_no=float(np.mean(np.abs(y_true - p_no))),
                mae_ex=float(np.mean(np.abs(y_true - p_ex))),
                smape_no=smape(y_true, p_no),
                smape_ex=smape(y_true, p_ex),
            )
        )

    res = pd.DataFrame(rows)
    print("\nPer-origin results (T degC, h=48):")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Summary (mean across origins) ===")
    print(f"  MASE  no_exog: {res['mase_no'].mean():.4f}   exog: {res['mase_ex'].mean():.4f}   "
          f"Δ%: {(res['mase_ex'].mean() / res['mase_no'].mean() - 1) * 100:+.2f}%")
    print(f"  MAE   no_exog: {res['mae_no'].mean():.4f}   exog: {res['mae_ex'].mean():.4f}   "
          f"Δ%: {(res['mae_ex'].mean() / res['mae_no'].mean() - 1) * 100:+.2f}%")
    print(f"  sMAPE no_exog: {res['smape_no'].mean():.2f}    exog: {res['smape_ex'].mean():.2f}    "
          f"Δ%: {(res['smape_ex'].mean() / res['smape_no'].mean() - 1) * 100:+.2f}%")
    wins = (res["mase_ex"] < res["mase_no"]).sum()
    print(f"\n  exog beats no_exog on {wins}/{len(res)} origins (MASE)")


if __name__ == "__main__":
    main()
