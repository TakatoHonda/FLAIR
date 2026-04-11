"""Validate exog implementation on UCI Bike Sharing (daily, 2 years).

Rolling-origin evaluation:
- For each origin, train on the prior `train_len` days, forecast next `h` days.
- Compare vanilla FLAIR vs FLAIR + weather exog.
- Metric: MASE (Mean Absolute Scaled Error, scale-free).

The bike rental count is strongly driven by weather (temperature, humidity)
and calendar effects (workday).  This is the canonical exog test case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Worktree-local flaircast
sys.path.insert(0, str(Path(__file__).parent.parent))
from flaircast import forecast  # noqa: E402


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, m: int = 7) -> float:
    """Seasonal MASE with seasonality `m` (= weekly for daily data)."""
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
    df = pd.read_csv(here / "bike_daily.csv", quotechar='"')
    df = df.sort_values("dteday").reset_index(drop=True)
    print(f"Loaded {len(df)} daily rows from {df['dteday'].iloc[0]} to {df['dteday'].iloc[-1]}")

    # Target: total rental count
    y_all = df["cnt"].astype(float).to_numpy()

    # Exogenous: numeric weather + calendar dummies (no leakage of cnt itself)
    exog_cols = ["temp", "atemp", "hum", "windspeed"]
    workday_dummy = (df["workday"] == "Y").astype(float).to_numpy()
    holiday_dummy = (df["holiday"] == "Y").astype(float).to_numpy()
    weather_good = (df["weather"] == "GOOD").astype(float).to_numpy()
    X_all_num = df[exog_cols].astype(float).to_numpy()
    X_all = np.column_stack([X_all_num, workday_dummy, holiday_dummy, weather_good])
    print(f"Exog shape: {X_all.shape}, columns: {exog_cols + ['workday', 'holiday', 'weather_good']}")

    # ── Rolling-origin evaluation ──────────────────────────────────────
    train_len = 365  # 1 year history
    horizon = 14  # 2-week forecast
    n_origins = 12  # roughly monthly origins through year 2
    n_total = len(y_all)
    origin_step = (n_total - train_len - horizon) // n_origins
    origins = [train_len + i * origin_step for i in range(n_origins)]

    print(f"\nRolling-origin: train_len={train_len}, horizon={horizon}, n_origins={n_origins}")
    print(f"Origins span y indices {origins[0]}..{origins[-1]} (out of {n_total})")

    rng_seed = 42
    n_samples = 200

    rows = []
    for origin in origins:
        y_hist = y_all[origin - train_len : origin]
        y_true = y_all[origin : origin + horizon]
        X_hist = X_all[origin - train_len : origin]
        X_future = X_all[origin : origin + horizon]

        # Vanilla FLAIR (no exog)
        s_no = forecast(y_hist, horizon, "D", n_samples=n_samples, seed=rng_seed)
        p_no = np.median(s_no, axis=0)

        # FLAIR + exog
        s_ex = forecast(
            y_hist,
            horizon,
            "D",
            n_samples=n_samples,
            seed=rng_seed,
            X_hist=X_hist,
            X_future=X_future,
        )
        p_ex = np.median(s_ex, axis=0)

        mase_no = mase(y_true, p_no, y_hist, m=7)
        mase_ex = mase(y_true, p_ex, y_hist, m=7)
        mae_no = float(np.mean(np.abs(y_true - p_no)))
        mae_ex = float(np.mean(np.abs(y_true - p_ex)))
        smape_no = smape(y_true, p_no)
        smape_ex = smape(y_true, p_ex)

        rows.append(
            dict(
                origin=origin,
                date=df["dteday"].iloc[origin],
                mase_no=mase_no,
                mase_ex=mase_ex,
                mae_no=mae_no,
                mae_ex=mae_ex,
                smape_no=smape_no,
                smape_ex=smape_ex,
            )
        )

    res = pd.DataFrame(rows)
    print("\nPer-origin results:")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Summary (mean across origins) ===")
    print(f"  MASE  no_exog: {res['mase_no'].mean():.4f}   exog: {res['mase_ex'].mean():.4f}   "
          f"Δ%: {(res['mase_ex'].mean() / res['mase_no'].mean() - 1) * 100:+.2f}%")
    print(f"  MAE   no_exog: {res['mae_no'].mean():.2f}     exog: {res['mae_ex'].mean():.2f}     "
          f"Δ%: {(res['mae_ex'].mean() / res['mae_no'].mean() - 1) * 100:+.2f}%")
    print(f"  sMAPE no_exog: {res['smape_no'].mean():.2f}   exog: {res['smape_ex'].mean():.2f}   "
          f"Δ%: {(res['smape_ex'].mean() / res['smape_no'].mean() - 1) * 100:+.2f}%")

    wins = (res["mase_ex"] < res["mase_no"]).sum()
    print(f"\n  exog beats no_exog on {wins}/{len(res)} origins (MASE)")


if __name__ == "__main__":
    main()
