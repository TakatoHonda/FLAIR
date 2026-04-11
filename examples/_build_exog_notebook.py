"""Generate examples/exogenous_variables.ipynb.

Notebook contents are defined here as Python literals so the cells stay
trivially editable / reviewable.  Run this once whenever the notebook
needs to be regenerated:

    uv run python examples/_build_exog_notebook.py

The output is overwritten in place at examples/exogenous_variables.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"id": cell_id},
        "source": source,
    }


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": cell_id},
        "outputs": [],
        "source": source,
    }


CELLS: list[dict] = [
    md(
        """\
# FLAIR with Exogenous Variables

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakatoHonda/FLAIR/blob/main/examples/exogenous_variables.ipynb)

**Factored Level And Interleaved Ridge** — zero-hyperparameter time series forecasting *with covariates*.

This notebook shows how to feed external information (weather, calendar flags, prices, …) into FLAIR via the new `X_hist` / `X_future` parameters introduced in v0.5.0, and demonstrates a real-world accuracy gain on the UCI Bike Sharing dataset.

What you'll see:

1. Why exogenous variables (and the design behind FLAIR's exog support)
2. Loading the UCI Bike Sharing daily data
3. Vanilla FLAIR baseline (no covariates)
4. FLAIR + weather + calendar covariates
5. Side-by-side comparison and metrics
6. Tips, recommended training-window size, and the intra-period limitation""",
        cell_id="header",
    ),
    md("## 0. Install", cell_id="install-header"),
    code(
        """\
!pip install -q "flaircast>=0.5.0" pandas matplotlib
import flaircast
print(f"flaircast {flaircast.__version__}")
assert flaircast.__version__ >= "0.5.0", (
    "Exogenous variable support requires flaircast >= 0.5.0. "
    "If you just upgraded, restart the runtime: Runtime > Restart session"
)""",
        cell_id="install",
    ),
    code(
        """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flaircast import forecast""",
        cell_id="imports",
    ),
    md(
        """\
## 1. Why exogenous variables, and how FLAIR uses them

Vanilla FLAIR forecasts a series from its own history alone.  When the target depends on something *external* — temperature driving bike rentals, holidays driving retail sales, prices driving demand — passing that external signal in as a covariate can substantially improve accuracy.

The exog API is intentionally minimal:

```python
samples = forecast(
    y, horizon, freq,
    X_hist=X_hist,        # shape (n, k) or (n,)  — historical covariates
    X_future=X_future,    # shape (h, k) or (h,)  — future covariates
)
```

Under the hood:

- `X_hist` and `X_future` are aggregated to the per-period (Level) timescale via period mean.
- The columns are z-scored using **training-window statistics only** so the Ridge regularization sees comparable scales.
- The standardized columns are appended directly to the Level Ridge feature matrix — **no separate gating step**.  FLAIR's existing LOOCV soft-averaged Ridge already shrinks columns whose contribution to leave-one-out error is negligible, so noise covariates are naturally damped.

"One SVD, One Ridge" is preserved.  When `X_hist=None` the result is bit-identical to the v0.4.x behavior.""",
        cell_id="why-exog",
    ),
    md(
        """\
## 2. Load the UCI Bike Sharing daily data

We use the daily aggregation of the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset): two years (2011–2012) of total rental counts plus weather and calendar covariates.

The CSV is bundled with the FLAIR repo so this notebook is self-contained.""",
        cell_id="load-header",
    ),
    code(
        """\
url = "https://raw.githubusercontent.com/TakatoHonda/FLAIR/main/validation/bike_daily.csv"
df = pd.read_csv(url, quotechar='"').sort_values("dteday").reset_index(drop=True)

print(f"{len(df)} daily rows from {df['dteday'].iloc[0]} to {df['dteday'].iloc[-1]}")
df.head()""",
        cell_id="load-csv",
    ),
    code(
        """\
y_all = df["cnt"].astype(float).to_numpy()  # daily rental count

# Numeric weather + calendar dummies — anything you'd "know in advance"
exog_cols = ["temp", "atemp", "hum", "windspeed"]
workday  = (df["workday"]  == "Y").astype(float).to_numpy()
holiday  = (df["holiday"]  == "Y").astype(float).to_numpy()
weather  = (df["weather"]  == "GOOD").astype(float).to_numpy()
X_all = np.column_stack([df[exog_cols].astype(float).to_numpy(), workday, holiday, weather])

print(f"y shape: {y_all.shape}, exog shape: {X_all.shape}")
print(f"exog columns: {exog_cols + ['workday', 'holiday', 'weather_good']}")""",
        cell_id="load-arrays",
    ),
    md(
        """\
## 3. Train / test split

We hold out the **last 14 days** as the forecast horizon and train on the preceding **365 days** (one full year of seasonality).""",
        cell_id="split-header",
    ),
    code(
        """\
horizon = 14
train_len = 365
origin = len(y_all) - horizon

y_hist   = y_all[origin - train_len : origin]
y_actual = y_all[origin : origin + horizon]
X_hist   = X_all[origin - train_len : origin]
X_future = X_all[origin : origin + horizon]

print(f"train: {len(y_hist)} days, horizon: {horizon} days")
print(f"actual range: {y_actual.min():.0f} – {y_actual.max():.0f}")""",
        cell_id="split",
    ),
    md("## 4. Vanilla FLAIR baseline (no covariates)", cell_id="vanilla-header"),
    code(
        """\
samples_no = forecast(y_hist, horizon=horizon, freq="D", n_samples=200, seed=42)

point_no = np.median(samples_no, axis=0)
lo_no, hi_no = np.percentile(samples_no, [10, 90], axis=0)

print(f"samples shape: {samples_no.shape}")
print(f"point forecast: {np.round(point_no).astype(int)}")""",
        cell_id="vanilla-forecast",
    ),
    md("## 5. FLAIR with weather + calendar covariates", cell_id="exog-header"),
    code(
        """\
samples_ex = forecast(
    y_hist,
    horizon=horizon,
    freq="D",
    n_samples=200,
    seed=42,
    X_hist=X_hist,
    X_future=X_future,
)

point_ex = np.median(samples_ex, axis=0)
lo_ex, hi_ex = np.percentile(samples_ex, [10, 90], axis=0)

print(f"samples shape: {samples_ex.shape}")
print(f"point forecast: {np.round(point_ex).astype(int)}")""",
        cell_id="exog-forecast",
    ),
    md(
        """\
## 6. Side-by-side comparison

The same FLAIR call, with and without covariates.  The exog version aligns more closely with the actual rental counts on weekday-vs-weekend transitions and on warm/cold days.""",
        cell_id="compare-header",
    ),
    code(
        """\
fig, ax = plt.subplots(figsize=(12, 5))

# History (last 60 days for context)
hist_view = 60
hist_x = np.arange(-hist_view, 0)
ax.plot(hist_x, y_hist[-hist_view:], color="black", linewidth=0.9, label="History")

# Actual held-out values
fc_x = np.arange(horizon)
ax.plot(fc_x, y_actual, color="gray", linestyle="--", linewidth=1.4, label="Actual")

# Vanilla FLAIR
ax.plot(fc_x, point_no, color="#dc2626", linewidth=1.6, label="Vanilla FLAIR")
ax.fill_between(fc_x, lo_no, hi_no, alpha=0.12, color="#dc2626")

# FLAIR + exog
ax.plot(fc_x, point_ex, color="#2563eb", linewidth=1.6, label="FLAIR + exog")
ax.fill_between(fc_x, lo_ex, hi_ex, alpha=0.18, color="#2563eb")

ax.axvline(x=0, color="gray", linestyle=":", linewidth=0.8)
ax.legend(loc="upper left")
ax.set_title("UCI Bike Sharing — 14-day forecast")
ax.set_xlabel("Days from forecast origin")
ax.set_ylabel("Rental count")
plt.tight_layout()
plt.show()""",
        cell_id="compare-plot",
    ),
    md("## 7. Metrics", cell_id="metrics-header"),
    code(
        """\
def mase(y_true, y_pred, y_train, m=7):
    \"\"\"Seasonal MASE with weekly seasonality.\"\"\"
    naive_err = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / naive_err)


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)) * 100)


mae_no  = float(np.mean(np.abs(y_actual - point_no)))
mae_ex  = float(np.mean(np.abs(y_actual - point_ex)))
mase_no = mase(y_actual, point_no, y_hist)
mase_ex = mase(y_actual, point_ex, y_hist)
sm_no   = smape(y_actual, point_no)
sm_ex   = smape(y_actual, point_ex)

print(f"{'metric':<8} {'vanilla':>10} {'+exog':>10}   delta")
print(f"{'MAE':<8} {mae_no:>10.1f} {mae_ex:>10.1f}   {(mae_ex / mae_no - 1) * 100:+.1f}%")
print(f"{'MASE':<8} {mase_no:>10.4f} {mase_ex:>10.4f}   {(mase_ex / mase_no - 1) * 100:+.1f}%")
print(f"{'sMAPE':<8} {sm_no:>10.2f} {sm_ex:>10.2f}   {(sm_ex / sm_no - 1) * 100:+.1f}%")""",
        cell_id="metrics",
    ),
    md(
        """\
A single train/test split is noisy.  For a proper rolling-origin evaluation across 12 forecast origins (covering year 2 of the dataset) FLAIR + exog reduces MASE by **−9.4%** and wins **9 out of 12** origins.  See `validation/validate_bike_daily.py` in the FLAIR repository for the full benchmark.""",
        cell_id="metrics-note",
    ),
    md(
        """\
## 8. What the BIC-free design buys you

It is sometimes useful to verify that the gain on real covariates does not come at the cost of catastrophic regression on bad covariates.  Here we replace the real exog with **pure Gaussian noise of the same shape** and re-run the forecast.

Because FLAIR's LOOCV soft-averaged Ridge shrinks columns that don't reduce LOO error, noise covariates are naturally damped — there is no model-selection step that can over-accept noise.""",
        cell_id="noise-header",
    ),
    code(
        """\
noise_rng = np.random.RandomState(7)
X_hist_noise   = noise_rng.randn(*X_hist.shape)
X_future_noise = noise_rng.randn(*X_future.shape)

samples_noise = forecast(
    y_hist,
    horizon=horizon,
    freq="D",
    n_samples=200,
    seed=42,
    X_hist=X_hist_noise,
    X_future=X_future_noise,
)
point_noise = np.median(samples_noise, axis=0)

mase_noise = mase(y_actual, point_noise, y_hist)
print(f"MASE — vanilla:     {mase_no:.4f}")
print(f"MASE — real exog:   {mase_ex:.4f}  ({(mase_ex / mase_no - 1) * 100:+.1f}%)")
print(f"MASE — noise exog:  {mase_noise:.4f}  ({(mase_noise / mase_no - 1) * 100:+.1f}%)")""",
        cell_id="noise-test",
    ),
    md(
        """\
## 9. Tips and limitations

**Recommended setup**

- Provide at least a few dozen complete periods of training history — for daily data with a weekly period, ~60–90 days is a comfortable minimum for stable coefficient estimates.  Below ~30 periods the LOOCV becomes noisy enough that exog is hit-or-miss.
- `X_future` must be the values you actually expect to see during the forecast horizon.  In practice this is either a known schedule (calendar / promo flags) or a downstream forecast (weather forecast).  In benchmarks it is the held-out actuals — *perfect-foresight*, the standard convention in the literature.
- The columns can be a mix of types (continuous + 0/1 dummies); FLAIR z-scores everything internally.

**Backward compatibility**

`X_hist=None` (the default) is byte-identical to the v0.4.x behavior.  You can drop the new arguments into existing code without changing any other output.

**Limitation: coarse temporal resolution**

`X_hist` and `X_future` are aggregated to the per-period (Level) timescale via period mean.  Intra-period variation in the covariates — for example, hourly temperature within a daily period — is dropped by design.  Use cases that require sub-period covariate effects (e.g. hour-by-hour electricity load reacting to hour-by-hour temperature) are a known non-goal of v0.5.0.

**Failure mode: noise**

In the noise control above, the MASE inflation should be **well under 5%** on average; in the rolling-origin benchmark it averages **+0.93%**.  This is graceful degradation, not a hard guarantee — adversarial small-sample setups can occasionally shift the median forecast more than that.""",
        cell_id="tips",
    ),
    md(
        """\
---

**Links**: [GitHub](https://github.com/TakatoHonda/FLAIR) · [PyPI](https://pypi.org/project/flaircast/) · [API Reference](https://github.com/TakatoHonda/FLAIR#api-reference) · [Quickstart notebook](https://colab.research.google.com/github/TakatoHonda/FLAIR/blob/main/examples/quickstart.ipynb)""",
        cell_id="footer",
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).parent / "exogenous_variables.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
