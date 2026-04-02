# FLAIR

[![PyPI](https://img.shields.io/pypi/v/flaircast)](https://pypi.org/project/flaircast/)
[![Python](https://img.shields.io/pypi/pyversions/flaircast)](https://pypi.org/project/flaircast/)
[![CI](https://github.com/TakatoHonda/FLAIR/actions/workflows/ci.yml/badge.svg)](https://github.com/TakatoHonda/FLAIR/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/TakatoHonda/FLAIR)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TakatoHonda/FLAIR/blob/main/examples/quickstart.ipynb)

[日本語版はこちら](README_ja.md)

**Factored Level And Interleaved Ridge** — a single-equation time series forecasting method.

Zero hyperparameters. One SVD. CPU only.

- **#1 on [Chronos Benchmark II](https://github.com/amazon-science/chronos-forecasting)** (25 zero-shot datasets) — Agg. Rel. MASE **0.696** vs. Chronos-Bolt-Base 0.791 (205M params, GPU)
- **Best statistical method on [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)** (97 configs, 23 datasets) — relMASE **0.864**, relCRPS **0.614**
- **~500 lines of Python**. Dependencies: numpy + scipy

## Table of Contents

- [Pipeline](#pipeline)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Supported Frequencies](#supported-frequencies)
- [How It Works](#how-it-works)
- [Benchmark Results](#benchmark-results)
- [API Reference](#api-reference)
- [Design Principles](#design-principles)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

## Pipeline

FLAIR reshapes a time series by its primary period, then separates *what happens* (level) from *how it happens* (shape):

```
y(phase, period) = Level(period) × Shape(phase)
```

<p align="center">
  <img src="assets/fig_pipeline.png" alt="FLAIR Pipeline" width="100%">
</p>

Shape is structural (not learned), so it does not overfit. Level is a smooth, compressed series — one value per period instead of P values — forecast by Ridge regression. Two compressions happen simultaneously: summing P phases into one Level value reduces noise by ~√P, and forecasting Level requires only ⌈H/P⌉ recursive steps instead of H.

## Quick Start

```python
import numpy as np
from flaircast import forecast, FLAIR

y = np.random.rand(500) * 100  # your time series

# ── Functional API ───────────────────────────
samples = forecast(y, horizon=24, freq='H')
point   = samples.mean(axis=0)           # (24,)
lo, hi  = np.percentile(samples, [10, 90], axis=0)

# ── Class API (handy in loops) ───────────────
model   = FLAIR(freq='H')
samples = model.predict(y, horizon=24)

# ── From pandas ──────────────────────────────
import pandas as pd
ts = pd.read_csv('data.csv')['value']
samples = forecast(ts.values, horizon=12, freq='M')
```

## Installation

```bash
pip install flaircast
```

Or install from source:

```bash
git clone https://github.com/TakatoHonda/FLAIR.git
cd FLAIR
pip install .
```

## Supported Frequencies

| Freq string | Period | Meaning | MDL candidates |
|:-----------:|:------:|---------|:--------------:|
| `S` | 60 | Second | 60 |
| `T` | 60 | Minute | 60 |
| `5T` | 12 | 5-minute | 12, 288 |
| `10T` | 6 | 10-minute | 6, 144 |
| `15T` | 4 | 15-minute | 4, 96 |
| `10S` | 6 | 10-second | 6, 360 |
| `H` | 24 | Hourly | 24, 168 |
| `D` | 7 | Daily | 7, 365 |
| `W` | 52 | Weekly | 52 |
| `M` | 12 | Monthly | 12 |
| `Q` | 4 | Quarterly | 4 |
| `A` / `Y` | 1 | Annual | — |

When multiple candidates exist, FLAIR uses BIC on the SVD spectrum (MDL principle) to select the period that best supports a rank-1 structure.

## How It Works

1. **MDL Period Selection**: BIC on SVD spectrum selects the primary period P from calendar candidates
2. **Reshape** the series into a (P × n_complete) matrix
3. **Shape₁** = Dirichlet posterior mean per context (`context = period_index % C`). Shrinks toward the global average when data is scarce
4. **Level** = period totals
5. **Shape₂** = secondary periodic pattern in Level, estimated as `w × raw + (1−w) × prior`, where `w = nc₂/(nc₂+cp)`. The prior is selected by BIC: first harmonic (2 params) when justified, flat (0 params) otherwise. Level is deseasonalized by dividing by Shape₂
6. **Ridge** on deseasonalized Level: Box-Cox → NLinear → intercept + trend + lags → soft-average GCV
7. **Stochastic Level paths**: LOO residuals are injected into the recursive forecast — errors propagate through the Ridge lag dynamics naturally. Mean-reverting series saturate; random-walk series grow as √step. No scaling formula needed
8. **Phase noise** from SVD Residual Quantiles: E = M − fitted gives phase-specific relative noise. Sampled scenario-coherently — all phases within the same forecast step share one historical period's residual pattern, preserving cross-phase correlation. Combined with Level paths: `sample = Level_path × Shape₁ × (1 + phase_noise)`

## Benchmark Results

### Chronos Benchmark II (25 zero-shot datasets)

Evaluated on the [Chronos](https://github.com/amazon-science/chronos-forecasting) Benchmark II protocol (Ansari et al., 2024). Agg. Relative Score = geometric mean of (method / Seasonal Naive) per dataset. Lower is better.

<p align="center">
  <img src="assets/fig_chronos.png" alt="Chronos Benchmark" width="85%">
</p>

| Rank | Model | Params | Agg. Rel. MASE | GPU |
|:----:|-------|--------|:--------------:|:---:|
| **1** | **FLAIR** | **~6** | **0.696** | **No** |
| 2 | Chronos-Bolt-Base | 205M | 0.791 | Yes |
| 3 | Moirai-Base | 311M | 0.812 | Yes |
| 4 | Chronos-T5-Base | 200M | 0.816 | Yes |
| 5 | Chronos-Bolt-Small | 48M | 0.819 | Yes |
| 6 | Chronos-T5-Large | 710M | 0.821 | Yes |
| 7 | Chronos-T5-Small | 46M | 0.830 | Yes |
| 8 | AutoARIMA | — | 0.865 | No |
| 9 | Chronos-T5-Tiny | 8M | 0.870 | Yes |
| 10 | TimesFM | 200M | 0.879 | Yes |
| 11 | AutoETS | — | 0.937 | No |
| 12 | Seasonal Naive | — | 1.000 | No |

Baseline results from [autogluon/fev](https://github.com/autogluon/fev) and [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting). FLAIR outperforms Chronos-T5-Small (46M params) on **14 of 25 datasets** in point forecast accuracy.

### GIFT-Eval (97 configs, 23 datasets)

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — 7 domains, short/medium/long horizons, 53 non-agentic methods (no test leakage):

<p align="center">
  <img src="assets/fig_benchmark.png" alt="GIFT-Eval Benchmark" width="100%">
</p>

| Model | Type | relMASE | relCRPS | GPU |
|-------|------|:-------:|:-------:|:---:|
| **FLAIR** | **Statistical** | **0.864** | **0.614** | **No** |
| PatchTST | Deep Learning | 0.849 | 0.587 | Yes |
| Moirai-large | Foundation | 0.875 | 0.599 | Yes |
| iTransformer | Deep Learning | 0.893 | 0.620 | Yes |
| TFT | Deep Learning | 0.915 | 0.605 | Yes |
| N-BEATS | Deep Learning | 0.938 | 0.816 | Yes |
| SeasonalNaive | Baseline | 1.000 | 1.000 | No |
| AutoARIMA | Statistical | 1.074 | 0.912 | No |
| Prophet | Statistical | 1.540 | 1.061 | No |

### Long-term Forecasting (8 datasets)

Standard benchmark from PatchTST, iTransformer, DLinear, Autoformer. Channel-independent (univariate) evaluation. MSE on StandardScaler-normalized data. Horizons: {96, 192, 336, 720}.

Average MSE across all 4 horizons:

| Dataset | FLAIR | iTransformer | PatchTST | DLinear | GPU needed |
|---------|:-----:|:------------:|:--------:|:-------:|:----------:|
| **ETTh2** | **0.366** | 0.383 | 0.387 | 0.559 | **No** |
| **ETTm2** | **0.257** | 0.288 | 0.281 | 0.350 | **No** |
| **Weather** | **0.248** | 0.258 | 0.259 | 0.265 | **No** |
| ECL | 0.215 | **0.178** | 0.205 | 0.212 | Yes |
| Traffic | 0.434 | **0.428** | 0.481 | 0.625 | Yes |
| ETTh1 | 0.591 | **0.454** | 0.469 | 0.456 | Yes |
| ETTm1 | 0.511 | 0.407 | **0.387** | 0.403 | Yes |
| Exchange | 0.815 | 0.360 | 0.366 | **0.354** | Yes |

FLAIR outperforms GPU-trained Transformers on **3 of 8 datasets** and **11 of 32 individual settings**. Accuracy is higher on datasets with clear periodicity and lower on non-periodic series (Exchange).

### Why does FLAIR work?

Three compressions act simultaneously:

1. **Noise reduction**: summing P phases into one Level value reduces noise by ~√P
2. **Horizon compression**: forecasting Level requires only ⌈H/P⌉ steps instead of H, reducing error accumulation
3. **Shape is fixed**: Shape is a Dirichlet posterior, not a learned parameter — it does not overfit

## API Reference

### `forecast(y, horizon, freq, n_samples=200, seed=None)`

Generate probabilistic forecasts for a univariate time series.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y` | array-like (n,) | Historical observations |
| `horizon` | int | Number of steps to forecast |
| `freq` | str | Frequency string (see [table](#supported-frequencies)) |
| `n_samples` | int | Number of sample paths (default: 200) |
| `seed` | int or None | Random seed for reproducibility (default: None) |

**Returns**: `ndarray` of shape `(n_samples, horizon)` — probabilistic forecast sample paths.

```python
from flaircast import forecast
samples = forecast(y, horizon=24, freq='H')
point   = samples.mean(axis=0)
median  = np.median(samples, axis=0)
lo, hi  = np.percentile(samples, [10, 90], axis=0)
```

### `FLAIR(freq, n_samples=200, seed=None)`

Class wrapper. Useful when forecasting multiple series with the same frequency.

| Method | Description |
|--------|-------------|
| `predict(y, horizon, n_samples=None, seed=None)` | Same as `forecast()`, uses instance defaults |

```python
from flaircast import FLAIR
model = FLAIR(freq='D', n_samples=500)
for series in dataset:
    samples = model.predict(series, horizon=7)
```

### Constants

| Name | Description |
|------|-------------|
| `FREQ_TO_PERIOD` | Maps frequency strings to primary periods |
| `FREQ_TO_PERIODS` | Maps frequency strings to MDL candidate periods |

## Design Principles

FLAIR applies the **Minimum Description Length** principle at every scale:

| Scale | Mechanism | MDL Role |
|-------|-----------|----------|
| Period P | BIC on SVD spectrum | Select simplest rank-1 structure |
| Shape₁ | Dirichlet shrinkage | Shrink to global average (simplest distribution) |
| Shape₂ | BIC-gated shrinkage | BIC selects prior: harmonic (2 params) vs flat (0 params) |
| Ridge α | GCV soft-average | Select model complexity via cross-validation |

## Limitations

- **Non-periodic series**: the Level × Shape decomposition provides no compression benefit when there is no periodicity (e.g., exchange rates). Use a dedicated non-periodic model instead
- **Intermittent demand**: series with >30% zeros are poorly served by the multiplicative structure. Croston-type methods are better suited
- **No exogenous variables**: the core API does not accept external features (calendar events, prices, promotions)
- **Short series**: fewer than 3 complete periods forces P=1 degeneration (plain Ridge on raw series)

## Citation

```
@misc{flair2026,
  title={FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting},
  year={2026}
}
```

## License

Apache License 2.0
