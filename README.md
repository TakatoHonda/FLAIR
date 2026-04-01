# FLAIR

**Factored Level And Interleaved Ridge** — a single-equation time series forecasting method.

## The Idea

Reshape a time series by its primary period, then separate *what* (level) from *how* (shape):

```
y(phase, period) = Level(period) × Shape(phase)
```

- **Shape₁**: within-period proportions via Dirichlet-Multinomial empirical Bayes, with context from the secondary period
- **Shape₂**: secondary periodicity in Level, handled by the same proportional decomposition — raw proportions shrunk toward a BIC-selected prior (first harmonic or flat) via empirical Bayes
- **Level**: period totals, deseasonalized by Shape₂, forecast by Ridge with soft-average GCV
- **Location shift**: automatic handling of negative-valued series
- **P=1 degeneration**: no separate fallback model — one unified code path

Zero hyperparameters. No neural network. CPU only.

## Usage

```python
import numpy as np
from flair import forecast

y = np.random.rand(500) * 100  # your time series
samples = forecast(y, horizon=24, freq='H')
point_forecast = samples.mean(axis=0)
```

## Dependencies

- numpy
- scipy

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

### Chronos Benchmark II / Monash (25 zero-shot datasets)

Evaluated on the [Chronos](https://github.com/amazon-science/chronos-forecasting) Benchmark II protocol (Ansari et al., 2024). The 25 datasets are primarily from the [Monash Time Series Forecasting Archive](https://forecastingdata.org/) (Godahewa et al., NeurIPS 2021) plus M4, M5, and others. Agg. Relative Score = geometric mean of (method / Seasonal Naive) per dataset. Lower is better.

All scores computed on the same 25 datasets. Baseline results from [autogluon/fev](https://github.com/autogluon/fev/tree/main/benchmarks/chronos_zeroshot/results) and [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/evaluation/results). Deep Learning baselines from Chronos paper Figure 5.

| Rank | Model | Params | Agg. Rel. MASE | Agg. Rel. WQL | GPU |
|:----:|-------|--------|:--------------:|:-------------:|:---:|
| **1** | **FLAIR** | **~6** | **0.691** | 0.801 | **No** |
| 2 | Moirai-Large | 1B | 0.787 | 0.633 | Yes |
| 3 | TimesFM-2.0 | 200M | 0.797 | 0.719 | Yes |
| 4 | Chronos-Bolt-Base | 205M | 0.803 | **0.639** | Yes |
| 5 | PatchTST | per-dataset | 0.810 | 0.684 | Yes |
| 6 | Moirai-Base | 311M | 0.812 | 0.635 | Yes |
| 7 | Chronos-T5-Base | 200M | 0.822 | 0.648 | Yes |
| 8 | Chronos-T5-Large | 710M | 0.830 | 0.659 | Yes |
| 9 | N-HiTS | per-dataset | 0.830 | 0.672 | Yes |
| 10 | Chronos-Bolt-Small | 48M | 0.832 | 0.651 | Yes |
| 11 | N-BEATS | per-dataset | 0.835 | 0.681 | Yes |
| 12 | Chronos-T5-Small | 46M | 0.839 | 0.675 | Yes |
| 13 | TFT | per-dataset | 0.847 | 0.639 | Yes |
| 14 | AutoARIMA | - | 0.865 | 0.741 | No |
| 15 | TimesFM | 200M | 0.879 | 0.711 | Yes |
| 16 | AutoTheta | - | 0.881 | 0.795 | No |
| 17 | Moirai-Small | 91M | 0.890 | 0.707 | Yes |
| 18 | AutoETS | - | 0.937 | 0.815 | No |
| 19 | Seasonal Naive | - | 1.000 | 1.000 | No |

**FLAIR ranks #1 on point forecast accuracy (MASE)** out of 19 methods — beating Moirai-Large (1B params) by 12.2%, all Chronos variants (up to 710M params), per-dataset Deep Learning (PatchTST, N-BEATS, TFT), and all statistical baselines. No GPU. No pretraining. No hyperparameters.

### GIFT-Eval Benchmark (97 configs, 23 datasets)

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — 7 domains, short/medium/long horizons:

| Model | Type | relMASE | relCRPS | GPU |
|-------|------|:-------:|:-------:|:---:|
| **FLAIR** | **Statistical** | **0.866** | **0.615** | **No** |
| PatchTST | Deep Learning | 0.849 | 0.587 | Yes |
| Moirai-large | Foundation | 0.875 | 0.599 | Yes |
| iTransformer | Deep Learning | 0.893 | 0.620 | Yes |
| TFT | Deep Learning | 0.915 | 0.605 | Yes |
| N-BEATS | Deep Learning | 0.938 | 0.816 | Yes |
| SeasonalNaive | Baseline | 1.000 | 1.000 | No |
| AutoARIMA | Statistical | 1.074 | 0.912 | No |
| Prophet | Statistical | 1.540 | 1.061 | No |

### Long-term Forecasting Benchmark (8 datasets)

Standard benchmark used by PatchTST, iTransformer, DLinear, Autoformer, etc. Channel-independent (univariate) evaluation. Metrics: MSE on StandardScaler-normalized data. Prediction horizons: {96, 192, 336, 720}.

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

FLAIR wins **3/8 datasets** (ETTh2, ETTm2, Weather) and **11/32 individual settings** against GPU-trained Transformers — with zero training, zero hyperparameters, and CPU only. FLAIR is strongest on datasets with clear periodicity; weaker on non-periodic series (Exchange) where the Level × Shape decomposition provides no compression.

Baseline numbers from the [iTransformer paper](https://arxiv.org/abs/2310.06625) (ICLR 2024).

## Design Principles

FLAIR applies the **Minimum Description Length** principle at every scale:

| Scale | Mechanism | MDL Role |
|-------|-----------|----------|
| Period P | BIC on SVD spectrum | Select simplest rank-1 structure |
| Shape₁ | Dirichlet shrinkage | Shrink to global average (simplest distribution) |
| Shape₂ | BIC-gated shrinkage | BIC selects prior: harmonic (2 params) vs flat (0 params) |
| Ridge α | GCV soft-average | Select model complexity via cross-validation |

## Citation

```
@misc{flair2026,
  title={FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting},
  year={2026}
}
```

## License

MIT
