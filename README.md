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
from flair import flair_forecast

y = np.random.rand(500) * 100  # your time series
samples = flair_forecast(y, horizon=24, freq='H')
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
8. **Phase noise** from SVD Residual Quantiles: E = M − fitted gives phase-specific relative noise. Combined with Level paths: `sample = Level_path × Shape₁ × (1 + phase_noise)`

## Benchmark Results

### Chronos Zero-Shot Benchmark (25 datasets)

Evaluated on the same protocol as [Chronos](https://github.com/amazon-science/chronos-forecasting) (Ansari et al., 2024). Agg. Relative Score = geometric mean of (method / Seasonal Naive) per dataset. Lower is better.

Baseline results from [autogluon/fev](https://github.com/autogluon/fev/tree/main/benchmarks/chronos_zeroshot/results) and [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/evaluation/results).

| Model | Params | Agg. Rel. MASE | Agg. Rel. WQL | GPU |
|-------|--------|:--------------:|:-------------:|:---:|
| **FLAIR** | **~6** | **0.704** | 0.799 | **No** |
| Chronos-Bolt-Base | 205M | 0.791 | **0.624** | Yes |
| Moirai-Base | 311M | 0.812 | 0.637 | Yes |
| Chronos-T5-Base | 200M | 0.816 | 0.642 | Yes |
| Chronos-Bolt-Small | 48M | 0.819 | 0.636 | Yes |
| Chronos-T5-Large | 710M | 0.821 | 0.650 | Yes |
| Chronos-T5-Small | 46M | 0.830 | 0.665 | Yes |
| Chronos-T5-Mini | 20M | 0.841 | 0.689 | Yes |
| Chronos-Bolt-Tiny | 9M | 0.845 | 0.668 | Yes |
| AutoARIMA | - | 0.865 | 0.742 | No |
| Chronos-T5-Tiny | 8M | 0.870 | 0.711 | Yes |
| TimesFM | 200M | 0.879 | 0.711 | Yes |
| AutoETS | - | 0.937 | 0.812 | No |
| Seasonal Naive | - | 1.000 | 1.000 | No |

**FLAIR ranks #1 on point forecast accuracy (MASE)** — beating every Chronos variant (up to 710M params), Moirai, TimesFM, AutoARIMA, and AutoETS. No GPU. No pretraining.

### GIFT-Eval Benchmark (97 configs, 23 datasets)

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — 7 domains, short/medium/long horizons:

| Model | Type | relMASE | relCRPS | GPU |
|-------|------|:-------:|:-------:|:---:|
| **FLAIR** | **Statistical** | **0.866** | **0.615** | **No** |
| Chronos-Small | Foundation | 0.892 | — | Yes |
| N-BEATS | Deep Learning | 0.938 | 0.816 | Yes |
| TFT | Deep Learning | 0.915 | 0.605 | Yes |
| SeasonalNaive | Baseline | 1.000 | 1.000 | No |
| AutoARIMA | Statistical | 1.074 | 0.912 | No |
| DeepAR | Deep Learning | 1.343 | 0.853 | Yes |
| Prophet | Statistical | 1.540 | 1.061 | No |

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
