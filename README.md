# FLAIR

**Factored Level And Interleaved Ridge** — a single-equation time series forecasting method.

## The Idea

Reshape a time series by its primary period, then separate *what* (level) from *how* (shape):

```
y(phase, period) = Level(period) × Shape(phase)
```

- **Level**: period totals, forecast by Ridge regression with soft-average GCV
- **Shape**: within-period proportions via Dirichlet-Multinomial empirical Bayes, with context derived from the secondary period structure
- **Location shift**: automatic handling of negative-valued series via shift-before-Box-Cox

One SVD. Zero hyperparameters. No neural network.

## Usage

```python
import numpy as np
from flair import flair_forecast

y = np.random.rand(500) * 100  # your time series (can include negatives)
samples = flair_forecast(y, horizon=24, freq='H')
point_forecast = samples.mean(axis=0)
```

## Dependencies

- numpy
- scipy

## How It Works

1. **Location shift**: shift all values to be positive (handles negative-valued series like temperature)
2. **Reshape** the series by the primary calendar period (e.g., 24 for hourly)
3. **Shape** = Dirichlet posterior mean per context (`context = period_index % C`, where `C = secondary/primary` period). Shrinks toward the global average when context-specific data is scarce
4. **Level** = period totals → Box-Cox → NLinear → Ridge with soft-average GCV
5. **Forecast** Level for `ceil(H/P)` future periods via recursive prediction
6. **Reconstruct** `y_hat = Level_hat × Shape`, undo shift, and generate conformal prediction intervals

## GIFT-Eval Results

97 configurations, 23 datasets, 7 domains ([GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval)):

| Model | relMASE | relCRPS | GPU | Time |
|-------|---------|---------|-----|------|
| **FLAIR** | **0.920** | **0.690** | No | 29 min |
| DLinear | 1.061 | 0.846 | Yes | — |
| AutoARIMA | 1.074 | 0.912 | No | — |
| AutoTheta | 1.090 | 1.244 | No | — |
| Prophet | 1.540 | 1.061 | No | 597 min |
| SeasonalNaive | 1.000 | 1.000 | No | — |

Per-horizon: short=0.900, medium=0.929, long=0.965 — beats SeasonalNaive on all horizons.

## Citation

```
@misc{flair2026,
  title={FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting},
  year={2026}
}
```

## License

MIT
