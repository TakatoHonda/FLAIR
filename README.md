# FLAIR

**Factored Level And Interleaved Ridge** — a single-equation time series forecasting method.

## The Idea

Reshape a time series by its primary period, then separate *what* (level) from *how* (shape):

```
y(phase, period) = Level(period) × Shape(phase)
```

- **Level**: period totals, forecast by Ridge regression with soft-average GCV
- **Shape**: within-period proportions, estimated structurally from recent data

One SVD. No model selection. No neural network.

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

1. **Reshape** the series by the primary calendar period (e.g., 24 for hourly → daily)
2. **Shape** = average within-period proportions from the last K periods
3. **Level** = period totals → Box-Cox → NLinear (subtract last value) → Ridge with soft-average GCV
4. **Forecast** Level for `ceil(H/P)` future periods via recursive prediction
5. **Reconstruct** `y_hat = Level_hat × Shape` and generate conformal prediction intervals

## GIFT-Eval Results

| Model | relMASE | relCRPS | Time |
|-------|---------|---------|------|
| **FLAIR** | **1.028** | **0.750** | 28 min |
| MFLES | 1.405 | 1.015 | 578 min |
| SeasonalNaive | 1.000 | 1.000 | — |

Evaluated on all 97 configurations across 23 datasets ([GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval)).

## Citation

```
@misc{flair2026,
  title={FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting},
  year={2026}
}
```

## License

MIT
