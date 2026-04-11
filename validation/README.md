# Exogenous variable validation

Empirical comparison of vanilla FLAIR vs FLAIR with exogenous variables
on open datasets.

## Datasets

### 1. UCI Bike Sharing (daily)

Already committed as `bike_daily.csv` (157 KB, 728 rows).
Source: https://github.com/christophM/interpretable-ml-book/blob/master/data/bike.csv

Target: `cnt` (daily rental count). Exog: `temp`, `atemp`, `hum`,
`windspeed`, `workday`, `holiday`, `weather_good` (7 columns).

```bash
uv run --with pandas python validation/validate_bike_daily.py
```

### 2. Jena Climate (hourly)

Not committed (43 MB). Download with:

```bash
cd validation
curl -sL -o jena.zip "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
unzip -o jena.zip
rm jena.zip
```

Target: `T (degC)` resampled to hourly. Exog: `p (mbar)`, `rh (%)`,
`wv (m/s)` (3 columns).

```bash
uv run --with pandas python validation/validate_jena.py
uv run --with pandas python validation/validate_noise_robustness.py
```

## Results

Rolling-origin MASE comparison (vanilla FLAIR vs FLAIR + exog):

| Dataset | n_origins | vanilla MASE | +exog MASE | Δ% | wins |
|---|---|---|---|---|---|
| Bike Sharing daily | 12 | 1.178 | **1.067** | **−9.42%** | 9/12 |
| Jena Climate (90-day train) | 24 | 1.043 | **0.881** | **−15.49%** | 19/24 |
| Jena Climate (noise exog control) | 24 | 1.043 | 1.053 | +0.93% | — |

The noise control confirms graceful degradation: passing pure-noise
exogenous variables inflates MASE by less than 1% on average.
