# Research: Improving FLAIR on Short/Low-Frequency Series

**Date**: 2026-03-21
**Context**: FLAIR (Level x Shape Ridge) fails on short series with low frequency where n_complete < ~15 or seasonality is weak. MFLES beats FLAIR on: m4_yearly (n~25, P=1), m4_daily (n~100, P=7), restaurant/D, car_parts/M, hospital/M, bitbrains_rnd/H, solar/H.

---

## 1. Shrinkage Toward Seasonal Naive (Ridge with Non-Zero Prior Mean)

### Core Insight

Standard Ridge shrinks beta toward **zero**, which is arbitrary. In forecasting, a better shrinkage target is the **seasonal naive forecast** — the last observed period repeated. This is the Bayesian interpretation of Ridge with non-zero prior mean.

### Mathematical Formulation

Standard Ridge solves:
```
beta_ridge = argmin_beta ||y - X*beta||^2 + lambda * ||beta||^2
           = (X'X + lambda*I)^{-1} X'y
```

Generalized Ridge with prior mean mu_0:
```
beta_gen = argmin_beta ||y - X*beta||^2 + lambda * ||beta - mu_0||^2
         = (X'X + lambda*I)^{-1} (X'y + lambda*mu_0)
```

This is a **weighted average** between OLS and the prior:
```
beta_gen = (X'X + lambda*I)^{-1} X'X * beta_OLS + (X'X + lambda*I)^{-1} lambda * mu_0
         = W * beta_OLS + (I - W) * mu_0
```
where `W = (X'X + lambda*I)^{-1} X'X` is the shrinkage matrix.

### Constructing mu_0 from Seasonal Naive

For the Level series in V9 (period totals L_1, ..., L_n_complete):
- Seasonal naive for Level = repeat last value: L_hat(k) = L(n_complete) for all k
- In NLinear space (innovations relative to last): seasonal naive = 0 for all h
- Therefore mu_0 = 0 for all coefficients, which is exactly standard Ridge!

**Key insight**: Standard Ridge on NLinear-transformed data is ALREADY shrinkage toward seasonal naive.

For the raw (non-NLinear) series, the prior should be:
- Intercept coefficient: last observed value
- Trend coefficient: 0 (no trend = naive)
- Fourier coefficients: 0 (no seasonality = naive)
- Lag-1 coefficient: 1 (random walk = naive)
- Seasonal lag coefficient: 0

```python
mu_0 = np.zeros(n_feat)
mu_0[0] = y_last          # intercept -> last value
mu_0[n_base] = 1.0        # lag-1 -> random walk
# all other coefficients -> 0 (shrink toward no trend, no seasonality)

beta = np.linalg.solve(X.T @ X + alpha * I, X.T @ y + alpha * mu_0)
```

### Adaptive Shrinkage Strength

The shrinkage strength lambda should **increase** as data gets shorter:
- With n observations, estimation error scales as p/n
- James-Stein shrinkage factor: `B = 1 - (p-2)*sigma^2 / ||beta_OLS||^2`
- When n is small relative to p, B is small -> more shrinkage toward prior

**GCV still works** for selecting lambda because the formula is unchanged:
```
GCV(lambda) = (1/n) * sum_i [(y_i - hat_y_i) / (1 - h_ii)]^2
```
The soft-average over lambda values naturally adapts: short series with noisy estimates will favor larger lambda, pulling toward the prior.

### Implementation with SVD (Preserving Current Architecture)

For the non-zero prior case, rewrite as:
```python
# Transform to zero-prior problem
y_tilde = y - X @ mu_0  # subtract prior prediction
# Standard Ridge on transformed target
beta_delta = ridge_solve(X, y_tilde, alpha)  # = (X'X + alpha*I)^{-1} X' y_tilde
# Add back prior
beta = beta_delta + mu_0
```

This is mathematically equivalent and uses the existing SVD-based solver unchanged.

### Verdict for FLAIR

FLAIR's NLinear trick already provides shrinkage toward naive (since the target becomes innovations around last value). This is elegant and correct. The remaining question is whether the Fourier features for Shape estimation could benefit from a non-zero prior.

**For Shape**: Instead of estimating from last K periods, shrink toward uniform (1/P):
```
S_shrunk = w * S_empirical + (1 - w) * (1/P)
w = min(1, n_complete / 10)  # full empirical weight when enough data
```

---

## 2. Theta Method Integration into Ridge

### Theta Method Essence (Hyndman & Billah 2003)

The Theta method won M3. Hyndman proved it equals SES with drift:
```
y_hat(t+h) = l_t + b * h
```
where:
- `l_t` = level from SES with optimized alpha
- `b = slope_OLS / 2` (half the OLS linear trend slope)

The **halved drift** is the key insight: full OLS trend overshoots on short series because it overfits the endpoints. Halving is a form of shrinkage.

### Why Theta Works on Short Series

1. **Implicit regularization of trend**: Halving the slope = shrinking toward zero trend
2. **SES adapts the level**: The SES component tracks recent level changes
3. **No seasonality modeling**: Theta operates on deseasonalized data, avoiding overfitting seasonal patterns when data is short
4. **Implicit combination**: The Theta method combines a linear extrapolation (theta=0 line) with a doubled-curvature line (theta=2 line), which is equivalent to combining a trend model with a level model

### Incorporating Theta's Insight into Ridge

The Ridge framework can replicate Theta's behavior through **differential penalty scales**:

```python
def _penalty_scales_theta_inspired(n_base, n_feat, n_complete):
    """Theta-inspired: shrink trend more than level, shrink Fourier heavily for short series."""
    scales = np.ones(n_feat)
    scales[0] = 0.01          # intercept (level): almost free
    scales[1] = 2.0           # trend: penalize MORE (Theta halves the slope)

    # Fourier: penalty increases as data gets shorter
    fourier_penalty = max(1.0, 10.0 / n_complete)
    col = 2
    for k in range(n_fourier_pairs):
        scales[col] = fourier_penalty * (k + 1)
        scales[col + 1] = fourier_penalty * (k + 1)
        col += 2

    scales[n_base:] = 0.5     # lag features: moderate
    return scales
```

This achieves the Theta effect within Ridge:
- When n_complete is large, Fourier penalty is ~1 -> standard Ridge behavior
- When n_complete is small, Fourier penalty is large -> Ridge shrinks seasonal coefficients toward zero, effectively doing "deseasonalized trend + level" like Theta
- The trend coefficient is always penalized 2x more than default, replicating the "halved slope" effect

### Generalized Theta as Ridge Combination

The Optimised Theta Method (Fiorucci et al. 2015) uses multiple theta-lines with optimized weights. This is equivalent to:

```
y_hat = w_1 * trend_extrapolation + w_2 * SES_forecast
```

In Ridge terms, this is the **ensemble of feature sets** (which FLAIR V6 already does!):
- Config 1: intercept + trend only (= theta-0 line)
- Config 2: intercept + lag-1 only (= SES approximation)
- Config 3: full features (= theta-2 line approximation)

The GCV-weighted soft-average selects the optimal combination.

---

## 3. Forecast Combination (The Ensemble Approach)

### The Forecast Combination Puzzle

Literature consensus (Bates & Granger 1969, Smith & Wallis 2009, recent 2023 survey):
- Simple average of forecasts almost always beats estimated-optimal weights
- Inverse-variance weighting is the best practical alternative to equal weights
- The reason: weight estimation error dominates the gain from optimal weighting

### Bates-Granger Inverse Variance Weights

Given K forecast methods with LOO MSE estimates sigma_k^2:
```
w_k = sigma_k^{-2} / sum_j(sigma_j^{-2})
```

### Ridge-Compatible Combination

FLAIR already has the GCV-weighted ensemble in V6. The question is: **what methods to combine?**

Proposed combination set (all Ridge-based, no if/else):

```python
def combined_forecast(y, horizon, period, freq_str, n_samples):
    """Combine multiple Ridge forecasts with inverse-GCV weights."""

    forecasts = []
    gcv_scores = []

    # (A) Full FLAIR: NLinear + Fourier + lags on Level x Shape
    fc_a, gcv_a = flar_v9(y, horizon, period, freq_str)
    forecasts.append(fc_a)
    gcv_scores.append(gcv_a)

    # (B) Simple Ridge: intercept + trend + lag-1 only (no Fourier, no reshape)
    fc_b, gcv_b = simple_ridge(y, horizon)
    forecasts.append(fc_b)
    gcv_scores.append(gcv_b)

    # (C) Seasonal Naive (constant, always included)
    fc_c = seasonal_naive(y, period, horizon)
    forecasts.append(fc_c)
    gcv_scores.append(gcv_a * 1.5)  # assign moderate GCV (worse than FLAR if FLAR works)

    # Inverse-GCV soft-average
    gcv_arr = np.array(gcv_scores)
    log_w = -(gcv_arr - gcv_arr.min()) / max(gcv_arr.min(), 1e-10)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()

    return sum(wi * fi for wi, fi in zip(w, forecasts))
```

### Why This Helps Short Series

- When FLAIR works well (long series, strong seasonality), GCV for (A) is much lower than (B) or (C), so w_A ≈ 1
- When FLAIR fails (short series, no seasonality), GCV for (A) is high, and the simple Ridge (B) or naive (C) get higher weight
- This is **automatic model selection via continuous weighting**, not discrete if/else

### The "Simple Ridge" Component

A minimal Ridge model that always works:
```python
def simple_ridge(y, horizon, n_samples=20):
    """Ridge on [1, t/n, lag-1] — works for any series length >= 3."""
    n = len(y)
    lam = _bc_lambda(y)
    y_t = _bc(y + 1, lam)

    # NLinear
    last = y_t[-1]
    y_innov = y_t - last

    # Features: intercept + trend + lag-1
    X = np.column_stack([
        np.ones(n - 1),
        np.arange(1, n, dtype=float) / n,
        y_innov[:-1]
    ])
    yt = y_innov[1:]

    beta, loo, gcv = _ridge_gcv_loo_softavg(X, yt)

    # Recursive forecast
    y_ext = np.concatenate([y_innov, np.zeros(horizon)])
    for h in range(horizon):
        ti = n + h
        x = np.array([1.0, ti / n, y_ext[ti - 1]])
        pred = x @ beta
        y_ext[ti] = pred

    fc = np.maximum(_bc_inv(y_ext[n:] + last, lam) - 1, 0.0)
    return fc, gcv
```

---

## 4. Frequency-Adaptive Period Selection

### The Problem

FLAIR assumes `FREQ_TO_PERIOD` maps (e.g., D -> 7, H -> 24). But:
- Many daily series have no weekly pattern (m4_daily: financial data)
- Some hourly series have no daily pattern (bitbrains: server workloads)
- The assumed period can be **wrong**, causing reshape to create noise

### Seasonality Strength Detection

From Hyndman's tsfeatures: `F_s = 1 - Var(R) / Var(S + R)`

where S = seasonal component, R = remainder from STL decomposition.

For FLAIR, a simpler proxy using autocorrelation at the period lag:
```python
def seasonality_strength(y, period):
    """Estimate strength of seasonality at given period. Returns 0-1."""
    if period < 2 or len(y) < 2 * period:
        return 0.0

    n = len(y)
    y_centered = y - np.mean(y)

    # Autocorrelation at lag=period
    if n > period:
        r_p = np.corrcoef(y_centered[period:], y_centered[:-period])[0, 1]
    else:
        r_p = 0.0

    # Also check lag=1 (trend dominance)
    r_1 = np.corrcoef(y_centered[1:], y_centered[:-1])[0, 1]

    # Strength: high autocorrelation at period relative to lag-1
    strength = max(0, r_p)  # negative autocorrelation = no seasonality

    return float(np.clip(strength, 0, 1))
```

### FFT-Based Period Detection

For series where the assumed period might be wrong:
```python
def detect_dominant_period(y, min_period=2, max_period=None):
    """Detect dominant period using FFT periodogram."""
    n = len(y)
    if max_period is None:
        max_period = n // 3

    # Detrend
    y_detrended = y - np.linspace(y[0], y[-1], n)

    # FFT
    fft_vals = np.fft.rfft(y_detrended)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n)

    # Convert to periods and find peak
    valid = (freqs > 0) & (1.0 / freqs >= min_period) & (1.0 / freqs <= max_period)
    if not np.any(valid):
        return 1, 0.0

    periods = 1.0 / freqs[valid]
    powers = power[valid]

    best_idx = np.argmax(powers)
    best_period = int(round(periods[best_idx]))

    # Strength: peak power relative to mean
    strength = powers[best_idx] / (np.mean(powers) + 1e-10)

    return best_period, strength
```

### Adaptive Period Integration into FLAIR

Instead of hard if/else, **continuously weight the reshape vs non-reshape paths**:

```python
def adaptive_flair(y, horizon, period, freq_str, n_samples=20):
    """FLAIR with adaptive period selection."""

    # Detect seasonality strength
    ss = seasonality_strength(y, period)

    # Always compute both paths
    fc_reshape, gcv_reshape = flar_v9_core(y, horizon, period, freq_str)  # Level x Shape
    fc_flat, gcv_flat = simple_ridge(y, horizon)  # No reshape

    # GCV-weighted combination (seasonality strength as prior on reshape)
    # If seasonality is strong, lower the effective GCV of reshape path
    gcv_reshape_adj = gcv_reshape / max(ss, 0.1)  # boost reshape when seasonal
    gcv_flat_adj = gcv_flat / max(1 - ss, 0.1)    # boost flat when non-seasonal

    # Soft combination
    log_w = np.array([
        -gcv_reshape_adj / max(min(gcv_reshape_adj, gcv_flat_adj), 1e-10),
        -gcv_flat_adj / max(min(gcv_reshape_adj, gcv_flat_adj), 1e-10)
    ])
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()

    return w[0] * fc_reshape + w[1] * fc_flat
```

**Better approach** (no adjustment needed): Just let GCV decide naturally. If reshape helps, its GCV will be lower. If not, the flat model wins. The only requirement is that **both paths always run**.

---

## 5. Cross-Learning / Pooled Ridge for Multivariate Series

### The Opportunity

Datasets like bitbrains_fast_storage/H have 2500 series. Currently FLAIR fits each independently. But series within a dataset may share:
- Common periodicity
- Similar trend patterns
- Correlated noise

### Pooled Ridge Formulation

For N series, each with features X_i and targets y_i:

**Local model**: beta_i = (X_i'X_i + alpha*I)^{-1} X_i'y_i (current FLAIR)

**Pooled model**: Stack all series into one regression:
```
X_pool = [X_1; X_2; ...; X_N]    (N*T_i rows, p columns)
y_pool = [y_1; y_2; ...; y_N]
beta_pool = (X_pool'X_pool + alpha*I)^{-1} X_pool'y_pool
```

This shares coefficients across all series. Equivalent to assuming all series have the same dynamics.

### Hierarchical / Empirical Bayes Approach

Better than full pooling: estimate a **population prior** from all series, then use it as the Ridge prior for each series:

```python
def hierarchical_ridge(X_list, y_list, alpha_local):
    """Two-stage hierarchical Ridge across multiple series."""

    # Stage 1: Fit each series locally to get initial estimates
    betas_local = []
    for X_i, y_i in zip(X_list, y_list):
        beta_i = ridge_solve(X_i, y_i, alpha_local)
        betas_local.append(beta_i)

    betas_local = np.array(betas_local)  # (N, p)

    # Stage 2: Compute population mean and variance
    mu_pop = np.mean(betas_local, axis=0)      # (p,) — empirical Bayes prior mean
    var_pop = np.var(betas_local, axis=0) + 1e-8  # (p,) — between-series variance

    # Stage 3: Shrink each local estimate toward population mean
    # James-Stein-like shrinkage
    betas_shrunk = []
    for X_i, y_i, beta_i in zip(X_list, y_list, betas_local):
        n_i = len(y_i)
        sigma2_i = np.mean((y_i - X_i @ beta_i) ** 2)

        # Shrinkage factor per coefficient
        # B_j = var_pop[j] / (var_pop[j] + sigma2_i / n_i)
        B = var_pop / (var_pop + sigma2_i / n_i)

        beta_shrunk = B * beta_i + (1 - B) * mu_pop
        betas_shrunk.append(beta_shrunk)

    return betas_shrunk
```

### Practical Integration with FLAIR

For multivariate datasets (num_variates > 1), FLAIR currently loops over variates independently. The hierarchical approach would:

1. Build features X_i for each variate i
2. Fit local Ridge to get beta_i
3. Compute mu_pop = mean of all beta_i
4. Refit each series with shrinkage toward mu_pop

The shrinkage toward population mean is especially helpful when individual series are **short** but share common patterns with other series in the dataset.

```python
def flar_v9_multivariate_pooled(target_2d, prediction_length, period, freq_str, n_samples):
    """FLAIR V9 with cross-series shrinkage for multivariate data."""
    nv, T = target_2d.shape

    # Phase 1: Fit each series independently, collect betas
    local_results = []
    for v in range(nv):
        result = flar_v9_with_beta(target_2d[v], prediction_length, period, freq_str)
        local_results.append(result)  # (beta, X, y_train, ...)

    # Phase 2: Compute population prior (only if enough series)
    if nv >= 5:
        all_betas = np.array([r['beta'] for r in local_results])
        mu_pop = np.median(all_betas, axis=0)  # robust: median instead of mean
        var_pop = np.var(all_betas, axis=0) + 1e-8

        # Phase 3: Refit with shrinkage
        for v in range(nv):
            r = local_results[v]
            sigma2 = np.mean(r['loo_resid'] ** 2)
            n_i = len(r['y_train'])

            B = var_pop / (var_pop + sigma2 / max(n_i, 1))
            r['beta_shrunk'] = B * r['beta'] + (1 - B) * mu_pop

    # Phase 4: Forecast with (possibly shrunk) betas
    samples = np.zeros((n_samples, prediction_length, nv))
    for v in range(nv):
        beta = local_results[v].get('beta_shrunk', local_results[v]['beta'])
        samples[:, :, v] = forecast_from_beta(beta, local_results[v], n_samples)

    return samples
```

### Computational Cost

Pooled Ridge with SVD: O(N * n * p + p^3) — the SVD dominates, same as local.
Hierarchical Ridge: 2x local cost (two passes), but p^3 is small for FLAIR's feature count (~10-20).

---

## 6. Robust Trend Estimation for Short Noisy Series

### The Problem

OLS trend on short series (n < 30) is unreliable:
- Sensitive to outliers at endpoints
- High variance when n is small
- Recursive forecasting amplifies trend estimation error

### Theil-Sen Estimator

Median of all pairwise slopes:
```
slope = median{ (y_j - y_i) / (j - i) : i < j }
intercept = median{ y_i - slope * i }
```

Properties:
- 29% breakdown point (can handle up to 29% outliers)
- O(n^2) computation for all pairs, O(n log n) with randomized algorithm
- Available in `scipy.stats.theilslopes`

### Repeated Median Regression

Even more robust (50% breakdown point):
```
slope = median_i { median_j { (y_j - y_i) / (j - i) } }
```

### Integration with Ridge

Rather than replacing Ridge's trend estimation, use robust trend as a **detrending step**:

```python
from scipy.stats import theilslopes

def robust_detrend(y):
    """Detrend using Theil-Sen, return detrended series and trend parameters."""
    n = len(y)
    t = np.arange(n)
    slope, intercept, _, _ = theilslopes(y, t)
    trend = intercept + slope * t
    return y - trend, slope, intercept
```

Then Ridge operates on the detrended residuals, which are more stationary and easier to model with Fourier + lags.

**Caution**: This adds a preprocessing step that may conflict with NLinear (which already handles level shifts). The better approach is:

### Ridge as Robust-ified via Penalty

Instead of external robust estimation, increase the **trend penalty** in Ridge for short series:

```python
# In penalty_scales:
trend_penalty = max(1.0, 5.0 / np.sqrt(n_complete))
# n_complete=3 -> penalty=2.9 (heavy shrinkage)
# n_complete=30 -> penalty=0.9 (light shrinkage)
# n_complete=100 -> penalty=0.5 (minimal shrinkage)
```

This achieves robust-like behavior through Ridge's shrinkage, maintaining the single-pass closed-form architecture.

---

## 7. M4 Competition Insights for Short Series

### M4 Yearly (n ~ 13-47, median 25, H=6)

Top methods:
- **Smyl (1st place)**: ES-RNN hybrid — too complex for FLAIR's scope
- **Theta/4Theta (benchmark)**: SES + halved drift — see Section 2
- **Comb (2nd-tier benchmark)**: Equal-weight average of SES, Holt, Damped Holt

**Key finding**: On M4 yearly, MFLES gets MASE 3.956 vs FLAIR 3.613. FLAIR actually wins here. The issue is m4_yearly/A/short has P=1 (annual), so FLAIR falls back to V5 (flat Ridge without reshape), which works.

### M4 Daily (n ~ 93-9933, median 1096, H=14)

- FLAIR MASE: 3.332, MFLES MASE: 3.284 — very close
- Period = 7 (weekly). Many series have no weekly pattern (finance)
- Winners used: combination of Theta + ETS + ARIMA

### M4 Monthly (n ~ 42-2794, median 383, H=18)

- FLAIR MASE: 1.092, MFLES MASE: 1.226 — FLAIR wins clearly
- Period = 12 (annual). Strong seasonality in most series.

### Restaurant/D (n=296, P=7)

- FLAR: 0.822, MFLES: 1.099 — FLAR actually wins
- The quick_analysis shows FLAR loses to ADPC on 46/48 series (ADPC=0.878)

### Car_parts/M (n=51, P=12, but many zeros)

- FLAR: 1.027, MFLES: 1.406 — FLAR wins
- This is intermittent demand. The zeros make Box-Cox + Ridge work better than MFLES.

### Key Takeaway

FLAIR's main weakness is NOT short series per se, but specifically:
1. **Multivariate with diverse patterns** (bitbrains: 2500 series, each needs different treatment)
2. **Non-periodic or weak seasonality** where Fourier features add noise
3. **Very noisy/spiky data** (bitbrains_rnd: 60% spiky)

The fix is: **always include a simple fallback in the GCV ensemble, not just the V5 fallback**.

---

## 8. Unified Architecture: FLAIR v10 Proposal

### Core Idea: "Always Combine, Never Select"

Instead of if/else between V9 (reshape) and V5 (flat), **always run both and let GCV weight them**:

```python
def flar_v10(y_raw, horizon, period, freq_str, n_samples=20):
    """
    FLAIR v10: Always-Combine architecture.

    Three Ridge models, GCV-weighted:
      (A) Level x Shape (V9) — when reshape helps
      (B) Flat Ridge with Fourier + lags — when reshape doesn't help
      (C) Minimal Ridge (intercept + trend + lag-1) — fallback for very short/noisy

    No if/else for model selection. GCV does all selection continuously.
    """
    y = np.maximum(np.nan_to_num(np.asarray(y_raw, float), nan=0.0), 0.0)
    n = len(y)

    cal = get_periods(freq_str)
    P = cal[0] if cal else period
    n_complete = n // P if P >= 2 else 0

    forecasts = []
    gcv_scores = []
    loo_residuals = []

    # --- Path A: Level x Shape (V9) ---
    if P >= 2 and n_complete >= 3:
        fc_a, gcv_a, loo_a = _v9_core(y, horizon, P, cal, n_complete)
        if fc_a is not None:
            forecasts.append(fc_a)
            gcv_scores.append(gcv_a)
            loo_residuals.append(loo_a)

    # --- Path B: Flat Ridge with Fourier + lags (V5-like) ---
    fc_b, gcv_b, loo_b = _flat_ridge(y, horizon, P, cal)
    if fc_b is not None:
        forecasts.append(fc_b)
        gcv_scores.append(gcv_b)
        loo_residuals.append(loo_b)

    # --- Path C: Minimal Ridge (always succeeds for n >= 3) ---
    fc_c, gcv_c, loo_c = _minimal_ridge(y, horizon)
    forecasts.append(fc_c)
    gcv_scores.append(gcv_c)
    loo_residuals.append(loo_c)

    # --- GCV-weighted soft-average ---
    gcv_arr = np.array(gcv_scores)
    log_w = -(gcv_arr - gcv_arr.min()) / max(gcv_arr.min(), 1e-10)
    log_w -= log_w.max()
    w = np.exp(log_w); w /= w.sum()

    point_fc = sum(wi * fi for wi, fi in zip(w, forecasts))
    point_fc = np.maximum(point_fc, 0.0)

    # --- Conformal samples from best model's LOO ---
    best_idx = int(np.argmin(gcv_arr))
    loo_best = loo_residuals[best_idx]
    samples = _conformal_samples(point_fc, loo_best, horizon, n_samples)

    return samples
```

### The Three Models in Detail

**(A) Level x Shape** (existing V9):
- Reshape y into (P, n_complete)
- Shape = proportions from last K periods
- Level = period totals -> NLinear -> Ridge with cross-period Fourier
- Forecast: L_hat * S

**(B) Flat Ridge** (existing V5 but always with NLinear):
- NLinear: y_innov = y_t - y_t[-1]
- Features: [1, t/n, cos/sin(2*pi*t/p) for p in periods, lag-1, lag-P]
- Standard Ridge with GCV soft-average

**(C) Minimal Ridge** (new, always-works fallback):
- NLinear: y_innov = y_t - y_t[-1]
- Features: [1, t/n, lag-1] — only 3 features
- Works for ANY series with n >= 3
- This is Ridge's version of the Theta method: level + trend + AR(1)

### Theta-Inspired Penalty Structure for All Three Models

```python
def _adaptive_penalty_scales(n_feat, n_base, n_observations, n_complete):
    """
    Penalty scales that adapt to data length.
    Short series -> heavier shrinkage on seasonal features.
    Replicates Theta's implicit "halved trend" behavior.
    """
    scales = np.ones(n_feat)

    # Data-adaptive shrinkage factor
    data_ratio = min(1.0, n_observations / 100.0)  # 0 to 1

    # Intercept: always free
    scales[0] = 0.01

    # Trend: penalize more for short series (Theta effect)
    scales[1] = 0.1 + (1.0 - data_ratio) * 2.0  # short: 2.1, long: 0.1

    # Fourier harmonics: heavy penalty when few observations per cycle
    col = 2
    for k in range(n_base - 2):  # Fourier pairs
        harmonic_order = k // 2 + 1
        scales[col] = harmonic_order * (1.0 + (1.0 - data_ratio) * 5.0)
        col += 1

    # Lag features: moderate
    scales[n_base:] = 0.5

    return scales
```

### What This Achieves

| Scenario | Path A (V9) | Path B (V5) | Path C (Minimal) | Expected Winner |
|----------|-------------|-------------|-------------------|-----------------|
| Long + seasonal (electricity/H) | Low GCV | Medium GCV | High GCV | A |
| Medium + seasonal (m4_monthly) | Low GCV | Low GCV | Medium GCV | A or B |
| Short + seasonal (m4_hourly) | Low GCV | Medium GCV | Medium GCV | A |
| Short + no season (m4_yearly) | N/A (P=1) | Medium GCV | Low GCV | C |
| Noisy multivariate (bitbrains) | High GCV | Medium GCV | Low GCV | B or C |
| Strong trend (covid_deaths) | Medium GCV | Low GCV | Low GCV | B or C |

The key is that **no explicit model selection logic is needed**. GCV handles it automatically.

---

## 9. NLinear for Non-Reshaped Series

### DLinear/NLinear from Zeng et al. (2023)

**NLinear**: Subtract last value, apply linear layer, add back.
```
y_hat = Linear(y - y[-1]) + y[-1]
```

**DLinear**: Decompose into trend (moving average) + remainder, two linear layers.
```
trend = MovingAvg(y, kernel_size=25)
remainder = y - trend
y_hat = Linear_trend(trend) + Linear_remainder(remainder)
```

### FLAIR Already Does NLinear

FLAIR V9's `L_innov = L_bc - last_L` is exactly NLinear on the Level series. V5 does not do this on the raw series.

### Recommendation

Apply NLinear to ALL Ridge paths (A, B, C):

```python
# Before building features:
last_val = y_t[-1]
y_innov = y_t - last_val  # NLinear subtraction

# After forecasting:
fc_bc = fc_innov + last_val  # NLinear addition
fc_original = _bc_inv(fc_bc, lam) - 1
```

This is already done in V9 for Level. Ensure V5 and the minimal Ridge also use it.

### When NLinear Fails

NLinear fails when the last value is an outlier. For robustness:
```python
# Use median of last K values instead of just last
K = min(5, n)
last_val = np.median(y_t[-K:])
```

---

## 10. Implementation Roadmap

### Priority 1: Always-Combine Architecture (HIGH IMPACT, LOW RISK)

1. Implement `_minimal_ridge()` — 3-feature fallback
2. Run all three paths (A, B, C) always
3. GCV-weighted soft-average for point forecast
4. Best-model LOO for conformal intervals
5. Test on GIFT-Eval benchmark

### Priority 2: Theta-Inspired Adaptive Penalties (MEDIUM IMPACT, LOW RISK)

1. Implement `_adaptive_penalty_scales()` that scales with data length
2. Heavier trend penalty (halved drift effect)
3. Heavier Fourier penalty for short series
4. Test specifically on m4_yearly, m4_daily, bitbrains

### Priority 3: NLinear on All Paths (LOW RISK)

1. Apply NLinear transformation consistently to V5 path
2. Use robust last value (median of last K)
3. Verify no regression on existing benchmarks

### Priority 4: Cross-Series Pooling for Multivariate (MEDIUM RISK)

1. Implement 2-stage hierarchical Ridge for multivariate
2. James-Stein shrinkage of per-series betas toward population mean
3. Test on bitbrains_fast_storage and bitbrains_rnd

### Priority 5: Seasonality Strength Detection (LOW IMPACT)

1. Implement autocorrelation-based seasonality strength
2. Use as GCV adjustment factor (not hard threshold)
3. Test on datasets with weak/no seasonality

---

## Sources

- [Hyndman & Billah (2003) — Unmasking the Theta Method](https://robjhyndman.com/papers/Theta.pdf)
- [Fiorucci et al. (2015) — The Optimised Theta Method](https://arxiv.org/pdf/1503.03529)
- [Montero-Manso et al. (2020) — Principles and Algorithms for Forecasting Groups of Time Series](https://arxiv.org/pdf/2008.00444)
- [Hewamalage et al. (2021) — Global Models for Time Series Forecasting: A Simulation Study](https://arxiv.org/abs/2012.12485)
- [Zeng et al. (2023) — Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/pdf/2205.13504)
- [Lee & Lee (2023) — Solving the Forecast Combination Puzzle](https://arxiv.org/pdf/2308.05263)
- [Wang et al. (2022) — Forecast Combinations: An Over 50-Year Review](https://arxiv.org/pdf/2205.04216)
- [Hoff (2025) — Shrinkage and Empirical Bayes](https://www2.stat.duke.edu/~pdh10/Teaching/732/Notes/shrinkage.pdf)
- [Efron — Chapter 7: James-Stein Estimation and Ridge Regression](https://efron.ckirby.su.domains/other/CASI_Chap7_Nov2014.pdf)
- [Breheny — Ridge Regression Bayesian Interpretation](https://myweb.uiowa.edu/pbreheny/7240/s21/notes/2-10.pdf)
- [MFLES Documentation — Nixtla](https://nixtlaverse.nixtla.io/statsforecast/docs/models/mfles.html)
- [Hyndman — Measuring Time Series Characteristics](https://robjhyndman.com/hyndsight/tscharacteristics/)
- [M4 Competition — GitHub](https://github.com/Mcompetitions/M4-methods)
- [Theil-Sen Estimator — Wikipedia](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator)
- [L1 Trend Filtering — Boyd et al.](https://web.stanford.edu/~boyd/papers/l1_trend_filter.html)
- [Ridge with Non-Zero Prior — Bayesian Interpretation](https://statisticaloddsandends.wordpress.com/2018/12/29/bayesian-interpretation-of-ridge-regression/)
- [Investigating Cross-Learning Methods — Smyl et al.](https://www.sciencedirect.com/science/article/abs/pii/S0169207020301850)
- [Hansen — Forecast Combination Lecture Notes](https://www.ssc.wisc.edu/~bhansen/390/2010/390Lecture24.pdf)
