# Research: Improving Short-Horizon Forecasting in Cross-Period (Period-Reshape) Frameworks

## Problem Statement

FLAR V7 reshapes time series by primary period P, creating P phase-series of length n=T/P.
Each phase predicts m=ceil(H/P) cross-period steps via shared Ridge regression.

**The short-horizon problem**: When H <= P (e.g., H=24 with P=24), m=1. Each phase predicts
only a single value — the next period's value at that phase position. All within-period
temporal dynamics (hour 0 -> hour 1 -> hour 2 transitions) are lost. The model treats each
phase as completely independent, ignoring the fact that consecutive hours within a day are
strongly correlated.

---

## 1. SparseTSF (ICML 2024 Oral / TPAMI 2025)

### How It Works

SparseTSF uses the same period-reshape approach as FLAR V7:

```
Input:  x ∈ R^L                         (length L)
Step 1: Reshape to (n, w) where n=L/w   (n periods, w phases)
Step 2: Transpose to (w, n)             (w phase-series, each length n)
Step 3: Linear(n → m) per phase         (shared weights across phases)
Step 4: Transpose to (m, w), reshape    (back to forecast length H=m*w)
```

Where `w = period_len` (e.g., 24 for hourly), `n = seq_len // period_len`, `m = pred_len // period_len`.

The core model is a single linear layer of size (n x m) applied identically to each of the w
phase-series. A 1D convolution with kernel size ~ period_len is applied as preprocessing
(residual connection) to smooth outliers before reshaping.

### How SparseTSF Handles Short Horizons

**It does NOT handle short horizons.** SparseTSF was designed explicitly for Long-Term
Time Series Forecasting (LTSF). Key evidence:

1. `seg_num_y = pred_len // period_len` — uses integer division, requiring H >= P
2. When H < P, `seg_num_y = 0`, which makes the linear layer degenerate (output dim 0)
3. All experiments use H in {96, 192, 336, 720} with period=24, so m in {4, 8, 14, 30}
4. The paper explicitly states it "focus[es] on cross-period trend prediction"

**No mechanism for within-period dynamics exists.** The 1D convolution preprocessing
operates on the raw sequence before reshaping, providing local smoothing but not modeling
phase-to-phase transitions. After reshaping, each phase is treated independently.

### TPAMI 2025 Extension

The extended version adds SparseTSF/MLP (dual-layer MLP instead of single linear) and
deeper theoretical analysis of implicit regularization, but does NOT address the
short-horizon limitation. The method remains focused on LTSF.

### Key Insight for FLAR

SparseTSF confirms that the period-reshape approach is fundamentally designed for H >> P.
The short-horizon problem is intrinsic to the reshape architecture — not a bug but a
design assumption. Any fix must add something outside the reshape framework.

---

## 2. PatchTST / Patching Approaches (ICLR 2023)

### How Patching Relates to Period-Reshaping

PatchTST divides time series into overlapping or non-overlapping patches of fixed length P_patch,
then treats each patch as a token for a Transformer.

**Key difference from period-reshaping**: Patches are *sequential* segments, NOT downsampled
phase-series. A patch of length 24 from hourly data contains hours [0,1,...,23] of one day,
while period-reshaping creates 24 separate series of [day0_h0, day1_h0, day2_h0, ...].

```
Period-reshape:  Phase 0: [h0_day0, h0_day1, h0_day2, ...]  (same hour across days)
                 Phase 1: [h1_day0, h1_day1, h1_day2, ...]
Patching:        Patch 0: [h0_day0, h1_day0, h2_day0, ...]  (all hours of one day)
                 Patch 1: [h0_day1, h1_day1, h2_day1, ...]
```

### Can Patches Capture Both Dimensions?

PatchTST benefits from "local semantic information retained in the embedding" — each patch
preserves the within-period shape. However, cross-period dynamics (how Monday's pattern
differs from Tuesday's) are captured only through attention over the sequence of patch tokens.

**The dual problem**: Period-reshaping captures cross-period well but loses within-period.
Patching captures within-period well but relies on attention for cross-period.

### Insight for FLAR: Patch Features

For Ridge regression, we cannot use attention. But we CAN extract patch-level features:

**Approach**: For each period (day), compute summary statistics of the within-period shape
and use these as additional features in the cross-period Ridge:

```python
# For phase j's cross-period series, add features from the within-period context
# E.g., for hour 14, include "what happened at hours 12,13" of the same period
within_period_features = [
    y[period_i, phase_j - 1],   # previous hour's value in same day
    y[period_i, phase_j - 2],   # two hours back
    mean(y[period_i, :phase_j]), # running mean up to this hour
]
```

This is the "phase-coupling" approach detailed in Section 5 below.

---

## 3. Multi-Resolution / Multi-Scale Approaches

### TimeMixer (ICLR 2024)

TimeMixer operates at M scales simultaneously via average-pooling downsampling:

```
Scale 0: original series x_0 ∈ R^T          (finest)
Scale 1: avg-pool by 2,  x_1 ∈ R^(T/2)
Scale 2: avg-pool by 4,  x_2 ∈ R^(T/4)
...
Scale M: avg-pool by 2^M, x_M ∈ R^(T/2^M)  (coarsest)
```

**Past-Decomposable-Mixing (PDM)**: Decomposes each scale into seasonal + trend, then:
- Seasonal: mixes bottom-up (fine -> coarse) — aggregating detailed periodicity
- Trend: mixes top-down (coarse -> fine) — propagating macro-level trends

**Future-Multipredictor-Mixing (FMM)**: Independent predictors per scale, summed:
`forecast = sum(predictor_m(x_m) for m in scales)`

### MICN (ICLR 2023)

Uses multi-scale isometric convolutions with local-global context. Decomposes into
seasonal + trend via multi-scale hybrid decomposition. Each scale captures different
pattern granularity.

### Adaptation to Ridge Regression: Multi-Scale Ensemble

The TimeMixer insight can be adapted to FLAR without neural networks:

**Approach: Dual-Scale Ridge Ensemble**

```
Scale A (cross-period): Period-reshape, Ridge on phase-series → m cross-period predictions
Scale B (within-period): Full-resolution Ridge with Fourier features → H direct predictions

Final = w_A * Scale_A + w_B * Scale_B    (GCV-weighted)
```

For short horizons (H <= P), Scale B dominates because it directly models hour-to-hour
transitions. For long horizons (H >> P), Scale A dominates because recursive error doesn't
accumulate (m is small).

**Mathematical formulation:**

```
# Scale A: Cross-period (existing FLAR V7)
y_A = reshape(ridge_cross_period(phases, features_A))  # (P, m) -> (H,)

# Scale B: Full-resolution (FLAR V5-style)
y_B = ridge_full_resolution(y_raw, features_B)          # H recursive steps

# Combine with GCV-based weights
gcv_A = GCV(ridge_A)
gcv_B = GCV(ridge_B)
w = softmax(-[gcv_A, gcv_B] / min(gcv_A, gcv_B))
forecast = w[0] * y_A + w[1] * y_B
```

This is essentially what FLAR V7 already does with its V5 fallback, but currently it's
binary (V7 if enough periods, else V5). Making it a **weighted blend** would be more
principled.

---

## 4. Dual-Axis / 2D Approaches

### GridTST (2024)

GridTST treats multivariate time series as a 2D grid:
- X-axis: time patches (sequential segments)
- Y-axis: variates (different channels)

Applies alternating horizontal (temporal) and vertical (cross-variate) self-attention.
This is analogous to treating (phase x period) as a 2D matrix and modeling both axes.

### Crossformer (ICLR 2023)

Uses Two-Stage Attention (TSA): Cross-Time attention within each variate, then
Cross-Dimension attention across variates at each time step.

### iTransformer (ICLR 2024 Spotlight)

Inverts the standard approach: embeds each variate as a token, uses attention for
cross-variate correlations and FFN for temporal representation.

### 2D ARIMAX for Cohort Data (2025)

The 2D ARIMAX model treats data as a matrix with:
- Rows: cohort index (analogous to period index)
- Columns: time-since-event (analogous to phase)

Key formula: `Y_{t,u} = mu_u + phi*Y_{t-1,u} + beta*Y_hat_{t,u-1} + epsilon`

The term `beta*Y_hat_{t,u-1}` is crucial: it uses the **previous column's prediction**
as a feature for the current column. In period-phase terms, this means using
**phase (j-1)'s predicted value** to inform phase j's prediction.

### Adaptation to Ridge: 2D Ridge with Phase-Coupling

The 2D ARIMAX insight translates directly to Ridge regression:

**Approach: Sequentially predict phases within each period**

Instead of predicting all P phases independently, predict them sequentially:

```
For period (n+1):  (the forecast period)
  Phase 0: predict using cross-period features only
  Phase 1: predict using cross-period features + Phase 0's prediction
  Phase 2: predict using cross-period features + Phase 1's prediction
  ...
  Phase j: predict using cross-period features + Phase (j-1)'s prediction
```

This requires TWO sets of features in the Ridge:
1. Cross-period features: [trend, Fourier, lag-1_cross, lag-P_cross]
2. Within-period feature: [previous_phase_value]

The mathematical formulation is in Section 5.

---

## 5. Within-Period Features for Cross-Period Models (Phase-Coupling)

### The Core Idea

Currently, FLAR V7 fits one shared Ridge where each phase is independent:

```
For phase j: z_j[t+1] = beta^T * [1, trend, fourier, z_j[t], z_j[t-cp]] + epsilon
```

To capture within-period dynamics, add the **adjacent phase's value** as a feature:

```
For phase j: z_j[t+1] = beta^T * [1, trend, fourier, z_j[t], z_j[t-cp], z_{j-1}[t+1]] + epsilon
```

where `z_{j-1}[t+1]` is the value of the previous phase at the SAME period.

### Mathematical Formulation

Let the reshaped matrix be M ∈ R^(P x n) where M[j, t] = y[t*P + j].

After NLinear normalization: `Z[j, t] = M[j, t] - M[j, n-1]`

**Standard (phase-independent) features for phase j at cross-period time t:**

```
x_standard = [1, t/n, cos(2*pi*t/cp), sin(2*pi*t/cp), Z[j, t-1], Z[j, t-cp]]
```

**Phase-coupled features (Option A: use previous phase from same period):**

```
x_coupled = [x_standard, M[(j-1) mod P, t]]     # raw value of adjacent phase
```

But there is a subtlety: during training, `M[(j-1) mod P, t]` is observed. During
forecasting of the NEXT period, `M[(j-1) mod P, n]` is either:
- Known (if we're predicting within the first forecast period and j > 0, because
  phase j-1 was already predicted), OR
- Unknown (if j = 0, or if we're in period n+2, n+3, ...)

**This creates a natural sequential prediction scheme within each forecast period:**

```python
def predict_period(period_idx, previous_predictions):
    predictions = np.zeros(P)
    for j in range(P):
        x = cross_period_features(phase=j, t=period_idx)
        if j > 0:
            # Within-period coupling: use the just-predicted phase
            x = np.append(x, predictions[j-1])
        else:
            # Phase 0: use last known value of phase P-1 from previous period
            x = np.append(x, previous_predictions[P-1] if previous_predictions is not None
                          else last_known[P-1])
        predictions[j] = x @ beta
    return predictions
```

### Training the Phase-Coupled Ridge

During training, the adjacent-phase feature is always observed:

```python
# Build training data with phase-coupling
X_all = []
y_all = []
for j in range(P):
    for t in range(start, n_complete):
        features = [1, t/n, ...]              # standard features
        features.append(Z[j, t-1])            # lag-1 cross-period
        features.append(Z[(j-1) % P, t])      # adjacent phase (SAME period)
        X_all.append(features)
        y_all.append(Z[j, t])
```

The Ridge coefficient for the adjacent-phase feature will learn the typical
hour-to-hour transition pattern.

### Practical Considerations

1. **Phase 0 circularity**: Phase 0 uses phase (P-1) from the same period. During
   training this is observed. During forecasting of the next period, phase (P-1) of
   the current period is the last prediction of the previous period — introducing
   a recursive dependency between periods.

2. **Error propagation**: Sequential within-period prediction means errors cascade
   from phase 0 to phase P-1. For short horizons (m=1) this is at most P steps of
   cascade, which is acceptable. For long horizons (m >> 1), the cross-period
   recursion (m steps) is much smaller than the within-period chain (P steps),
   but each period's within-period chain is bounded by P.

3. **Feature normalization**: The adjacent-phase feature should be in the same space
   as the target (innovation space after NLinear subtraction).

### Expected Benefit

For H=24 with P=24: instead of m=1 independent predictions per phase (no temporal
structure), we get a sequential chain of 24 predictions where each hour uses the
previous hour. This directly models diurnal transitions.

For H=168 with P=24: m=7. The cross-period recursion is 7 steps (good, as before).
Within each of the 7 predicted days, the 24 hours are sequentially linked (added benefit).

---

## 6. Interpolation / Temporal Disaggregation Approaches

### The Idea

After cross-period prediction gives m values per phase, we have a complete (P x m) matrix
of coarse predictions. We can then use the most recent observed within-period "shape" to
interpolate/adjust within each predicted period.

### Approach A: Chow-Lin Style Disaggregation

From temporal disaggregation theory (Chow & Lin 1971):

```
y_high = X * beta_hat + Q * C^T * (y_low - C * X * beta_hat)
```

Where:
- y_low: low-frequency aggregates (cross-period predictions)
- y_high: desired high-frequency values (hourly predictions)
- C: aggregation matrix (maps high-freq to low-freq)
- X: high-frequency indicator series
- Q: covariance matrix (e.g., AR(1))

In our context:
- y_low = cross-period predictions for each phase (m values per phase)
- The "high-frequency indicator" is the within-period shape from recent history
- C = identity (each phase value is already high-frequency)

This doesn't directly apply because our phases ARE the high-frequency values.
Temporal disaggregation is more relevant when predicting daily totals and wanting
to distribute to hourly values.

### Approach B: Shape-Based Interpolation (More Relevant)

**The key insight**: After cross-period Ridge predicts the "level" at each phase
for the next period, use the recent within-period "shape" as a template.

```
Step 1: Cross-period Ridge predicts level_j for each phase j in forecast period
Step 2: Compute the within-period shape from the last K observed periods:
        shape_j = mean(y[t, j] / mean(y[t, :])) for t in last K periods
Step 3: Adjust: forecast_j = level_j * shape_j / mean(shape)
```

But this is essentially what the period-reshape already does — each phase has its
own level prediction. The "shape" is implicit in the different levels across phases.

### Approach C: Denton Proportional Smoothing

More useful application: if cross-period predictions have discontinuities at
period boundaries, apply Denton smoothing:

```
minimize sum((y_smooth[t] / y_raw[t] - y_smooth[t-1] / y_raw[t-1])^2)
subject to: sum(y_smooth[k*P:(k+1)*P]) = sum(y_predicted[k*P:(k+1)*P])
```

Where y_raw is a "template" series (e.g., the most recent period's shape tiled).
This ensures smooth transitions at period boundaries while preserving period totals.

### Practical Formulation for FLAR

```python
def denton_smooth(cross_period_forecast, recent_shape, P, m):
    """
    cross_period_forecast: (P, m) matrix of phase predictions
    recent_shape: (P,) vector of within-period proportions from last period
    """
    # Tile recent shape as template
    template = np.tile(recent_shape, m)  # (P*m,)

    # Flatten cross-period forecast
    raw_forecast = cross_period_forecast.T.reshape(-1)  # (P*m,)

    # Period totals to preserve
    period_totals = [cross_period_forecast[:, k].sum() for k in range(m)]

    # Denton proportional: adjust template to match period totals
    smoothed = np.zeros(P * m)
    for k in range(m):
        seg = template[k*P:(k+1)*P]
        target_sum = period_totals[k]
        smoothed[k*P:(k+1)*P] = seg * (target_sum / max(seg.sum(), 1e-10))

    return smoothed
```

### Assessment

Interpolation is LESS useful for the short-horizon problem than phase-coupling
(Section 5) because:
- The cross-period predictions already produce one value per phase
- The problem isn't interpolation resolution — it's that the phase predictions
  don't capture hour-to-hour transitions
- However, Denton smoothing IS useful for period-boundary discontinuities in
  long-horizon predictions

---

## 7. Residual Correction: Cross-Period Level x Within-Period Shape

### The Core Idea

Classical seasonal decomposition: `y[t] = Level[t] * Seasonal[t] + Residual[t]`

Apply this to cross-period forecasting:

```
forecast[t*P + j] = cross_period_level[j, t] * within_period_shape[j]
```

Where:
- `cross_period_level[j, t]`: predicted by Ridge for phase j at cross-period time t
- `within_period_shape[j]`: the typical relative proportion of phase j within a period

### Formulation: Two-Stage Multiplicative Model

**Stage 1: Cross-period Ridge (existing FLAR V7)**

Predict deseasonalized levels per phase:
```
level_j[t] = beta^T * x_j[t]   (cross-period features for phase j)
```

**Stage 2: Within-period shape from recent history**

Estimate the within-period shape as proportional weights:
```
shape[j] = (1/K) * sum_{k=1}^{K} y[n-k, j] / mean_j(y[n-k, :])
```

where K is the number of recent periods to average over.

**Combined forecast:**
```
forecast[j, t] = level_j[t] * shape[j]
```

### More Sophisticated: Adaptive Shape via Ridge

Instead of a fixed shape, learn the shape correction as a function of recent history:

```python
# Stage 1: Cross-period Ridge predicts period-level
period_level[t] = mean_j(y[t, :])  # target: average across phases
# Features: [trend, cross-period Fourier, lag-1, lag-cp]
# Ridge: period_level_hat[t] = beta_1^T * x_1[t]

# Stage 2: Within-period Ridge predicts phase ratios
ratio[j, t] = y[t, j] / period_level[t]  # target: phase j's ratio
# Features: [phase_indicator, recent_ratio_lag1, ratio_lag_cp, day_of_week]
# Ridge: ratio_hat[j, t] = beta_2^T * x_2[j, t]

# Forecast
forecast[j, t] = period_level_hat[t] * ratio_hat[j, t]
```

### Two-Stage Framework (ICASSP 2021 Insight)

From the "Two-Stage Framework for Seasonal Time Series Forecasting":
1. First stage learns long-range structure beyond the forecast horizon
2. Second stage enhances short-range accuracy using first-stage results

Applied to FLAR:
1. First stage: cross-period Ridge captures multi-period trends (long-range)
2. Second stage: within-period Ridge captures hour-to-hour dynamics (short-range),
   using first-stage predictions as input features

### Mathematical Elegance: Tensor Ridge

The most elegant formulation treats the (phase x period) matrix as a 2D signal
and applies a ridge regression with Kronecker-structured features:

```
vec(Y) = (X_cross kron X_within) * vec(B) + epsilon
```

Where:
- X_cross: cross-period features (n x p_cross)
- X_within: within-period features (P x p_within)
- B: coefficient matrix (p_cross x p_within)
- kron: Kronecker product

This naturally captures interactions between cross-period trends and
within-period patterns. The Ridge penalty on vec(B) shrinks all coefficients.

However, Kronecker structure requires that cross-period and within-period effects
are separable (multiplicative), which may not always hold.

---

## Summary of Implementable Techniques (Ranked by Elegance and Expected Impact)

### Technique 1: Phase-Coupled Sequential Ridge (RECOMMENDED)

**What**: Add adjacent-phase value as a feature in the shared Ridge. During forecasting,
predict phases sequentially within each period.

**Why**: Directly addresses the fundamental problem (independent phases). Minimal code
change. Maintains single Ridge with one SVD. The phase-coupling coefficient is
automatically learned.

**Expected impact**: Large for short horizons (H <= P), moderate for medium horizons,
neutral for long horizons.

**Implementation complexity**: Low. Add one feature column per training sample.
Change forecast loop to sequential within each period.

```python
# Training: add Z[(j-1) % P, t] as feature for phase j at time t
# Forecasting: predict phase 0, then phase 1 using phase 0's prediction, etc.
```

### Technique 2: GCV-Weighted Dual-Scale Blend

**What**: Instead of binary V7-or-V5 fallback, always compute BOTH cross-period (V7) and
full-resolution (V5) forecasts, combine with GCV-based soft weights.

**Why**: For short horizons, V5's full-resolution recursive prediction naturally captures
hour-to-hour dynamics. V7's cross-period prediction captures level well. The blend
gets the best of both.

**Expected impact**: Moderate across all horizons. Robustness improvement.

**Implementation complexity**: Low. Both models are already implemented. Just always
run both and blend.

```python
fc_v7 = flar_v7(y, H, P, freq)
fc_v5 = flar_v5(y, H, P, freq)
w = softmax(-[gcv_v7, gcv_v5] / min(gcv_v7, gcv_v5))
forecast = w[0] * fc_v7 + w[1] * fc_v5
```

### Technique 3: Two-Stage Level x Shape

**What**: Stage 1 predicts period-level via cross-period Ridge. Stage 2 predicts
within-period shape proportions via a second Ridge on phase ratios.

**Why**: Separates the cross-period trend question ("what's the overall level?")
from the within-period shape question ("how is it distributed across hours?").

**Expected impact**: Moderate for short horizons. Particularly useful when the
within-period shape varies (e.g., weekday vs weekend patterns).

**Implementation complexity**: Medium. Requires a second Ridge fit on ratio data.

### Technique 4: Multi-Phase Lag Features

**What**: Instead of just one adjacent phase, include multiple within-period lag features:
z_{j-1}[t], z_{j-2}[t], z_{j-P/2}[t] (e.g., "12 hours ago in same period").

**Why**: Richer within-period context. The shared Ridge can learn complex intra-day
patterns from these features.

**Expected impact**: Moderate. Risk of overfitting if too many features added.

**Implementation complexity**: Low. Just more feature columns.

### Technique 5: Denton Boundary Smoothing (Post-Processing)

**What**: After cross-period predictions, apply Denton proportional smoothing using
the recent period's shape as template, preserving period totals.

**Why**: Eliminates discontinuities at period boundaries (e.g., hour 23 -> hour 0
transitions). Particularly useful for long horizons with multiple predicted periods.

**Expected impact**: Small for short horizons, moderate for long horizons.

**Implementation complexity**: Low. Pure post-processing, no retraining needed.

---

## Recommended Implementation Order

1. **Phase-Coupled Sequential Ridge** (Technique 1) — highest expected ROI
2. **GCV-Weighted Dual-Scale Blend** (Technique 2) — safety net for all horizons
3. **Two-Stage Level x Shape** (Technique 3) — if within-period shape varies significantly
4. Techniques 4 and 5 as refinements

---

## Sources

- [SparseTSF (ICML 2024)](https://arxiv.org/abs/2405.00946) — period-reshape architecture, cross-period sparse forecasting
- [SparseTSF GitHub](https://github.com/lss-1138/SparseTSF) — source code confirming seg_num_y = pred_len // period_len
- [PatchTST (ICLR 2023)](https://arxiv.org/abs/2211.14730) — patching approach, local semantic preservation
- [TimeMixer (ICLR 2024)](https://arxiv.org/abs/2405.14616) — multi-scale mixing, PDM fine-to-coarse + coarse-to-fine
- [GridTST](https://arxiv.org/abs/2405.13810) — 2D grid structure, horizontal + vertical attention
- [Crossformer (ICLR 2023)](https://openreview.net/forum?id=vSVLM2j9eie) — Two-Stage Attention for cross-time + cross-dimension
- [iTransformer (ICLR 2024)](https://arxiv.org/abs/2310.06625) — inverted axis: variate tokens + cross-time FFN
- [2D ARIMAX for Cohort Data](https://arxiv.org/html/2508.15369v1) — 2D matrix model with column-to-column coupling
- [Optimized Ridge for Fixed-Period TS](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002971) — cross-period accumulation + seasonal indices in Ridge
- [Two-Stage Seasonal Forecasting (ICASSP 2021)](https://arxiv.org/abs/2103.02144) — long-range structure stage + short-range refinement stage
- [Temporal Disaggregation (tempdisagg)](https://cran.r-project.org/web/packages/tempdisagg/vignettes/intro.html) — Chow-Lin / Denton methods for frequency conversion
- [Temporal Disaggregation Python](https://arxiv.org/html/2503.22054v1) — mathematical formulation of Chow-Lin GLS + Denton smoothing
- [NLinear/DLinear (AAAI 2023)](https://github.com/cure-lab/LTSF-Linear) — subtraction normalization, channel independence
- [Hyndman: Complex Seasonality](https://robjhyndman.com/publications/complex-seasonality/) — multi-seasonal exponential smoothing
- [Time-Varying Parameters as Ridge Regressions](https://arxiv.org/html/2009.00401v4) — TVP-Ridge connection
