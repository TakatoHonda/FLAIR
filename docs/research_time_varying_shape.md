# Research: Time-Varying Shape in FLAIR

## Problem Statement

FLAIR decomposes a time series by reshaping by period P into a (P x n_periods) matrix, then factorizes:

```
y(phase, period) = Level(period) x Shape(phase)
```

where Shape is a FIXED vector (average proportions from last K periods). The problem: Shape changes over time. E.g., weekday traffic has morning/evening peaks but weekend traffic has a noon peak. A fixed Shape averages these, producing incorrect forecasts for both.

We tried rank-r SVD (multiple shapes), but rank selection requires if/else (ugly).

**Goal**: Find a method where Shape varies over time, is closed-form, embeds in a single Ridge regression, requires no if/else or thresholds, and naturally reduces to fixed Shape when variation is absent.

---

## 1. Time-Varying Coefficients via Weighted Ridge (EWMA Ridge)

### Core Idea

Instead of fitting Ridge to all observations equally, use **exponentially decaying observation weights** so that recent periods dominate. This makes the effective Shape time-varying because the Ridge solution adapts to recent data.

### Mathematical Formulation

Standard Ridge:
```
beta = argmin ||y - X*beta||^2 + lambda*||beta||^2
     = (X'X + lambda*I)^{-1} X'y
```

Weighted Ridge with exponential decay:
```
beta = (X'WX + lambda*I)^{-1} X'Wy
```

where W = diag(rho^{n-1}, rho^{n-2}, ..., rho^0) and rho in (0, 1) is the decay factor.

### Relationship: half-life H = -log(2) / log(rho)

If we want observations from H periods ago to have half the weight:
```
rho = 2^{-1/H}
```

### Parameter-Free Choice for rho

Tie decay to n_complete (number of complete periods):
```
H = n_complete / 2    (half-life = half the available data)
rho = 2^{-2/n_complete}
```

This means: with 10 periods of data, observations 5 periods ago have half weight. With 100 periods, observations 50 periods ago have half weight. The formula is self-adjusting.

### Application to FLAIR

Currently FLAIR builds X (features) and y (Level innovations) and solves Ridge. Simply insert the weight matrix W:

```python
# Weight vector: exponential decay, most recent = 1.0
w = rho ** np.arange(n_train - 1, -1, -1)  # [rho^{n-1}, ..., rho^0]
W_sqrt = np.sqrt(w)

# Transform: weighted Ridge = standard Ridge on (W^{1/2} X, W^{1/2} y)
Xw = X * W_sqrt[:, None]
yw = y_target * W_sqrt

beta, loo_resid, _ = _ridge_sa(Xw, yw)
```

This is **zero additional parameters** (rho is derived from n_complete), **closed-form** (same SVD-based Ridge), and **naturally reduces to fixed weights** when rho -> 1 (no decay = uniform weights).

### Verdict

**Highly promising.** Elegant, parameter-free, zero code complexity increase. But this only makes Level time-varying, not Shape directly. To make Shape time-varying, we'd need to apply the same principle to the Shape computation:

```python
# Exponentially weighted Shape
w_shape = rho_shape ** np.arange(n_complete - 1, -1, -1)
w_shape /= w_shape.sum()
# Weighted proportions
S = (proportions * w_shape[None, :]).sum(axis=1)
S /= S.sum()
```

This replaces the simple average of last K periods with an exponentially weighted average over ALL periods.

---

## 2. Conditional Shape via Indicator Interactions (Prophet-Style)

### Core Idea (Prophet)

Prophet handles conditional seasonality by multiplying Fourier basis functions by indicator variables:

```
s(t) = s_weekday(t) * I(weekday) + s_weekend(t) * I(weekend)
```

where each s_i(t) is a separate Fourier series with its own coefficients.

### Embedding in Ridge Without if/else

Instead of conditional logic, use **multiplicative interaction features**. For FLAIR's matrix formulation, the "context" is the period index's position in a higher-level cycle (e.g., which day of the week for hourly data with P=24).

Define context indicators z_j(period) for j = 1,...,C (e.g., C=7 for day-of-week):

```
y(phase, period) = Level(period) * [Shape_base(phase) + sum_j z_j(period) * Delta_j(phase)]
```

In Ridge form, for each observation (phase i, period k):

```
y_{ik} / Level_k = S_base_i + sum_j z_j(k) * Delta_j_i
```

The feature vector for observation (i, k) is:
```
x = [e_i, z_1(k)*e_i, z_2(k)*e_i, ..., z_{C-1}(k)*e_i]
```

where e_i is the i-th standard basis vector (one-hot for phase).

### Feature Count

This gives P + (C-1)*P features. For hourly data (P=24, C=7): 24 + 6*24 = 168 features. For n_complete >= 30 or so, this is fine for Ridge. But for short series it's too many.

### Compact Fourier Variant

Instead of one-hot phase encoding, use Fourier basis for Shape and context indicators for modulation:

```
Shape(phase, context) = sum_{k=1}^{K} [a_k + sum_j z_j * da_{kj}] * cos(2*pi*k*phase/P)
                       + [b_k + sum_j z_j * db_{kj}] * sin(2*pi*k*phase/P)
```

With K=2 Fourier harmonics and C=7 day-of-week:
- Base: 2*K = 4 features
- Interactions: 2*K*(C-1) = 24 features
- Total: 28 features (vs 168 for one-hot)

### Application to FLAIR

This fundamentally changes FLAIR's architecture. Instead of separate Shape estimation and Level Ridge, we'd do a SINGLE Ridge on the full y(phase, period) matrix:

```python
# For each cell (i, k) in the (P x n_complete) matrix:
# target: y[i, k]
# features: [Level_features(k)] x [Shape_features(i, context(k))]

# Kronecker-style: features are the outer product of Level and Shape features
```

The challenge: this requires knowing the context variable (day-of-week etc.), which needs calendar information beyond just the frequency string.

### Verdict

**Mathematically clean but requires calendar metadata.** For datasets where we know the start date and can compute day-of-week, this is powerful. But FLAIR currently only receives (y, horizon, freq) with no timestamp information, making this approach impractical without API changes.

---

## 3. Exponentially Smoothed Shape (Holt-Winters Style)

### Core Idea

Holt-Winters updates seasonal indices with exponential smoothing:

```
s_{t+m}^{new} = gamma * (y_t / ell_t) + (1 - gamma) * s_t
```

where gamma controls how fast seasonal indices adapt.

### Parameter-Free FLAIR Version

Instead of Shape = simple average of last K period proportions, use exponentially weighted proportions:

```python
# Current FLAIR:
K = min(5, n_complete)
S = proportions[:, -K:].mean(axis=1)

# Proposed: exponentially weighted over ALL periods
alpha = 1 - 2^{-1/max(n_complete/4, 1)}   # half-life = n_complete/4
weights = alpha * (1 - alpha)^{arange(n_complete-1, -1, -1)}
weights /= weights.sum()
S = (proportions * weights[None, :]).sum(axis=1)
S /= S.sum()
```

### Mathematical Properties

- When alpha -> 0 (long series, slow decay): S -> uniform average of all periods = fixed Shape
- When alpha -> 1 (short series, fast decay): S -> most recent period's proportions
- Half-life tied to n_complete/4: with 20 periods, observations 5 periods ago have half weight
- **Completely parameter-free**: alpha is determined by n_complete
- **Closed-form**: just a weighted average

### Connection to Holt-Winters

This is exactly the Holt-Winters seasonal smoothing equation, but:
1. Applied in batch (not recursively), using the closed-form exponential weights
2. The smoothing parameter gamma is not optimized but derived from data length
3. The "Level" denominator is our mat.sum(axis=0) per-period totals

### Elegant Property: Graceful Degradation

When the seasonal pattern is constant across all periods, exponential weighting and uniform weighting produce the same result (since all proportions are identical). So this **naturally reduces to fixed Shape** when variation is absent.

### Verdict

**Extremely promising. Simplest possible change to FLAIR.** One line of code changes the Shape computation. Parameter-free. Closed-form. Naturally degrades to fixed Shape. The only question is whether the half-life heuristic is good enough.

---

## 4. Shape as f(Level) via Interaction Terms in Ridge

### Core Idea

If busy days have different intra-day patterns than quiet days, Shape depends on Level. Model this as:

```
y(phase, period) = Level(period) * [Shape_base(phase) + Level(period) * Shape_delta(phase)]
```

This means: the proportions have a base pattern plus a Level-dependent adjustment.

### Ridge Formulation

Divide both sides by Level:

```
y(i, k) / L_k = S_base_i + L_k * S_delta_i
```

For each cell (i, k), the features are:
```
x = [e_i, L_k * e_i]
```

where e_i is a one-hot vector for phase i.

This is 2P features. For P=24: 48 features. Very manageable.

### Compact Fourier Version

```
Shape(phase, L) = sum_{k=1}^{K} [a_k + L * da_k] * cos(2*pi*k*phase/P)
                 + [b_k + L * db_k] * sin(2*pi*k*phase/P)
```

With K=3: only 12 features total.

### Integration with FLAIR's Level Ridge

The key insight: we can embed this in FLAIR's existing Ridge by changing what we predict. Instead of predicting Level totals and then multiplying by fixed Shape, predict all P*n_complete cells simultaneously:

```python
# Target: full matrix flattened
y_target = mat.flatten()  # P * n_complete values

# Features for cell (i, k):
# [Level_features(k)] * [1, L_normalized(k)] * [Fourier(i)]
```

But this changes the architecture from "predict Level, then multiply by Shape" to "predict everything jointly." The single Ridge regression handles both Level and Shape simultaneously.

### Smooth Transition Variant (STAR-Model Inspired)

Instead of a linear dependence Shape = f(Level), use a sigmoid transition:

```
Shape(phase, L) = Shape_low(phase) * (1 - G(L)) + Shape_high(phase) * G(L)
```

where G(L) = 1 / (1 + exp(-gamma * (L - c))) is a logistic function.

Problem: G introduces two parameters (gamma, c) that need optimization. Not closed-form in a single Ridge. However, if we fix c = median(Level) and gamma = 1/std(Level), it becomes parameter-free, but G(L) is nonlinear in L.

**Trick**: Expand G(L) as a feature. Precompute G(L_k) for each period k, then use it as a known feature in Ridge:

```python
c = np.median(L)
gamma_star = 1.0 / max(np.std(L), 1e-10)
G = 1.0 / (1.0 + np.exp(-gamma_star * (L - c)))

# Features for Shape estimation:
# x = [(1-G_k) * e_i, G_k * e_i]  for cell (i, k)
```

This is still 2P features, still a single Ridge, and the sigmoid transition makes it truly regime-switching without if/else.

### Verdict

**Elegant for Level-dependent seasonality.** The Fourier variant keeps features compact. The sigmoid variant gives true regime switching within Ridge. The question is whether Level is actually a good predictor of Shape variation (it often is for retail/traffic data, but not always).

---

## 5. Fourier Interaction Terms (Context x Phase)

### Core Idea

Instead of a fixed Shape vector, make Shape a function of both phase AND cross-period position using Fourier interactions. The within-period Fourier basis captures Shape; the cross-period Fourier basis captures how Shape varies over time.

### Mathematical Formulation

Let phi_j(i) = Fourier basis at phase i (within period), and psi_m(k) = Fourier basis at period k (cross-period). The model is:

```
y(i, k) = Level(k) * sum_{j,m} c_{jm} * phi_j(i) * psi_m(k)
```

Dividing by Level:

```
y(i, k) / L_k = sum_{j,m} c_{jm} * phi_j(i) * psi_m(k)
```

The features for cell (i, k) are the outer product phi(i) x psi(k).

### Feature Count Analysis

- Within-period: J Fourier pairs -> 2J+1 features (including constant)
- Cross-period: M Fourier pairs -> 2M+1 features
- Interaction: (2J+1) * (2M+1) features

With J=2 (captures main Shape), M=1 (captures one cycle of Shape variation):
- (5) * (3) = 15 features

With J=3, M=2:
- (7) * (5) = 35 features

**This is very manageable** and does NOT grow explosively because both J and M are small.

### Why This Works

The interaction phi_j(i) * psi_m(k) allows each Fourier component of Shape to vary sinusoidally across periods. This captures:
- Shape that rotates slowly (e.g., peak shifts from morning to afternoon over weeks)
- Shape amplitude that varies cyclically (e.g., weekday vs weekend within P=24)
- Shape that is constant (when cross-period Fourier coefficients are zero)

### Ridge Naturally Regularizes

If Shape variation is absent, Ridge will shrink the interaction coefficients c_{jm} (m > 0) to near zero, leaving only c_{j0} (the fixed Shape). This is the **natural reduction property** we want.

### Concrete Implementation

```python
def _build_shape_varying_features(P, n_complete, J=2, M=1):
    """Build Fourier interaction features for time-varying Shape.

    Returns X of shape (P * n_complete, n_features)
    """
    phases = np.arange(P) / P          # [0, 1/P, ..., (P-1)/P]
    periods = np.arange(n_complete) / n_complete  # [0, ..., (n-1)/n]

    # Within-period Fourier basis (Shape basis)
    phi = [np.ones(P)]
    for j in range(1, J + 1):
        phi.append(np.cos(2 * np.pi * j * phases))
        phi.append(np.sin(2 * np.pi * j * phases))
    phi = np.column_stack(phi)  # (P, 2J+1)

    # Cross-period Fourier basis (variation basis)
    psi = [np.ones(n_complete)]
    for m in range(1, M + 1):
        psi.append(np.cos(2 * np.pi * m * periods))
        psi.append(np.sin(2 * np.pi * m * periods))
    psi = np.column_stack(psi)  # (n_complete, 2M+1)

    # Interaction: Kronecker product for each (i, k) pair
    n_phi = phi.shape[1]
    n_psi = psi.shape[1]
    X = np.zeros((P * n_complete, n_phi * n_psi))

    for k in range(n_complete):
        for i in range(P):
            row = k * P + i  # or i * n_complete + k depending on layout
            X[row, :] = np.kron(phi[i], psi[k])

    return X
```

### Connection to Tensor Decomposition

This is mathematically equivalent to a rank-(2J+1) Tucker decomposition of the (P x n_complete) matrix with structured factor matrices (Fourier bases). The coefficients c_{jm} form the core tensor.

### Verdict

**Top candidate. Clean math, compact features, natural regularization, no parameters beyond J and M (which can be fixed at J=2, M=1 for all cases).** The key insight is that the Kronecker/outer-product structure keeps the feature count at (2J+1)*(2M+1) rather than P*n_complete.

---

## 6. Tensor Decomposition (CP / Tucker)

### CP Decomposition

The (P x n_complete) matrix M = sum_{r=1}^{R} u_r (x) v_r, where u_r in R^P (shape mode) and v_r in R^{n_complete} (temporal mode).

Rank-1 CP = FLAIR's current Level x Shape.
Rank-R CP gives R different shapes with R different temporal weightings.

### Problem with CP

Rank R needs to be chosen. ALS (alternating least squares) is iterative, not closed-form. And CP does not naturally embed in Ridge.

### Tucker Decomposition

M = G x_1 U x_2 V, where U in R^{P x J} (shape basis), V in R^{n_complete x M} (temporal basis), G in R^{J x M} (core tensor).

If we fix U and V as Fourier bases, then the only unknowns are G (the J x M core tensor coefficients). This is exactly the Fourier interaction approach from Section 5!

### Tucker with Fourier Bases = Section 5

```
M_{ik} = sum_{j,m} G_{jm} * U_{ij} * V_{km}
```

where U = Fourier basis for phases, V = Fourier basis for periods.

With fixed bases, estimating G is a linear regression problem -> Ridge compatible.

### Non-Negative Tucker

If we want non-negative Shape (proportions must be >= 0), we can:
1. Work in log-space: log(y/Level) = sum c_{jm} phi_j(i) psi_m(k) + noise
2. Or impose non-negativity via softmax after prediction

### Verdict

**Tucker with Fourier bases IS the Fourier interaction approach.** This validates Section 5 from a tensor decomposition perspective. CP is less useful because it requires iterative optimization.

---

## 7. Dynamic Mode Decomposition (DMD)

### Mathematical Formulation

Given snapshots arranged as columns of X1 (columns 1..n-1) and X2 (columns 2..n), DMD finds:

```
X2 ≈ A * X1
```

where A is the best-fit linear operator. Via SVD of X1 = U Sigma V*:

```
A_tilde = U* X2 V Sigma^{-1}   (reduced r x r operator)
```

Eigendecompose A_tilde = W Lambda W^{-1}, then DMD modes:

```
Phi = X2 V Sigma^{-1} W
```

Reconstruction/forecast:

```
x_k = Phi * Lambda^k * b
```

where b = Phi^+ x_1 are mode amplitudes.

### Application to FLAIR's Matrix

Our matrix is (P x n_complete). We can treat each column as a "snapshot" of the within-period pattern. DMD would find modes (shape patterns) that evolve over time with eigenvalues governing growth/decay/oscillation.

```python
X1 = mat[:, :-1]   # (P, n_complete-1)
X2 = mat[:, 1:]    # (P, n_complete-1)

U, s, Vt = np.linalg.svd(X1, full_matrices=False)
r = min(5, len(s))  # truncation rank
U_r, s_r, Vt_r = U[:, :r], s[:r], Vt[:r, :]

A_tilde = U_r.T @ X2 @ Vt_r.T @ np.diag(1/s_r)
eigvals, W = np.linalg.eig(A_tilde)

Phi = X2 @ Vt_r.T @ np.diag(1/s_r) @ W  # DMD modes (P, r)
b = np.linalg.lstsq(Phi, mat[:, 0], rcond=None)[0]  # amplitudes

# Forecast m steps ahead from last column:
b_now = np.linalg.lstsq(Phi, mat[:, -1], rcond=None)[0]
forecasts = np.column_stack([
    Phi @ np.diag(eigvals**(j+1)) @ b_now for j in range(m)
])
```

### Key Properties

- DMD modes are time-varying shapes: each mode phi_r oscillates with frequency determined by arg(lambda_r) and grows/decays with |lambda_r|
- When |lambda_r| = 1 and Im(lambda_r) = 0, the mode is FIXED (equivalent to FLAIR's current approach)
- DMD captures rotations in shape space (e.g., peak shifting over time)
- All computations are SVD-based -> closed-form
- No Ridge involved, but conceptually similar (least-squares fit of linear operator)

### Problem: Rank Selection

DMD still requires choosing the truncation rank r. Too low = misses important modes. Too high = overfits to noise. This is the same rank-selection problem as SVD.

### Possible Fix: Soft-Average Over Ranks

Apply FLAIR's GCV soft-average idea to DMD: compute forecasts at r=1,2,...,R_max, evaluate each with GCV-like criterion, and take a weighted average. This eliminates hard rank selection.

### Connection to FLAIR

DMD is MORE general than FLAIR's Level x Shape:
- FLAIR = rank-1 DMD where the operator A is a scalar (Level ratio)
- DMD allows rank-r with complex eigenvalues (rotating/evolving shapes)

But DMD does not naturally embed in Ridge because the eigenvalue structure is nonlinear.

### Verdict

**Theoretically elegant but doesn't integrate cleanly into FLAIR's Ridge framework.** DMD is a standalone forecasting method for the (P x n_complete) matrix. It could REPLACE FLAIR's Level x Shape + Ridge, but combining them is awkward. Rank selection remains a problem unless we use soft-averaging.

---

## 8. SparseTSF / PatchTST Perspective

### SparseTSF's Approach

SparseTSF (ICML 2024 Oral, TPAMI 2025) reshapes a length-L input into an (n x w) matrix where w = period, transposes to (w x n), applies a SHARED linear layer to each of the w rows, producing (w x m) output where m = ceil(H/w), then reshapes back to length H.

Mathematically:
```
X = Reshape(x, (n, w))^T     # (w, n)
Y = X @ W                     # (w, m), W is (n, m), SHARED across rows
x_hat = Reshape(Y^T, (H,))
```

### Key Insight: SparseTSF = FLAIR Without Shape

SparseTSF's linear layer W maps n past periods to m future periods. Each row of X is a single phase across all periods (exactly like FLAIR's matrix). The SHARED W means the same temporal dynamics apply to all phases -- this is equivalent to FLAIR's Level forecasting!

The difference: SparseTSF does NOT separate Level and Shape. It directly predicts each phase's future values using the same linear mapping. This implicitly assumes Shape is constant (same W for all phases).

### How SparseTSF Handles Time-Varying Seasonality

**It doesn't, explicitly.** SparseTSF assumes a fixed dominant period w and applies the same linear layer. Time-varying seasonality would need different W for different phases or rows -- which SparseTSF avoids for parameter efficiency.

The TPAMI 2025 extended version (SparseTSF/MLP) adds a small MLP to handle residuals, which can capture some Shape variation, but the core method is fixed-Shape.

### PatchTST / xPatch Approach

PatchTST (ICLR 2023) segments the series into patches and processes them with a Transformer. Patches can be period-aligned (like FLAIR's columns). The self-attention mechanism allows different patches to attend to different historical patterns, naturally capturing time-varying seasonality.

xPatch (2024) adds exponential seasonal-trend decomposition, explicitly allowing seasonal patterns to evolve.

### Lesson for FLAIR

The SparseTSF perspective validates FLAIR's reshape approach and confirms that the shared-linear-layer-across-phases design inherently assumes fixed Shape. To get time-varying Shape, we need PHASE-SPECIFIC or CONTEXT-MODULATED weights. The Fourier interaction approach (Section 5) achieves this within Ridge while keeping feature count compact.

### Verdict

**Validates FLAIR's architecture. Confirms that time-varying Shape requires departing from shared weights.** The Fourier interaction approach from Section 5 is the natural Ridge-compatible solution that SparseTSF lacks.

---

## Synthesis: Recommended Approaches

### Tier 1: Implement Immediately (Minimal Change)

#### A. Exponentially Weighted Shape (Section 3)

**Change**: Replace `proportions[:, -K:].mean(axis=1)` with exponentially weighted average.

```python
# Parameter-free: half-life = n_complete / 4
rho = 2 ** (-4.0 / max(n_complete, 4))
w = rho ** np.arange(n_complete - 1, -1, -1)
w /= w.sum()
S = (proportions * w[None, :]).sum(axis=1)
S /= max(S.sum(), 1e-10)
```

**Properties**: 1 line change, parameter-free, closed-form, natural reduction.

#### B. EWMA-Weighted Ridge for Level (Section 1)

**Change**: Apply exponential weights to Ridge observations.

```python
rho_ridge = 2 ** (-2.0 / max(n_complete, 2))
w_ridge = rho_ridge ** np.arange(n_train - 1, -1, -1)
w_sqrt = np.sqrt(w_ridge)
Xw = X * w_sqrt[:, None]
yw = y_target * w_sqrt
beta, loo_resid, _ = _ridge_sa(Xw, yw)
```

**Properties**: 3 lines change, parameter-free, closed-form, natural reduction.

### Tier 2: Moderate Refactor (Architecture Change)

#### C. Fourier Interaction Shape (Section 5 / Tucker)

**Change**: Replace separate Shape + Level with joint Ridge on interaction features.

For each cell (i, k) in the (P x n_complete) matrix, the target is y(i,k) and features are:

```
x = kron(phi(i), psi(k))
```

where phi = within-period Fourier (J=2 -> 5 terms), psi = cross-period Fourier + trend + lag features.

The cross-period features psi already exist in FLAIR (trend, cos/sin of cross-periods, lag). The within-period features phi are new (Fourier basis for phase). The Kronecker product creates interaction terms.

**Feature count**: 5 * (existing n_features) = 5 * ~6 = ~30 features for P*n_complete observations. Very well-conditioned for Ridge.

```python
# Within-period Fourier (J=2)
phases = np.arange(P) / P
phi = np.column_stack([
    np.ones(P),
    np.cos(2*np.pi*1*phases), np.sin(2*np.pi*1*phases),
    np.cos(2*np.pi*2*phases), np.sin(2*np.pi*2*phases),
])  # (P, 5)

# Cross-period features (reuse existing FLAIR features)
psi = X  # (n_complete, nf) -- existing feature matrix

# Joint features via Kronecker
# For cell (i, k): features = kron(phi[i], psi[k])
n_obs = P * n_train
n_joint = phi.shape[1] * psi.shape[1]
X_joint = np.zeros((n_obs, n_joint))
y_joint = np.zeros(n_obs)

for k in range(n_train):
    for i in range(P):
        row = k * P + i
        X_joint[row] = np.kron(phi[i], psi[k])
        y_joint[row] = mat[i, start + k]

beta_joint, loo_joint, _ = _ridge_sa(X_joint, y_joint)

# Forecast: for future period j, compute psi_future, then
# y_hat[i] = kron(phi[i], psi_future) @ beta_joint
```

**Properties**: Elegant, single Ridge, time-varying Shape via interactions, natural regularization shrinks to fixed Shape, no if/else.

#### D. Level-Dependent Shape via Sigmoid (Section 4)

**Change**: Add sigmoid-transformed Level as an interaction feature for Shape.

```python
L_norm = (L - np.median(L)) / max(np.std(L), 1e-10)
G = 1.0 / (1.0 + np.exp(-L_norm))  # sigmoid: maps Level to [0,1]

# Features for cell (i, k):
# x = [phi(i), G(k) * phi(i)]   -- base Shape + Level-modulated Shape
```

2P features (or 2*(2J+1) with Fourier). Ridge automatically learns whether Shape depends on Level.

### Tier 3: Standalone Alternative

#### E. DMD Forecaster (Section 7)

A separate forecasting method that replaces Level x Shape entirely. Good for comparison but doesn't integrate into Ridge.

### What NOT to Pursue

- **CP decomposition**: iterative, rank selection needed
- **Kalman filter**: iterative, requires process noise specification
- **Regime-switching with discrete states**: requires determining number of regimes
- **Calendar-conditional seasonality**: requires timestamp metadata that FLAIR doesn't have

---

## Recommended Implementation Order

1. **Exponentially Weighted Shape** (Section 3): Lowest risk, highest expected impact. One line of code. Test immediately.

2. **EWMA Ridge** (Section 1): Low risk, moderate impact. Three lines of code. Test alongside #1.

3. **Fourier Interaction** (Section 5): Medium risk, potentially highest impact. Architecture change but clean math. Test after #1 and #2 to see if simple approaches suffice.

4. **Level-Dependent Shape** (Section 4): Add sigmoid interaction if Level-Shape correlation exists in target datasets. Can be combined with #3.

---

## Key Mathematical Insight

The unifying principle across all successful approaches is:

**Shape(phase) -> Shape(phase, context)**

where "context" can be:
- **Time** (exponential weighting, Fourier cross-period basis)
- **Level** (sigmoid interaction, linear interaction)
- **Calendar** (day-of-week indicators, if available)

The Fourier interaction approach (Section 5) is the most general: it subsumes time-varying Shape and can be extended with Level interactions. All interactions are multiplicative features in Ridge, so regularization naturally handles the complexity.

The formula:
```
y(i, k) = sum_{j,m} c_{jm} * phi_j(i) * psi_m(k)
```

is a **bilinear model** -- linear in both Shape basis and temporal basis. Ridge on the Kronecker features c_{jm} gives the optimal low-rank approximation under L2 regularization. When variation is absent, Ridge shrinks interaction terms to zero, recovering fixed Shape. When variation is present, the interaction terms capture it. No if/else needed.
