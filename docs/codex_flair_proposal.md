# Proposal: FLAIR-Omega

## Thesis

Do not ensemble `Level x Shape` and `Simple Ridge` as two architectures.
Instead, make them two subspaces inside one generalized Ridge regression.

The cleanest formulation is a **single mixed-resolution linear model**:

- a **structured block** that lifts the current FLAIR `Level x Shape` idea back to raw time
- a **local block** that captures the same short-series / weak-seasonality behavior that `Simple Ridge` was rescuing
- one response vector
- one hat matrix
- one GCV/LOO calculation
- one closed-form generalized Ridge solve

This removes the unfair "compare LOO across architectures" problem completely, because there is no architecture-level comparison anymore.

---

## What the experiment history says

From `src/run_gift_eval_flar9.py` and the surrounding notes:

- `flar9` wins because it reduces the forecasting problem from raw resolution to a smoother period-level series `L_k`, then reconstructs with a shape vector `S`.
- That is mathematically elegant and sample-efficient when the periodic decomposition is real.
- The failure mode is also clear: for short / low-frequency / weakly seasonal series, the reshape step leaves too few complete periods, and the shape estimate becomes fragile.

From `src/run_gift_eval_flair2.py`:

- the attempted fix was to always compute both `Level x Shape` and `Simple Ridge`, then softmax-combine them using LOO MSE
- this is unsatisfactory because the two paths are not being evaluated on symmetric objects
- the `Level x Shape` path pays approximation error from the imposed decomposition, while the raw `Simple Ridge` path does not

So the right move is not "better model selection".
The right move is: **stop treating them as separate models**.

---

## Proposed architecture

Name: **FLAIR-Omega: Confidence-Weighted Mixed-Resolution Ridge**

Core idea:

```text
y_hat = structured_periodic_component + local_residual_component
```

with both components fit **jointly** in one generalized Ridge regression.

### 1. Data layout

For a primary period `P`, write the series as:

- `y_(k,j)` = value in period `k`, phase `j`
- `L_k = sum_j y_(k,j)` = period total
- `p^(k)_j = y_(k,j) / L_k` = within-period proportion

This preserves the current FLAIR semantics:

- `L_k` is the "level"
- `p_j` is the "shape"

### 2. Shape should be shrunk, not hard-estimated

Instead of using the empirical recent shape directly, estimate a **shrunk shape**:

```text
u = uniform shape = (1/P) * 1
s_bar = mean of recent normalized period shapes
s_hat = (tau * u + K * s_bar) / (tau + K)
```

where:

- `K` = number of recent complete periods used
- `tau` = pseudo-count / prior strength

This is the simplest Bayesian-Dirichlet style regularization of shape.

Behavior:

- many stable periods -> `s_hat` approaches empirical shape
- few / noisy periods -> `s_hat` smoothly collapses toward uniform

This already fixes part of the short-series pathology without any if/else.

### 3. Lift the FLAIR block back to raw time

Keep the current `flar9` level feature construction on `L_k`:

```text
x_k = [1, trend(k), cross-period Fourier on k, lag(L)_1, lag(L)_cp, ...]
```

Now define the structured raw-time predictor by multiplying those level features by the phase shape:

```text
structured prediction at (k,j) = s_hat_j * (x_k^T beta_s)
```

This is exactly the current `Level x Shape` logic, but expressed as columns of one raw-time design matrix.

If `X_s` is the stacked matrix of rows `s_hat_j * x_k^T`, then:

```text
y_struct = X_s beta_s
```

### 4. Add a local raw-time correction block

Build a second feature block from the current `Simple Ridge` idea:

```text
z_t = [1, trend(t), lag(y)_1]
```

or slightly richer if desired:

```text
z_t = [1, trend(t), lag(y)_1, lag(y)_P]
```

Then:

```text
y_local = X_l beta_l
```

This block is what rescues non-periodic, short, and low-frequency series.

### 5. Make it a true direct-sum decomposition

To keep the decomposition clean and avoid duplicated intercept/trend behavior, orthogonalize the local block against the structured block:

```text
X_l_perp = (I - P_s) X_l
P_s = X_s (X_s^T X_s)^(-1) X_s^T
```

Then the final design is:

```text
X = [X_s, X_l_perp]
```

Interpretation:

- `X_s` explains the periodic low-rank part
- `X_l_perp` only explains what `Level x Shape` cannot

That is a very clean mathematical story.

---

## The actual estimator

Fit one generalized Ridge:

```text
theta_hat
= argmin_theta ||y - X theta||^2 + alpha * theta^T Omega(c) theta
```

with block penalty

```text
Omega(c) = diag( (1/c) I_s , (1/(1-c)) I_l )
```

where:

- `c in [0,1]` is a **shape confidence**
- `I_s` is identity on the structured block
- `I_l` is identity on the local block

Recommended confidence:

```text
shape_signal = mean_j (s_bar_j - 1/P)^2
shape_noise  = mean_j Var_k( p^(k)_j )
c = (K / (K + tau)) * shape_signal / (shape_signal + shape_noise + eps)
c = clip(c, 1e-3, 1 - 1e-3)
```

This gives exactly the desired behavior:

- large `K`, stable non-uniform shape -> `c ~ 1` -> trust `Level x Shape`
- small `K` or weak / unstable seasonality -> `c ~ 0` -> trust the local raw block

No thresholding. No branching. No model selection.

### Closed-form solve

This is still standard Ridge after feature rescaling:

```text
Omega(c)^(-1/2) = diag( sqrt(c) I_s , sqrt(1-c) I_l )
X_tilde = X Omega(c)^(-1/2)
theta_tilde_hat = argmin ||y - X_tilde theta_tilde||^2 + alpha ||theta_tilde||^2
theta_hat = Omega(c)^(-1/2) theta_tilde_hat
```

So you can reuse the existing SVD-based `_ridge_gcv_loo_softavg(...)` almost unchanged:

1. build `X_tilde`
2. run the current soft-average GCV Ridge solver
3. map coefficients back through `Omega(c)^(-1/2)`

This satisfies all constraints:

- Ridge only
- closed form
- SVD-based
- no holdout
- no if/else

---

## Why this is better than FLAIR v2

### The comparability problem disappears

In `flair2`, the difficulty is:

- path A and path B are different architectures
- LOO is being used to compare them
- but they do not incur error in the same way

In FLAIR-Omega:

- there is only **one** architecture
- `Level x Shape` and `Simple Ridge` are just two column blocks in one `X`
- GCV only selects the Ridge strength of the single final estimator
- LOO only calibrates the single final estimator

That is much cleaner.

### The short-series fallback becomes intrinsic

`Simple Ridge` is no longer a fallback.
It is the orthogonal correction subspace of the same model.

That is the elegant unification you were looking for.

### It preserves the reason FLAIR v9 works

The structured block is still exactly the FLAIR insight:

- forecast period-level behavior
- distribute by shape
- leverage cross-period signals

But now the model is allowed to spend some of its capacity on raw-time local correction when the structural assumption is weak.

---

## Limiting cases

The proposal has the right limits.

### Strong periodic data

If shape is stable and many complete periods exist:

- `c -> 1`
- local block gets strongly shrunk
- the model behaves like current `flar9`

### Weak seasonality / short data

If shape is uncertain or nearly uniform:

- `c -> 0`
- structured block gets strongly shrunk
- the model behaves like `Simple Ridge`

### `P = 1`

Then shape is trivial and the structured block offers no special advantage.
The confidence naturally collapses toward the local block.
Again: no branch required.

---

## Forecast recursion

Training is on raw-time rows, but the structured block still uses period-level state.
Forecasting is straightforward:

1. Maintain a rolling history of raw values and period totals.
2. For each future period `k`:
   - construct `x_k` from observed/predicted past period totals exactly as in `flar9`
3. For each phase `j` inside that period:
   - construct local features `z_(k,j)` from the latest raw history
   - predict

```text
y_hat_(k,j) = s_hat_j * (x_k^T beta_s) + z_(k,j)^T beta_l
```

4. After a full future period is generated, sum it to obtain the next predicted `L_k`.

So the structured state evolves at period resolution, while the local correction evolves at raw resolution.
That is the whole point of the mixed-resolution design.

---

## Implementation sketch in this repo

This can be implemented directly from `src/run_gift_eval_flar9.py` and `src/run_gift_eval_flair2.py`.

### New helper functions

Add:

- `_estimate_shape_shrunk(mat, tau=...) -> (s_hat, c)`
- `_build_level_feature_matrix(L_bc, cross_periods, ...)`
- `_build_mixed_resolution_design(y, L, s_hat, ...) -> X_s, X_l, y_train`
- `_orthogonalize_block(X_base, X_block) -> X_block_perp`
- `_ridge_gcv_loo_softavg_generalized(X, y, penalty_scales)` or simply pre-scale columns and reuse the existing solver

### Minimal code path

1. Reuse `get_periods`, Box-Cox helpers, and the existing SVD Ridge routine.
2. Keep the current `flar9` level feature logic almost verbatim.
3. Replace the two-path logic in `flair2` with one design build:

```python
X_struct = lifted_level_shape_features(...)
X_local  = simple_ridge_features(...)
X_local  = orthogonalize(X_struct, X_local)
X_tilde  = np.column_stack([np.sqrt(c) * X_struct,
                            np.sqrt(1 - c) * X_local])
beta_tilde, loo_resid, _ = _ridge_gcv_loo_softavg(X_tilde, y_train)
```

4. During forecasting, use one recursive loop producing one point forecast.
5. Use the single unified LOO residual pool for conformal sampling.

### Why this is implementable

Nothing here requires:

- neural nets
- iterative optimization
- EM
- alternating minimization
- holdout validation

It is just feature engineering plus generalized Ridge.

---

## Recommended default choices

To keep the first implementation simple and defensible:

- use the same level features as `flar9`
- use local features `[1, t/n, lag1]`
- use `K = min(5, n_complete)` for shape estimation, as in `flar9`
- use `tau = 3` for shape shrinkage
- orthogonalize the local block
- use one global `alpha` selected by the current soft-average GCV

I would not start with a 2D search over `(alpha, c)` or `(alpha, rho)`.
The data-driven `c` from shape stability is the cleaner first paper story.

---

## Why this is publishable-quality elegant

Because the model can be stated in one sentence:

> Forecast raw time points with one generalized Ridge whose hypothesis space is the direct sum of a lifted low-rank `Level x Shape` subspace and an orthogonal local autoregressive correction subspace, with continuous shrinkage determined by posterior shape confidence.

That is a real model, not a bagged workaround.

The conceptual advantages are strong:

- one estimator instead of architecture selection
- one mathematically interpretable bias-variance knob
- one GCV
- one LOO
- exact continuity between periodic and non-periodic regimes
- closed-form implementation using the code you already have

---

## Bottom line

The most elegant unification is:

**do not choose between `Level x Shape` and `Simple Ridge`; embed both in one generalized Ridge with confidence-weighted block shrinkage.**

If you want the shortest practical name:

**FLAIR-Omega = lifted `Level x Shape` + orthogonal local Ridge + confidence-weighted generalized Ridge.**
