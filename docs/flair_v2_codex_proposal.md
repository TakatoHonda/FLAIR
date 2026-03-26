# FLAIR v2 Proposal: Curvature-Regularized Level Ridge

## Thesis

Keep FLAIR's winning decomposition exactly as it is:

`y(phase, period) = Level(period) x Shape(phase)`

Do **not** make `Shape` adaptive.
Do **not** mix architectures.
Do **not** add more latent dimensions.

Change only the geometry of the `Level` regularizer.

The current V9 level model uses ordinary Ridge in coefficient space. That means it shrinks coefficients, but it does **not** directly care whether the lag block produces a smooth, structural level correction or a noisy, high-curvature correction. The failed `diffpen` experiment showed that "penalize lag coefficients more" is too blunt. The next step is to regularize the **correction trajectory itself**, not the raw coefficients.

I propose:

**FLAIR-CR = FLAIR with Curvature-Regularized Level Ridge**

The idea is to keep the deterministic level block as-is, but penalize the **second difference of the lag-generated level correction across periods**.

This is a generalized Ridge / discrete-Sobolev penalty. It is mathematically clean, cheap, and fully compatible with the existing closed-form SVD pipeline.

---

## Why This Direction

### 1. What recent literature says

- **Toner and Darlow (ICML 2024)** show that many recent linear forecasting variants are functionally just linear regression with different feature parameterizations, and that the simpler closed-form solutions win on **72%** of dataset-horizon settings.  
  Link: https://proceedings.mlr.press/v235/toner24a.html

- **SparseTSF (ICML 2024)** succeeds by decoupling periodicity from cross-period trend with an extremely lightweight linear map. This validates FLAIR's core structural split: periodic structure should be handled structurally, while the learned part should stay simple.  
  Link: https://proceedings.mlr.press/v235/lin24n.html

- **Atanasov, Zavatone-Veth, Pehlevan (ICML 2025)** show that ordinary GCV is misspecified for correlated samples and introduce **CorrGCV**, a correlation-corrected alternative. This matters because FLAIR's level samples are time-correlated by construction.  
  Link: https://openreview.net/forum?id=GMwKpJ9TiR

- **Stratify (DMKD 2025)** shows that multi-step forecasting strategy matters and residual correction can help, but it does so by adding a second learner. For FLAIR, where `m = ceil(H/P)` is often small, I do not think "add another model" is the first lever to pull.  
  Link: https://link.springer.com/article/10.1007/s10618-025-01135-1

- **nmtCP (Machine Learning 2024)** shows conformal regions for multi-step time series can be improved independently of the point model. That is useful later, but it is not the first-order accuracy bottleneck here.  
  Link: https://link.springer.com/article/10.1007/s10994-024-06722-9

- **Goulet Coulombe (IJF 2025)** gives a very elegant time-varying-parameter-as-Ridge view, but explicit time-variation is too close to the adaptivity family that already overfit in your ablations.  
  Link: https://www.sciencedirect.com/science/article/pii/S0169207024000931

### 2. What the experiment history says

The failures are highly diagnostic:

- `EWMA Shape`, `Periodic Shape`, `rank-r SVD`, `Fourier Interaction`: all try to make Shape more adaptive or more expressive, and all inject noise.
- `EWMA-weighted Ridge`: adapts the level learner to recency and overreacts to noise.
- `diffpen`: coefficient-level shrinkage on lags is too weak and too coordinate-dependent.
- `FLAIR-Omega`: mixed spaces create numerical and comparability problems.

So the improvement should:

- stay entirely on the `Level` side,
- keep one model and one target series,
- avoid recency weighting,
- avoid adding feature interactions,
- replace coefficient shrinkage with **forecast-space shrinkage**.

That is exactly what curvature regularization does.

---

## Proposed Model

Let:

- `L_t` = total of period `t`
- `z_t = BoxCox(L_t + 1; lambda)`
- `u_t = z_t - z_n`  (the current NLinear level target)

Build the same FLAIR-V9 level features, but split them into two blocks:

- `b_t` = deterministic block = `[1, trend_t, cross-period Fourier_t]`
- `a_t` = lag block = `[u_{t-1}, u_{t-cp}]`

Stack over training periods:

- `B in R^{n x q}`
- `A in R^{n x r}`
- `u in R^n`

with `r` usually only `1` or `2`.

### Estimator

Fit:

```math
(\hat{\gamma}, \hat{\phi})
= \arg\min_{\gamma,\phi}
\|u - B\gamma - A\phi\|_2^2
+ \alpha \|\gamma\|_2^2
+ \alpha \, \phi^\top (\widetilde{R} + \varepsilon I_r)\phi
```

where:

```math
D_2 \in \mathbb{R}^{(n-2)\times n}, \qquad
(D_2 v)_t = v_t - 2v_{t-1} + v_{t-2}
```

and

```math
R = A^\top D_2^\top D_2 A
```

is the curvature matrix of the lag-generated correction path.

To keep scales comparable across datasets, normalize it:

```math
\widetilde{R}
= \frac{r}{\operatorname{tr}(R)+\varepsilon} R
```

with a tiny `eps` such as `1e-8`.

### Equivalent generalized Ridge form

Define:

```math
X = [B \; A], \qquad
\beta = \begin{bmatrix}\gamma \\ \phi\end{bmatrix}, \qquad
\Omega =
\begin{bmatrix}
I_q & 0 \\
0 & \widetilde{R} + \varepsilon I_r
\end{bmatrix}
```

Then:

```math
\hat{\beta}
= \arg\min_\beta \|u - X\beta\|_2^2 + \alpha \beta^\top \Omega \beta
```

This is ordinary generalized Ridge.

Whiten it by `Omega^{-1/2}`:

```math
\theta = \Omega^{1/2}\beta,\qquad
\widetilde{X} = X\Omega^{-1/2}
```

and solve:

```math
\hat{\theta}
= \arg\min_\theta \|u - \widetilde{X}\theta\|_2^2 + \alpha \|\theta\|_2^2
```

So the current SVD-based `soft-average GCV Ridge` code is reusable almost unchanged.

---

## Intuition

Current V9 says:

- deterministic structure lives in trend/Fourier features,
- local correction lives in lag features,
- all coefficients are shrunk isotropically.

FLAIR-CR says:

- deterministic structure still lives in trend/Fourier features,
- local correction still lives in lag features,
- but the lag block is only allowed to create **low-curvature corrections** across periods.

This is the right inductive bias for `Level`:

- the structural periodic part is already removed by `Shape`,
- the remaining level path should be smoother than the raw series,
- lag features should correct, not chase every spike.

In other words:

- `diffpen` penalized **how large** the lag coefficients are,
- `FLAIR-CR` penalizes **how wiggly** the lag correction they generate is.

The latter is the more invariant and more meaningful object.

---

## Why It Avoids Previous Failures

### Not adaptive Shape

`Shape` is unchanged:

- still the recent structural average of within-period proportions,
- still not learned,
- still not time-varying.

So this proposal respects the strongest empirical lesson in the ablation history.

### Not EWMA / recency adaptation

No recency weights are introduced.
All complete periods contribute equally to the level fit.
This avoids the `EWMA Ridge` failure mode.

### Not extra dimensions

No rank expansion, no tensor block, no interaction block, no per-phase learner.
The feature count is identical to V9.

### Not coordinate-dependent shrinkage

`diffpen` depends on which columns happen to be tagged as "lag columns".
Curvature regularization depends on the **trajectory induced by the lag block**.

That is a stronger and more elegant prior.

### Not architecture mixing

There is still:

- one level series,
- one design matrix,
- one Ridge solve,
- one reconstruction step,
- one conformal layer.

So the numerical fragility of `FLAIR-Omega` is avoided.

### Better long-horizon behavior

Because recursive forecasts feed back through the lag block, noisy lag coefficients create oscillatory rollouts.
A curvature penalty directly damps that mechanism, so medium/long horizons should be more stable without requiring `H` separate models.

---

## Pseudocode

```python
def flair_cr(y_raw, horizon, period, freq_str, n_samples=20):
    y = sanitize(y_raw)
    P, secondary = get_primary_and_secondary_periods(freq_str, period)
    if P < 2 or len(y) // P < 3:
        return flar_v5(...)

    # 1. Structural shape: unchanged from V9
    mat = reshape_complete_periods(y, P)           # (P, n_complete)
    S = recent_average_proportions(mat, K=5)       # shape vector, structural only

    # 2. Level series: unchanged preprocessing
    L = mat.sum(axis=0)
    lam = boxcox_lambda(L)
    z = boxcox(L + 1, lam)
    u = z - z[-1]                                  # NLinear

    # 3. Same feature split as V9
    B, A, u_train = build_training_blocks(u, secondary)
    X = np.column_stack([B, A])

    # 4. Curvature penalty on lag-generated correction
    D2 = second_difference_matrix(len(u_train))
    R = A.T @ D2.T @ D2 @ A
    R = (A.shape[1] / (np.trace(R) + 1e-8)) * R

    q = B.shape[1]
    r = A.shape[1]
    Omega = block_diag(np.eye(q), R + 1e-8 * np.eye(r))

    # 5. Whitening -> ordinary Ridge on transformed features
    eigval, Q = np.linalg.eigh(Omega)
    Omega_mhalf = Q @ np.diag(1.0 / np.sqrt(np.maximum(eigval, 1e-8))) @ Q.T
    X_tilde = X @ Omega_mhalf

    # 6. Same soft-average Ridge machinery
    theta, loo_resid, _ = ridge_gcv_loo_softavg(X_tilde, u_train)
    beta = Omega_mhalf @ theta

    # 7. Same recursive level forecast
    u_hat = recursive_forecast_level(beta, horizon_m=ceil(horizon / P))
    z_hat = u_hat + z[-1]
    L_hat = np.maximum(boxcox_inv(z_hat, lam) - 1, 0.0)

    # 8. Same reconstruction
    point_fc = (L_hat[:, None] * S[None, :]).reshape(-1)[:horizon]

    # 9. Same LOO conformal layer as V9
    return conformal_samples_from_phase_residuals(point_fc, loo_resid, S, mat, ...)
```

---

## Expected Effects

### Point accuracy

Primary expected gain:

- better `relMASE` on high-frequency noisy series where the lag block currently chases local spikes,
- better medium/long rollouts because recursive level forecasts become less oscillatory,
- little to no damage on strong clean seasonal series because the deterministic block is untouched.

The high-risk / high-reward datasets for this idea are exactly the ones where V9 still loses badly:

- `bitbrains_fast_storage/H/short`
- `bizitobs_service/10S/*`
- `bizitobs_application/10S/*`
- `electricity/15T/short`
- `solar/10T/long`

These are precisely the cases where "structural shape + noisy level correction" is the bottleneck.

### Probabilistic accuracy

I expect a smaller but real `relCRPS` gain:

- the conformal layer itself is unchanged,
- but smoother point fits should tighten and de-noise the LOO residual pool,
- so intervals should become less erratic.

### Runtime and stability

Very cheap:

- `A` has only 1-2 columns,
- `Omega` is tiny,
- one eigendecomposition of an `nf x nf` matrix is negligible.

Numerically this should be much safer than mixed-resolution or tensor proposals.

---

## Recommended Ablation Order

To isolate the effect cleanly:

1. Compare `V9` vs `CR` with the **same** current soft-average GCV selector.
2. If `CR` wins, then replace vanilla GCV with **CorrGCV** for the same generalized-Ridge design.

Reason:

- the new penalty is the main modeling change,
- CorrGCV is theoretically attractive, but it should be tested after the penalty itself is validated.

---

## Bottom Line

The beautiful move is not to make `Shape` smarter.
It is to make the `Level` regularizer act on the **forecast correction path** rather than on arbitrary coefficient coordinates.

That gives FLAIR a stronger structural prior without giving it more flexibility.
It is exactly the kind of change suggested by both the ablation history and the recent linear-forecasting literature.

---

## References

- Toner, W., Darlow, L. N. (2024). *An Analysis of Linear Time Series Forecasting Models*. ICML 2024. https://proceedings.mlr.press/v235/toner24a.html
- Lin, S. et al. (2024). *SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters*. ICML 2024. https://proceedings.mlr.press/v235/lin24n.html
- Atanasov, A., Zavatone-Veth, J. A., Pehlevan, C. (2025). *Risk and cross validation in ridge regression with correlated samples*. ICML 2025. https://openreview.net/forum?id=GMwKpJ9TiR
- Green, R. et al. (2025). *Stratify: unifying multi-step forecasting strategies*. Data Mining and Knowledge Discovery. https://link.springer.com/article/10.1007/s10618-025-01135-1
- Schlembach, F. et al. (2024). *Conformal multistep-ahead multivariate time-series forecasting*. Machine Learning. https://link.springer.com/article/10.1007/s10994-024-06722-9
- Goulet Coulombe, P. (2025). *Time-varying parameters as ridge regressions*. International Journal of Forecasting. https://www.sciencedirect.com/science/article/pii/S0169207024000931
- Emura, T. et al. (2024). *g.ridge: An R Package for Generalized Ridge Regression for Sparse and High-Dimensional Linear Models*. Symmetry. https://www.mdpi.com/2073-8994/16/2/223
