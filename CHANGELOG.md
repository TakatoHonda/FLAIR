# Changelog

## 0.5.0 (2026-04-11)

- **Exogenous variable support**: `forecast()` and `FLAIR.predict()` now accept `X_hist` / `X_future` parameters. Standardized exog columns are appended directly to the Level Ridge feature matrix; the LOOCV soft-averaged Ridge handles regularization, so no separate gating step is required ("One Ridge" preserved). When `X_hist=None` the result is bit-identical to the previous behavior.
- **Empirical validation** in `validation/`: rolling-origin MASE on UCI Bike Sharing daily (−9.4%, 9/12 origins win) and Jena Climate hourly (−15.5%, 19/24 origins win), plus a noise-control showing graceful degradation (mean +0.9%).
- New tests: `tests/test_exogenous.py` adds 21 cases covering smoke, validation, backward compatibility (byte-identity), and effect (informative shifts forecast, noise drift bounded).
- Remove unused module-level `_DIAG` dict (no readers; cleanup only).

## 0.4.1 (2026-04-08)

- Fix: restore historical-range clipping for forecast samples — inverse Box-Cox combined with recursive Student-t simulation can produce extreme values that blow up MASE on certain configs; clip to `[y_lo − range, y_hi + range]` based on a recent-history window, applied after the post-hoc shrinkage.
- Fix: post-hoc interval shrinkage to correct Student-t overdispersion (`sqrt((ν−2)/ν)` toward the median). Validated MASE-safe on GIFT-Eval before re-applying.
- Sync `pyproject.toml` version with `flaircast.__version__`.

## 0.4.0 (2026-04-08)

- **LWCP (Leverage-Weighted Conformal Prediction)**: LOO residuals are now LWCP-normalized (`e_i^LOO / sqrt(1 + h_ii)`) and the test-point interval is scaled by `sqrt(1 + h_test)` per horizon, computed from the same Ridge SVD. Removes leverage-dependent heteroscedasticity in the predictive distribution.
- **Student-t predictive distribution in Box-Cox space**: when σ² is unknown and estimated from LOO residuals, the predictive distribution in BC space is Student-t with `ν = n_train − p`, not Gaussian. As `n → ∞`, `t_ν → N(0,1)`; for short series, the heavier tails reflect higher uncertainty automatically — zero new hyperparameters.
- **Scenario-coherent phase noise** restored as the default: SVD residual columns are sampled in whole columns rather than independently per phase, preserving cross-phase correlation within each forecast block.
- **NaN interpolation** replaces silent 0-fill (`np.interp` for both interior and boundary NaNs); 0-fill biased the Shape and Level estimates downward.
- Refactor: remove the Shape^γ damping experiment ("restore One SVD") — the additional knob did not survive ablation and conflicted with the "One SVD, One Ridge" design philosophy.

## 0.3.0 (2026-04-07)

- LSR1 diff-target reparameterization: Level を random walk として再定式化し、Ridge が差分 ΔL を直接予測
- Damped trend: Level 外挿時に exponential damping を適用し、長期予測の発散を抑制
- GIFT-Eval relMASE 0.885 → 0.857 (-3.2%), relCRPS 0.610 (97 configs, 23 datasets)

## 0.2.1 (2026-04-05)

- Predictive calibration of LOO residuals via leverage correction sqrt(1-h_ii)
- Change Shape K from 5 to 2 (sensitivity analysis shows <0.2% impact for K=2..50)
- Fix pandas offset-anchored frequency resolution (W-SUN to W, Q-DEC to Q)
- Change license from MIT to Apache 2.0

## 0.2.0 (2026-04-02)

- Add `seed` parameter to `forecast()` and `FLAIR` for reproducible results
- Add type hints to all functions (PEP 561 `py.typed` marker)
- Add input validation with `TypeError` / `ValueError` messages
- Add 128 tests (unit, integration, edge case, property-based, golden snapshot), 92% coverage
- Add GitHub Actions CI (Python 3.9-3.13, ruff, mypy)
- Add automated PyPI publish on GitHub release
- Add Colab quick start notebook
- Replace magic numbers with named constants
- Refactor `forecast()` into smaller functions

## 0.1.0 (2026-03-22)

- Initial release
- `forecast()` functional API and `FLAIR` class API
- Level x Shape decomposition with Dirichlet-Multinomial Shape estimation
- MDL period selection via BIC on SVD spectrum
- Shape2 deseasonalization with MDL-gated prior shrinkage
- Ridge regression with GCV soft-average (25 alphas, single SVD)
- Stochastic Level paths with LOO residual noise injection
- Phase noise from SVD Residual Quantiles
