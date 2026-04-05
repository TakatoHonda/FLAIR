# Changelog

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
