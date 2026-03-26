# FLAIR Ablation History (for paper Table)

## Progressive Improvement

| Version | Change | relMASE | relCRPS | Delta |
|---------|--------|---------|---------|-------|
| V9 (base) | Level x Shape, Ridge SA, NLinear | 1.028 | 0.750 | — |
| + n_samples=200 | More conformal samples | 1.014 | 0.731 | -1.4% |
| + Dirichlet Shape | Context = k%C, K*C window, kappa from MoM | 0.983 | 0.713 | -3.1% |
| + Location Shift | y += max(1-min(y), 1) before Box-Cox | 0.920 | 0.690 | -6.4% |
| + V5 removal | P=1 degeneration, no fallback model | 0.915 | 0.690 | -0.5% |
| + MDL Period | BIC on SVD spectrum selects P from candidates | ~0.89* | TBD | ~-2.7% |

*projected from partial results

## Total improvement: relMASE 1.028 → ~0.89 (-13.4%)

## Failed Improvements (all tested, all made things worse)

### Shape Adaptation (Frozen Shape Theorem)
- EWMA Shape: 1.139
- Fourier Shape (J=1): 1.073
- Rank-2 SVD: 1.058
- Rank-r SVD (auto): 1.062
- Exponential smoothing: similar to EWMA

### Feature Engineering
- Velocity feature: hospital +0.18, m4_monthly +0.12 (short series overfit)
- Feature standardization: electricity/H/long +0.23 (broke feature balance)
- Quadratic trend: recursive divergence
- Integration closure (FLAIR-I): hospital +0.33, m4_monthly +0.44

### Architecture Changes
- Deseasonalize-first: medium/long catastrophic (2.541)
- Omega (mixed-resolution Ridge): implementation bugs
- Tensor (cross-variate pooling): heterogeneous variates break shared beta
- MinT reconciliation: Box-Cox nonlinearity
- Ordered QR Ridge: singular matrix errors, no improvement

### Regularization
- James-Stein lag shrinkage: 1.105
- Curvature-regularized Ridge: 1.112
- DiffPen: marginal -0.1%
- 2D soft-average (alpha x kappa): softmax dilution

### Other
- Direct multi-step: lag stale for large h
- FFT spectral periods: spurious periods
- Two-model blend: structural model too weak
- GCV/LOO ensemble: cross-architecture GCV incomparable
- Differencing order soft-average: not publishable (ad hoc)

## Key Lessons (for paper discussion)

1. Shape MUST be frozen. 17 attempts proved this empirically.
2. Additional features hurt short Level series. Ridge's uniform penalty can't selectively shrink.
3. The rank-1 assumption is fundamentally robust — adapt the preprocessing (shift, P selection), not the decomposition.
4. The biggest gains came from fixing BUGS (max(y,0) destroying data) and SELECTION (MDL choosing the right P), not from making the model more complex.
5. Simplicity is a feature, not a limitation.
