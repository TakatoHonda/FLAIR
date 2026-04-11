"""Numerical constants used across the FLAIR pipeline.

Centralized so the algorithm modules (`_level`, `_shape`, `_period`,
`_forecast`) can import a single source of truth instead of redefining
fudge factors locally.  Every constant is documented with its role and
the comparable scale (so future readers can spot a misuse like
mixing `_EPS_BOXCOX` into a probability calculation).
"""

from __future__ import annotations

# General-purpose division guard.
_EPS = 1e-10
# Floor for Box-Cox input — `_bc()` clips `y` to `[_EPS_BOXCOX, ∞)` so the
# transform is never asked for the log of a non-positive number.
_EPS_BOXCOX = 1e-8
# Floor inside `log()` to avoid −∞ in BIC calculations.
_EPS_LOG = 1e-30
# Threshold below which a softmax weight is treated as zero (skipped in
# the LOOCV soft-average inside `_ridge_sa`).
_EPS_WEIGHT = 1e-15
# Floor for Shape proportions; keeps the multiplicative decomposition
# `y ≈ L ⊗ S` away from divide-by-zero on rare phases.
_EPS_SHAPE = 1e-6

# Clip range for `exp()` in the inverse Box-Cox path (`lam = 0`).
_BC_EXP_CLIP = 30
# Minimum positive observations required to estimate Box-Cox `lambda`.
_MIN_POSITIVE_FOR_BC = 10

# Period configuration.
# Minimum number of complete periods required for the non-degenerate
# `Level × Shape` decomposition; below this FLAIR falls back to `P = 1`.
_MIN_COMPLETE = 3
# Cap on complete periods (memory and speed guard).
_MAX_COMPLETE = 500

# LSR1 reparameterization toggle.  When True, the Level Ridge fits
# `ΔL_innov` (random-walk prior on `β₂`); when False, it fits `L_innov`
# directly (stationary prior).  See `_forecast._fit_level_ridge`.
_DIFF_TARGET = True

# Number of recent periods used for Shape estimation.  Sensitivity
# analysis in the paper shows < 0.2% impact for K ∈ [2, 50].
_SHAPE_K = 2

# Number of recent periods used for the phase-noise residual matrix.
_PHASE_NOISE_K = 50

# Ridge LOOCV soft-average grid (log-spaced alphas).
_N_ALPHAS = 25
_ALPHA_LOG_MIN = -4  # log10 of minimum Ridge alpha
_ALPHA_LOG_MAX = 4  # log10 of maximum Ridge alpha
