"""Shape estimation primitives.

Hosts the two Shape stages of the FLAIR pipeline:

- `_estimate_shape` — Shape₁ (Dirichlet-Multinomial empirical Bayes)
  conditioned on the secondary period context (e.g. day-of-week for
  hourly data).  Degenerates to a global K-period average when no
  secondary structure exists.

- `_compute_shape2` — Shape₂ (BIC-gated empirical Bayes shrinkage of a
  secondary periodic Level pattern, e.g. annual seasonality of a daily
  series).  Selects between a first-harmonic prior (2 params) and the
  flat prior (0 params) by the same MDL principle FLAIR uses for
  primary period selection.

- `_compute_cross_periods` — derives Ridge cross-period lag indices
  from the secondary period list returned by `_period._select_period`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._constants import _EPS, _EPS_LOG, _EPS_SHAPE, _SHAPE_K


def _compute_shape2(
    L: NDArray[np.floating],
    cp: int,
    n_complete: int,
) -> NDArray[np.floating] | None:
    """Shape₂ with MDL-gated empirical Bayes shrinkage.

    Shape₂ = w × raw_proportions + (1−w) × prior
    w = nc₂ / (nc₂ + cp)

    The prior is selected by BIC (MDL): first harmonic (2 params) vs
    flat (0 params).  When the harmonic is not justified by data, the
    flat prior `S₂ = 1` keeps deseasonalization negligible — same MDL
    principle as BIC period selection.
    """
    nc2 = n_complete // cp
    if nc2 < 2:
        return None

    pos = np.arange(n_complete) % cp
    S2_raw = np.zeros(cp)
    for d in range(cp):
        vals = L[pos == d]
        S2_raw[d] = vals.mean() if len(vals) > 0 else 1.0
    raw_mean = S2_raw.mean()
    if raw_mean < _EPS:
        return None
    S2_raw = S2_raw / raw_mean

    # First harmonic fit
    t = np.arange(cp, dtype=float)
    cos_b = np.cos(2 * np.pi * t / cp)
    sin_b = np.sin(2 * np.pi * t / cp)
    S2_c = S2_raw - 1.0
    a = 2.0 * np.mean(S2_c * cos_b)
    b = 2.0 * np.mean(S2_c * sin_b)
    S2_harmonic = 1.0 + a * cos_b + b * sin_b

    # MDL gate: BIC selects harmonic (2 params) vs flat (0 params)
    RSS_flat: float = float(np.sum(S2_c**2))
    RSS_harmonic: float = float(np.sum((S2_raw - S2_harmonic) ** 2))
    bic_flat = cp * np.log(max(RSS_flat / cp, _EPS_LOG))
    bic_harmonic = cp * np.log(max(RSS_harmonic / cp, _EPS_LOG)) + 2 * np.log(cp)
    S2_prior = S2_harmonic if bic_harmonic < bic_flat else np.ones(cp)

    # Empirical Bayes weight
    w = nc2 / (nc2 + cp)
    S2 = w * S2_raw + (1 - w) * S2_prior

    S2 = np.maximum(S2, _EPS_SHAPE)
    S2 = S2 / S2.mean()
    return np.asarray(S2, dtype=np.float64)


def _estimate_shape(
    mat: NDArray[np.floating],
    n_complete: int,
    P: int,
    secondary: list[int],
    L: NDArray[np.floating],
    horizon: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Dirichlet-Multinomial empirical Bayes Shape estimation.

    When a secondary period exists (e.g. day-of-week for hourly data,
    `C = secondary[0] // P`), per-context Shape vectors are estimated
    by Dirichlet-Multinomial smoothing toward the global K-period
    average; the smoothing strength `kappa` is fit by method-of-moments
    on the per-phase variance of the recent `K × C` periods.

    Returns
    -------
    S_forecast : ndarray, shape (m, P)
        Per-block forecast Shape, where `m = ceil(horizon / P)`.
    S_hist : ndarray, shape (n_complete, P)
        Per-period historical Shape used for the rank-1 reconstruction
        residual that drives the phase-noise sampler.
    m : int
        Number of period blocks the forecast horizon spans.
    """
    K = min(_SHAPE_K, n_complete)
    recent = mat[:, -K:]
    totals = recent.sum(axis=0, keepdims=True)
    S_global = np.where(totals > _EPS, recent / totals, 1.0 / P).mean(axis=1)
    S_global /= max(S_global.sum(), _EPS)

    C = (
        secondary[0] // P
        if (secondary and secondary[0] % P == 0 and n_complete >= secondary[0] // P)
        else 1
    )
    m = int(np.ceil(horizon / P))

    if C <= 1:
        S_forecast = np.tile(S_global, (m, 1))
        S_hist = np.tile(S_global, (n_complete, 1))
    else:
        K_ds = min(K * C, n_complete)
        ds_mat = mat[:, -K_ds:]
        ds_L = L[-K_ds:]
        ds_ctx = np.arange(n_complete - K_ds, n_complete) % C

        ds_totals = ds_mat.sum(axis=0, keepdims=True)
        ds_props = np.where(ds_totals > _EPS, ds_mat / ds_totals, 1.0 / P)
        mp = ds_props.mean(axis=1)
        vp = ds_props.var(axis=1, ddof=1)
        valid = (mp > _EPS_SHAPE) & (vp > _EPS)
        kappa = (
            max(float(np.median(mp[valid] * (1 - mp[valid]) / vp[valid] - 1)), 0.0)
            if valid.sum() >= 2
            else 1e6
        )

        S_ctx = np.empty((C, P))
        for c_val in range(C):
            mask = ds_ctx == c_val
            if mask.sum() == 0:
                S_ctx[c_val] = S_global
            else:
                S_c = (kappa * S_global + ds_mat[:, mask].sum(axis=1)) / max(
                    kappa + ds_L[mask].sum(), _EPS
                )
                S_ctx[c_val] = S_c / max(S_c.sum(), _EPS)

        S_forecast = S_ctx[(n_complete + np.arange(m)) % C]
        S_hist = S_ctx[np.arange(n_complete) % C]

    return S_forecast, S_hist, m


def _compute_cross_periods(
    secondary: list[int],
    P: int,
    period: int,
    n_complete: int,
) -> tuple[list[int], int]:
    """Compute cross-period Ridge lag indices from the secondary period list.

    Returns `(cross_periods, max_cp)` where `max_cp` is the lag at which
    the Ridge will inject a per-period seasonal feature.
    """
    cross_periods: list[int] = []
    for sp in secondary:
        cp = sp // P if P >= 2 else sp
        if 2 <= cp <= n_complete // 2:
            cross_periods.append(cp)
    if P == 1 and period >= 2 and period <= n_complete // 2:
        cross_periods = sorted(set(cross_periods) | {period})
    max_cp = max(cross_periods) if cross_periods else 0
    return cross_periods, max_cp
