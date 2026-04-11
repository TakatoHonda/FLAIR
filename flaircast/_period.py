"""MDL period selection.

`_select_period` chooses the primary period `P` for a series by
evaluating each candidate from the calendar table (`FREQ_TO_PERIODS`)
and picking the one whose period-folded matrix has the smallest residual
energy outside the leading singular triple, penalized by BIC.

Returns the chosen `P`, the list of plausible secondary periods, the
canonical primary period for the frequency string, and the raw calendar
candidate list (for use by `_compute_cross_periods`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._constants import _EPS_LOG, _MAX_COMPLETE, _MIN_COMPLETE
from ._frequency import _get_period, _get_periods


def _select_period(
    y: NDArray[np.floating],
    n: int,
    freq: str,
) -> tuple[int, list[int], int, list[int]]:
    """MDL period selection via BIC on the SVD spectrum.

    For each candidate period the series is reshaped into a `(P × n_complete)`
    matrix and the SVD spectrum is computed.  The "residual energy"
    `RSS₁ = Σᵢ₌₁∞ σᵢ²` measures how badly the leading rank-1 component
    fails to capture the period-folded structure.  A smaller `RSS₁`
    means the candidate period gives a more rank-1-friendly layout.

    BIC penalty: `(P + n_complete − 1) · log(T)` accounts for the
    parameters of a rank-1 + intercept fit at this candidate.
    """
    period = _get_period(freq)
    cal = _get_periods(freq)
    candidates = [p for p in cal if p >= 1 and n // p >= _MIN_COMPLETE] if cal else []
    if not candidates:
        candidates = [max(period, 1)] if n // max(period, 1) >= _MIN_COMPLETE else [1]

    if len(candidates) == 1:
        P = candidates[0]
    else:
        T_max = min(n, _MAX_COMPLETE * min(candidates))
        y_sel = y[-T_max:]
        best_P, best_bic = candidates[0], np.inf
        for p_cand in candidates:
            nc = T_max // p_cand
            if nc < _MIN_COMPLETE:
                continue
            mat_c = y_sel[-(nc * p_cand) :].reshape(nc, p_cand).T
            s = np.linalg.svd(mat_c, compute_uv=False)
            rss1 = float(np.sum(s[1:] ** 2))
            T = nc * p_cand
            bic = T * np.log(max(rss1 / T, _EPS_LOG)) + (p_cand + nc - 1) * np.log(T)
            if bic < best_bic:
                best_P, best_bic = p_cand, bic
        P = best_P

    secondary = [p for p in cal if p != P and p > P] if cal else []
    return P, secondary, period, cal
