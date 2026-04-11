"""Calendar tables and pandas-style frequency string resolution.

`FREQ_TO_PERIOD` and `FREQ_TO_PERIODS` are part of the public API and
get re-exported from the package root in `flaircast/__init__.py`.

`_resolve_freq`, `_get_period`, `_get_periods` are private helpers used
by `_period._select_period` and the test suite.
"""

from __future__ import annotations

# Primary period for each pandas-style frequency string.
FREQ_TO_PERIOD = {
    "S": 60,
    "T": 60,
    "5T": 12,
    "10T": 6,
    "15T": 4,
    "10S": 6,
    "H": 24,
    "D": 7,
    "W": 52,
    "M": 12,
    "Q": 4,
    "A": 1,
    "Y": 1,
}

# MDL candidate periods (primary + plausible secondary periodicities).
# `_period._select_period` picks the best of these via BIC on the SVD
# spectrum of the period-folded matrix.
FREQ_TO_PERIODS = {
    "10S": [6, 360],
    "S": [60],
    "5T": [12, 288],
    "10T": [6, 144],
    "15T": [4, 96],
    "H": [24, 168],
    "D": [7, 365],
    "W": [52],
    "M": [12],
    "Q": [4],
    "A": [],
    "Y": [],
}


def _resolve_freq(freq: str) -> str:
    """Normalize a pandas-style frequency string for table lookup.

    Strips offset anchors (`W-SUN` → `W`, `Q-DEC` → `Q`, `A-DEC` → `A`)
    and rewrites the legacy `MIN` alias to `T`.
    """
    f = freq.upper().replace("MIN", "T")
    for base in ("W", "Q", "A", "Y"):
        if f.startswith(base + "-"):
            return base
    return f


def _get_period(freq: str) -> int:
    """Look up the primary period for a frequency string.

    Falls back to a longest-suffix match for compound frequencies (e.g.
    `2H` → matches the `H` entry).  Returns `1` when nothing matches.
    """
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIOD:
        return FREQ_TO_PERIOD[f]
    for k in sorted(FREQ_TO_PERIOD, key=len, reverse=True):
        if f.endswith(k):
            return FREQ_TO_PERIOD[k]
    return 1


def _get_periods(freq: str) -> list[int]:
    """Look up the MDL candidate period list for a frequency string."""
    f = _resolve_freq(freq)
    if f in FREQ_TO_PERIODS:
        return list(FREQ_TO_PERIODS[f])
    for k in sorted(FREQ_TO_PERIODS, key=len, reverse=True):
        if f.endswith(k):
            return list(FREQ_TO_PERIODS[k])
    return []
