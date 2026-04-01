#!/usr/bin/env python3
"""Generate Chronos Benchmark comparison figures for FLAIR tech report.

Produces:
  - fig6_chronos_mase.png: Agg. Rel. MASE bar chart (all models)
  - fig7_chronos_per_dataset.png: Per-dataset FLAIR vs Chronos-Small heatmap

Usage:
    uv run python research/figures/generate_fig6_chronos_benchmark.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

OUTPUT_DIR = "docs/flair_tech_report"

# =========================================================================
# Data: Agg. Relative Scores (vs Seasonal Naive) on Chronos Zero-Shot (25 datasets)
# Sources:
#   - FLAIR: our evaluation (research/benchmarks/run_monash_benchmark.py)
#   - Chronos models: amazon-science/chronos-forecasting agg-rel-scores CSVs
#   - AutoETS, AutoARIMA, Moirai, TimesFM: autogluon/fev chronos_zeroshot CSVs
# =========================================================================

MODELS = [
    ("FLAIR\n(~6 params, CPU)",       0.696, 0.803, "#E74C3C", True),
    ("Chronos-Bolt-Base\n(205M)",     0.791, 0.624, "#3498DB", False),
    ("Moirai-Base\n(311M)",           0.812, 0.637, "#3498DB", False),
    ("Chronos-T5-Base\n(200M)",       0.816, 0.642, "#3498DB", False),
    ("Chronos-Bolt-Small\n(48M)",     0.819, 0.636, "#3498DB", False),
    ("Chronos-T5-Large\n(710M)",      0.821, 0.650, "#3498DB", False),
    ("Chronos-T5-Small\n(46M)",       0.830, 0.665, "#3498DB", False),
    ("AutoARIMA\n(CPU)",              0.865, 0.742, "#95A5A6", False),
    ("Chronos-T5-Tiny\n(8M)",        0.870, 0.711, "#3498DB", False),
    ("TimesFM\n(200M)",              0.879, 0.711, "#3498DB", False),
    ("AutoETS\n(CPU)",               0.937, 0.812, "#95A5A6", False),
    ("Seasonal Naive",               1.000, 1.000, "#BDC3C7", False),
]

# Per-dataset relative MASE (FLAIR vs Chronos-Small vs Seasonal Naive)
DATASETS_REL = {
    "dominick":        (0.603, 0.931),
    "ercot":           (1.121, 0.742),
    "exchange_rate":   (0.885, 1.043),
    "m4_quarterly":    (0.894, 0.775),
    "m4_yearly":       (0.843, 0.941),
    "m5":              (0.352, 0.670),
    "aus_electricity": (0.467, 0.977),
    "car_parts":       (0.319, 0.742),
    "cif_2016":        (0.785, 0.790),
    "covid_deaths":    (0.086, 0.902),
    "fred_md":         (0.742, 0.431),
    "hospital":        (0.916, 0.771),
    "m1_monthly":      (0.960, 0.892),
    "m1_quarterly":    (0.883, 0.870),
    "m1_yearly":       (0.717, 0.968),
    "m3_monthly":      (0.887, 0.773),
    "m3_quarterly":    (1.023, 0.897),
    "m3_yearly":       (0.917, 1.066),
    "nn5_weekly":      (0.756, 0.873),
    "tourism_monthly": (1.009, 1.180),
    "tourism_quarterly": (1.044, 1.037),
    "tourism_yearly":  (0.992, 1.123),
    "traffic":         (0.813, 0.762),
    "weather":         (0.554, 0.852),
    "nn5":             (0.606, 0.475),
}


def fig6_mase_bar():
    """Horizontal bar chart: Agg. Relative MASE for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [m[0] for m in MODELS]
    mase = [m[1] for m in MODELS]
    colors = [m[3] for m in MODELS]

    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, mase, color=colors, edgecolor="white", height=0.7)

    # Annotate
    for i, (bar, m) in enumerate(zip(bars, MODELS)):
        val = m[1]
        is_flair = m[4]
        ax.text(val + 0.008, y_pos[i], f"{val:.3f}",
                va='center', ha='left', fontsize=10,
                fontweight='bold' if is_flair else 'normal')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Agg. Relative MASE (lower is better)", fontsize=11)
    ax.set_title("Chronos Zero-Shot Benchmark — Point Forecast Accuracy\n"
                 "FLAIR (~6 params, CPU) vs Foundation Models (8M–710M params, GPU)",
                 fontsize=12, fontweight='bold')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Seasonal Naive')
    ax.set_xlim(0.55, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/fig6_chronos_mase.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def fig7_per_dataset():
    """Per-dataset comparison: FLAIR vs Chronos-Small relative MASE."""
    datasets = sorted(DATASETS_REL.keys(), key=lambda d: DATASETS_REL[d][0])
    flair_vals = [DATASETS_REL[d][0] for d in datasets]
    chronos_vals = [DATASETS_REL[d][1] for d in datasets]

    fig, ax = plt.subplots(figsize=(12, 7))

    y = np.arange(len(datasets))
    h = 0.35

    bars_f = ax.barh(y + h/2, flair_vals, h, label="FLAIR (~6 params, CPU)",
                     color="#E74C3C", alpha=0.85)
    bars_c = ax.barh(y - h/2, chronos_vals, h, label="Chronos-T5-Small (46M, GPU)",
                     color="#3498DB", alpha=0.85)

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=8)
    ax.set_xlabel("Relative MASE (method / Seasonal Naive, lower is better)", fontsize=10)
    ax.set_title("Per-Dataset Comparison — FLAIR vs Chronos-T5-Small\n"
                 "FLAIR wins 14/25 datasets on point forecast accuracy",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Mark winners
    for i, d in enumerate(datasets):
        f, c = DATASETS_REL[d]
        winner = "FLAIR" if f < c else "Chronos"
        color = "#E74C3C" if f < c else "#3498DB"
        marker = "◀" if f < c else "▶"

    fig.tight_layout()
    path = f"{OUTPUT_DIR}/fig7_chronos_per_dataset.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig6_mase_bar()
    fig7_per_dataset()
    print("Done.")
