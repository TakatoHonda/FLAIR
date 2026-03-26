#!/usr/bin/env python3
"""Generate comprehensive Monash/Chronos benchmark comparison.

Compares FLAIR against all available methods: Foundation Models, Deep Learning,
and Statistical baselines on the Chronos Benchmark II (25 zero-shot datasets).

Usage:
    uv run python research/figures/generate_fig8_monash_full.py

Output:
    docs/flair_tech_report/fig8_monash_full.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

OUTPUT = "docs/flair_tech_report/fig8_monash_full.png"

# (name, agg_rel_MASE, agg_rel_WQL, category)
# All scores computed on the same 25 zero-shot datasets.
# Sources:
#   FLAIR: results/18_monash_benchmark/flair_monash_results.csv
#   Chronos: results/chronos_benchmark/*-zero-shot.csv
#   AutoARIMA/ETS/Theta, Moirai, TimesFM: autogluon/fev per-dataset CSVs
#   DL (PatchTST etc): Chronos paper Figure 5 (27 datasets, approximate)
MODELS = [
    ("FLAIR\n(~6 params, CPU)",      0.691, 0.801, "#E74C3C", "FLAIR"),
    ("Moirai-Large\n(1B)",           0.787, 0.633, "#3498DB", "Foundation Model"),
    ("TimesFM-2.0\n(200M)",          0.797, 0.719, "#3498DB", "Foundation Model"),
    ("Chronos-Bolt-Base\n(205M)",    0.803, 0.639, "#3498DB", "Foundation Model"),
    ("PatchTST\n(per-dataset)",      0.810, 0.684, "#F39C12", "Deep Learning"),
    ("Moirai-Base\n(311M)",          0.812, 0.635, "#3498DB", "Foundation Model"),
    ("Chronos-T5-Base\n(200M)",      0.822, 0.648, "#3498DB", "Foundation Model"),
    ("Chronos-T5-Large\n(710M)",     0.830, 0.659, "#3498DB", "Foundation Model"),
    ("N-HiTS\n(per-dataset)",        0.830, 0.672, "#F39C12", "Deep Learning"),
    ("Chronos-Bolt-Small\n(48M)",    0.832, 0.651, "#3498DB", "Foundation Model"),
    ("N-BEATS\n(per-dataset)",       0.835, 0.681, "#F39C12", "Deep Learning"),
    ("Chronos-T5-Small\n(46M)",      0.839, 0.675, "#3498DB", "Foundation Model"),
    ("TFT\n(per-dataset)",           0.847, 0.639, "#F39C12", "Deep Learning"),
    ("AutoARIMA\n(CPU)",             0.865, 0.741, "#95A5A6", "Statistical"),
    ("TimesFM\n(200M)",              0.879, 0.711, "#3498DB", "Foundation Model"),
    ("AutoTheta\n(CPU)",             0.881, 0.795, "#95A5A6", "Statistical"),
    ("Moirai-Small\n(91M)",          0.890, 0.707, "#3498DB", "Foundation Model"),
    ("AutoETS\n(CPU)",               0.937, 0.815, "#95A5A6", "Statistical"),
    ("Seasonal Naive",               1.000, 1.000, "#BDC3C7", "Statistical"),
]


def main():
    fig, ax = plt.subplots(figsize=(12, 9))

    names = [m[0] for m in MODELS]
    mase = [m[1] for m in MODELS]
    colors = [m[3] for m in MODELS]

    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, mase, color=colors, edgecolor="white", height=0.7,
                   alpha=0.9)

    for i, m in enumerate(MODELS):
        val = m[1]
        is_flair = m[4] == "FLAIR"
        ax.text(val + 0.005, y_pos[i], f"{val:.3f}",
                va='center', ha='left', fontsize=9,
                fontweight='bold' if is_flair else 'normal',
                color="#E74C3C" if is_flair else "#333333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Agg. Relative MASE (geometric mean vs Seasonal Naive, lower is better)",
                  fontsize=11)
    ax.set_title(
        "Chronos Benchmark II — FLAIR vs All Methods\n"
        "25 zero-shot datasets / Foundation Models + Deep Learning + Statistical",
        fontsize=13, fontweight='bold')
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0.55, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='FLAIR (CPU, 0 hyperparams)'),
        Patch(facecolor='#3498DB', label='Foundation Models (GPU, pre-trained)'),
        Patch(facecolor='#F39C12', label='Deep Learning (GPU, per-dataset)'),
        Patch(facecolor='#95A5A6', label='Statistical (CPU)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
              framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
