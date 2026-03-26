"""
Chronos Benchmark II (27 datasets, zero-shot) — Comprehensive Model Comparison

Sources:
1. Figure 5 of Chronos paper (arXiv:2403.07815v3, TMLR 10/2024) — exact bar chart values
2. GitHub CSVs: amazon-science/chronos-forecasting/scripts/evaluation/results/ — per-dataset data
3. Chronos-2 paper (arXiv:2510.15821v1) — Table 5 (Win Rate / Skill Score)

Usage: python benchmark_summary.py
"""

# ============================================================
# Part 1: Benchmark II (27 datasets) — Agg. Relative Scores
#         (from Figure 5 of Chronos paper + Chronos-Bolt GitHub CSVs)
#         Lower is better. 1.000 = Seasonal Naive baseline.
# ============================================================

benchmark_ii_scores = {
    # Model: (Agg Rel WQL, Agg Rel MASE, Category)

    # --- Pretrained Models (Zero Shot = not trained on Benchmark II data) ---
    "Chronos-Bolt (Base)":     (0.624, 0.791, "Pretrained (Zero-Shot)"),   # from GitHub CSV
    "Chronos-Bolt (Small)":    (0.636, 0.819, "Pretrained (Zero-Shot)"),   # from GitHub CSV
    "Chronos-Bolt (Mini)":     (0.644, 0.822, "Pretrained (Zero-Shot)"),   # from GitHub CSV
    "Chronos-T5 (Large)":      (0.645, 0.823, "Pretrained (Zero-Shot)"),   # Figure 5
    "Chronos-T5 (Base)":       (0.662, 0.832, "Pretrained (Zero-Shot)"),   # Figure 5
    "Chronos-T5 (Small)":      (0.667, 0.841, "Pretrained (Zero-Shot)"),   # Figure 5
    "Chronos-Bolt (Tiny)":     (0.668, 0.845, "Pretrained (Zero-Shot)"),   # from GitHub CSV
    "Chronos-T5 (Mini)":       (0.678, 0.850, "Pretrained (Zero-Shot)"),   # Figure 5
    "Chronos-GPT2":            (0.687, 0.852, "Pretrained (Zero-Shot)"),   # Figure 5

    # --- Pretrained Models (Other = may have seen some Benchmark II data) ---
    "Moirai-1.0-R (Large)":    (0.720, 0.876, "Pretrained (Other)"),       # Figure 5
    "Moirai-1.0-R (Base)":     (0.696, 0.907, "Pretrained (Other)"),       # Figure 5
    "LLMTime":                 (0.804, 0.962, "Pretrained (Other)"),       # Figure 5
    "Lag-Llama":               (1.097, 1.291, "Pretrained (Other)"),       # Figure 5
    "ForecastPFN":             (None, 2.450, "Pretrained (Other)"),        # Figure 5 (MASE only)

    # --- Task-Specific Models (trained on each dataset) ---
    "TFT":                     (0.639, 0.847, "Task-Specific"),            # Figure 5
    "N-HiTS":                  (0.672, 0.830, "Task-Specific"),            # Figure 5
    "N-BEATS":                 (0.681, 0.835, "Task-Specific"),            # Figure 5
    "PatchTST":                (0.684, 0.810, "Task-Specific"),            # Figure 5
    "DeepAR":                  (0.733, 0.843, "Task-Specific"),            # Figure 5
    "SCUM":                    (0.728, 0.838, "Task-Specific"),            # Figure 5
    "DLinear":                 (0.757, 0.894, "Task-Specific"),            # Figure 5
    "WaveNet":                 (0.842, 0.951, "Task-Specific"),            # Figure 5
    "GPT4TS":                  (None, 0.895, "Task-Specific"),             # Figure 5 (MASE only)

    # --- Local Models ---
    "AutoARIMA":               (0.761, 0.908, "Local"),                    # Figure 5
    "AutoTheta":               (0.793, 0.875, "Local"),                    # Figure 5
    "AutoETS":                 (0.838, 0.953, "Local"),                    # Figure 5
    "Seasonal Naive":          (1.000, 1.000, "Local"),                    # baseline
    "Naive":                   (1.152, 1.188, "Local"),                    # Figure 5
}


# ============================================================
# Part 2: Chronos Benchmark II — Win Rate & Skill Score
#         (from Chronos-2 paper, Table 5, arXiv:2510.15821v1)
#         Newer models (2025). Higher is better.
# ============================================================

chronos2_benchmark_ii = {
    # Model: (WQL Win%, WQL Skill%, MASE Win%, MASE Skill%)
    "Chronos-2":       (79.8, 46.6, 81.5, 26.5),
    "TiRex":           (70.4, 41.7, 67.1, 22.2),
    "TimesFM-2.5":     (70.0, 42.4, 71.6, 23.3),
    "Toto-1.0":        (60.9, 41.9, 58.0, 22.3),
    "Moirai-2.0":      (56.0, 40.9, 53.5, 19.8),
    "Chronos-Bolt":    (49.4, 39.3, 50.6, 20.4),
    "TabPFN-TS":       (46.3, 32.6, 40.1, 10.5),
    "COSMIC":          (42.8, 36.7, 42.0, 18.1),
    "Sundial":         (14.4, 24.1, 21.8,  9.5),
    "Seasonal Naive":  (10.1,  0.0, 13.8,  0.0),
}


# ============================================================
# Part 3: GIFT-Eval — Win Rate & Skill Score
#         (from Chronos-2 paper, Table 4)
# ============================================================

gift_eval_results = {
    # Model: (WQL Win%, WQL Skill%, MASE Win%, MASE Skill%)
    "Chronos-2":       (81.9, 51.4, 83.8, 30.2),
    "TimesFM-2.5":     (77.5, 51.0, 77.7, 29.5),
    "TiRex":           (76.5, 50.2, 71.9, 27.6),
    "Toto-1.0":        (67.4, 48.6, 61.3, 25.2),
    "Moirai-2.0":      (64.4, 48.4, 64.3, 27.2),
    "COSMIC":          (56.4, 44.5, 51.9, 20.8),
    "Chronos-Bolt":    (53.8, 42.6, 58.4, 19.2),
    "TabPFN-TS":       (53.5, 43.1, 45.4, 16.6),
    "Sundial":         (49.1, 44.1, 53.4, 25.0),
}


def print_benchmark_ii():
    print("=" * 90)
    print("Chronos Benchmark II (27 datasets, zero-shot)")
    print("Agg. Relative Scores (geometric mean of per-dataset model/SeasonalNaive)")
    print("Lower is better. 1.000 = Seasonal Naive baseline.")
    print("=" * 90)

    # Sort by WQL (use MASE as fallback for models without WQL)
    items = sorted(
        benchmark_ii_scores.items(),
        key=lambda x: (x[1][0] if x[1][0] is not None else 999, x[1][1])
    )

    print(f"\n{'Model':<28} {'Agg Rel WQL':>12} {'Agg Rel MASE':>13}  {'Category'}")
    print("-" * 85)
    for name, (wql, mase, cat) in items:
        wql_str = f"{wql:.3f}" if wql is not None else "N/A"
        print(f"{name:<28} {wql_str:>12} {mase:>13.3f}  {cat}")


def print_chronos2():
    print("\n\n" + "=" * 90)
    print("Chronos Benchmark II — Win Rate & Skill Score (Chronos-2 paper, 2025)")
    print("Higher is better.")
    print("=" * 90)

    print(f"\n{'Model':<20} {'WQL Win%':>10} {'WQL Skill%':>12} {'MASE Win%':>11} {'MASE Skill%':>13}")
    print("-" * 70)
    for name, (ww, ws, mw, ms) in sorted(chronos2_benchmark_ii.items(), key=lambda x: -x[1][0]):
        print(f"{name:<20} {ww:>10.1f} {ws:>12.1f} {mw:>11.1f} {ms:>13.1f}")


def print_gift_eval():
    print("\n\n" + "=" * 90)
    print("GIFT-Eval — Win Rate & Skill Score (Chronos-2 paper, 2025)")
    print("Higher is better.")
    print("=" * 90)

    print(f"\n{'Model':<20} {'WQL Win%':>10} {'WQL Skill%':>12} {'MASE Win%':>11} {'MASE Skill%':>13}")
    print("-" * 70)
    for name, (ww, ws, mw, ms) in sorted(gift_eval_results.items(), key=lambda x: -x[1][0]):
        print(f"{name:<20} {ww:>10.1f} {ws:>12.1f} {mw:>11.1f} {ms:>13.1f}")


if __name__ == "__main__":
    print_benchmark_ii()
    print_chronos2()
    print_gift_eval()
