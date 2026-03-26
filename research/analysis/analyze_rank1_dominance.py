#!/usr/bin/env python3
"""Rank-1 dominance analysis for GIFT-Eval datasets.

Demonstrates that periodic time series are approximately rank-1
when reshaped by their period — the theoretical foundation of FLAIR.

Usage:
    uv run python research/analysis/analyze_rank1_dominance.py

Output:
    docs/flair_tech_report/fig_rank1_dominance.png
    results/rank1_dominance.csv
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

os.environ['GIFT_EVAL'] = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'gift-eval')

from gift_eval.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks'))
from run_gift_eval_flar9 import get_period, get_periods

warnings.filterwarnings('ignore')

DATASET_CONFIGS = [
    ("bitbrains_fast_storage", "5T", ["short"]),
    ("bitbrains_fast_storage", "H", ["short"]),
    ("bitbrains_rnd", "5T", ["short"]),
    ("bitbrains_rnd", "H", ["short"]),
    ("bizitobs_l2c", "5T", ["short"]),
    ("bizitobs_l2c", "H", ["short"]),
    ("car_parts", "M", ["short"]),
    ("covid_deaths", "D", ["short"]),
    ("electricity", "15T", ["short"]),
    ("electricity", "D", ["short"]),
    ("electricity", "H", ["short"]),
    ("electricity", "W", ["short"]),
    ("ett1", "15T", ["short"]),
    ("ett1", "D", ["short"]),
    ("ett1", "H", ["short"]),
    ("ett2", "15T", ["short"]),
    ("ett2", "D", ["short"]),
    ("ett2", "H", ["short"]),
    ("hierarchical_sales", "D", ["short"]),
    ("hierarchical_sales", "W", ["short"]),
    ("hospital", "M", ["short"]),
    ("jena_weather", "10T", ["short"]),
    ("jena_weather", "D", ["short"]),
    ("jena_weather", "H", ["short"]),
    ("kdd_cup_2018", "D", ["short"]),
    ("kdd_cup_2018", "H", ["short"]),
    ("loop_seattle", "5T", ["short"]),
    ("loop_seattle", "D", ["short"]),
    ("loop_seattle", "H", ["short"]),
    ("m4_daily", "D", ["short"]),
    ("m4_hourly", "H", ["short"]),
    ("m4_monthly", "M", ["short"]),
    ("m4_quarterly", "Q", ["short"]),
    ("m4_weekly", "W", ["short"]),
    ("m4_yearly", "A", ["short"]),
    ("m_dense", "D", ["short"]),
    ("m_dense", "H", ["short"]),
    ("restaurant", "D", ["short"]),
    ("saugeen", "D", ["short"]),
    ("saugeen", "M", ["short"]),
    ("saugeen", "W", ["short"]),
    ("solar", "10T", ["short"]),
    ("solar", "D", ["short"]),
    ("solar", "H", ["short"]),
    ("solar", "W", ["short"]),
    ("sz_taxi", "15T", ["short"]),
    ("sz_taxi", "H", ["short"]),
    ("temperature_rain", "D", ["short"]),
    ("us_births", "D", ["short"]),
    ("us_births", "M", ["short"]),
    ("us_births", "W", ["short"]),
]

NAME_MAP = {
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
    "temperature_rain": "temperature_rain_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE", "sz_taxi": "SZ_TAXI", "saugeen": "saugeenday",
}


def select_period_bic(y, freq_str, period):
    """Same MDL/BIC period selection as FLAIR-DS."""
    cal = get_periods(freq_str)
    candidates = cal if cal else [period]
    candidates = [p for p in candidates if p >= 1 and len(y) // p >= 5]
    if not candidates:
        return 1

    if len(candidates) == 1:
        return candidates[0]

    n = len(y)
    T_max = min(n, 500 * min(candidates))
    y_sel = y[-T_max:]
    best_P, best_bic = candidates[0], np.inf
    for p_cand in candidates:
        nc = T_max // p_cand
        if nc < 5:
            continue
        mat_c = y_sel[-(nc * p_cand):].reshape(nc, p_cand).T
        s = np.linalg.svd(mat_c, compute_uv=False)
        rss1 = np.sum(s[1:] ** 2)
        T = nc * p_cand
        penalty = (p_cand + nc - 1) * np.log(T)
        bic = T * np.log(max(rss1 / T, 1e-30)) + penalty
        if bic < best_bic:
            best_P, best_bic = p_cand, bic
    return best_P


def compute_rank1_ratio(y, P):
    """Compute sigma_1^2 / sum(sigma_i^2) for reshaped matrix."""
    n = len(y)
    nc = n // P
    if nc < 3 or P < 2:
        return None
    usable = nc * P
    mat = y[-usable:].reshape(nc, P).T  # (P, nc)
    s = np.linalg.svd(mat, compute_uv=False)
    total = np.sum(s ** 2)
    if total < 1e-30:
        return None
    return float(s[0] ** 2 / total)


def main():
    # Load FLAIR relMASE per config for correlation
    flair_path = "results/15_gift_eval/all_results_flair_mdl_svdmodal.csv"
    sn_path = "/tmp/gift-eval-space/results/seasonal_naive/all_results.csv"
    flair_df = pd.read_csv(flair_path).set_index("dataset")
    sn_df = pd.read_csv(sn_path).set_index("dataset")

    records = []
    for ds_name, freq, terms in DATASET_CONFIGS:
        load_name = NAME_MAP.get(ds_name, ds_name)
        term = terms[0]
        config_name = f"{ds_name}/{freq}/{term}"
        print(f"  {config_name} ... ", end="", flush=True)

        # Build load path: some datasets have freq subdirectories
        ds_path = os.path.join(os.environ['GIFT_EVAL'], load_name)
        actual_name = f"{load_name}/{freq}" if os.path.isdir(os.path.join(ds_path, freq)) else load_name
        try:
            dataset = Dataset(name=actual_name, term=term, to_univariate=False)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        period = get_period(freq)
        ratios = []
        for i, (ts, _) in enumerate(dataset.test_data):
            target = ts["target"]
            if isinstance(target, np.ndarray):
                y_arr = target.astype(float)
            else:
                y_arr = np.array(target, dtype=float)
            # Handle multivariate: analyze each variable
            if y_arr.ndim == 2:
                series_list = [y_arr[v] for v in range(y_arr.shape[0])]
            else:
                series_list = [y_arr]

            for y in series_list:
                y = np.nan_to_num(y, nan=0.0)
                if len(y) < 10:
                    continue
                # Location shift (same as FLAIR)
                y_min = np.min(y)
                y_shift = max(1 - y_min, 1.0) if y_min <= 0 else 0.0
                y = y + y_shift

                P = select_period_bic(y, freq, period)
                r = compute_rank1_ratio(y, P)
                if r is not None:
                    ratios.append(r)

        if not ratios:
            print("no valid series")
            continue

        median_r = float(np.median(ratios))
        mean_r = float(np.mean(ratios))
        p10 = float(np.percentile(ratios, 10))
        p90 = float(np.percentile(ratios, 90))

        # Get relMASE for this config
        config_key = f"{ds_name}/{freq}/{term}"
        rel_mase = None
        if config_key in flair_df.index and config_key in sn_df.index:
            m_f = flair_df.loc[config_key, "eval_metrics/MASE[0.5]"]
            m_s = sn_df.loc[config_key, "eval_metrics/MASE[0.5]"]
            if m_s > 0 and m_f > 0:
                rel_mase = float(m_f / m_s)

        records.append({
            "config": config_name,
            "dataset": ds_name,
            "freq": freq,
            "period": period,
            "n_series": len(ratios),
            "rank1_median": median_r,
            "rank1_mean": mean_r,
            "rank1_p10": p10,
            "rank1_p90": p90,
            "relMASE": rel_mase,
        })
        print(f"median={median_r:.3f}  (n={len(ratios)})")

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/rank1_dominance.csv", index=False)
    print(f"\nSaved: results/rank1_dominance.csv ({len(df)} configs)")

    # --- Visualization ---
    plt.rcParams['font.family'] = 'Hiragino Sans'
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) Histogram of rank-1 ratios
    ax = axes[0]
    ax.hist(df["rank1_median"], bins=20, color="#4a90d9", edgecolor="white",
            alpha=0.85)
    ax.axvline(x=df["rank1_median"].median(), color="#c0392b", linestyle="--",
               linewidth=2, label=f'Median = {df["rank1_median"].median():.1%}')
    ax.set_xlabel("Rank-1 variance ratio (σ₁² / Σσᵢ²)", fontsize=11)
    ax.set_ylabel("Number of configs", fontsize=11)
    ax.set_title("(a) Distribution of rank-1 dominance", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # (b) Rank-1 ratio vs relMASE scatter
    ax = axes[1]
    valid = df.dropna(subset=["relMASE"])
    ax.scatter(valid["rank1_median"], valid["relMASE"],
               c="#4a90d9", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
    # Annotate outliers
    for _, row in valid.iterrows():
        if row["relMASE"] > 1.5 or row["rank1_median"] < 0.5:
            ax.annotate(row["config"], (row["rank1_median"], row["relMASE"]),
                        fontsize=6, alpha=0.7, xytext=(4, 4),
                        textcoords="offset points")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    # Correlation
    corr = valid[["rank1_median", "relMASE"]].corr().iloc[0, 1]
    ax.set_xlabel("Rank-1 variance ratio", fontsize=11)
    ax.set_ylabel("FLAIR relMASE (lower = better)", fontsize=11)
    ax.set_title(f"(b) Rank-1 ratio vs FLAIR accuracy (r={corr:.2f})",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.15)

    # (c) Sorted bar chart by dataset
    ax = axes[2]
    sorted_df = df.sort_values("rank1_median", ascending=True)
    colors = ["#c0392b" if r < 0.8 else "#4a90d9" for r in sorted_df["rank1_median"]]
    bars = ax.barh(range(len(sorted_df)), sorted_df["rank1_median"],
                   color=colors, alpha=0.85, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["config"], fontsize=5.5)
    ax.axvline(x=0.9, color="gray", linestyle="--", alpha=0.4,
               label="90% threshold")
    ax.set_xlabel("Rank-1 variance ratio", fontsize=11)
    ax.set_title("(c) Per-config rank-1 dominance", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="x")

    plt.suptitle("Rank-1 Dominance: Periodic time series are approximately rank-1",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs("docs/flair_tech_report", exist_ok=True)
    plt.savefig("docs/flair_tech_report/fig_rank1_dominance.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: docs/flair_tech_report/fig_rank1_dominance.png")
    plt.close()


if __name__ == "__main__":
    main()
