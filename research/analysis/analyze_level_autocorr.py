#!/usr/bin/env python3
"""Level autocorrelation analysis for GIFT-Eval datasets.

Computes lag-1 autocorrelation of the Level series (L = mat.sum(axis=0))
and correlates with FLAIR's relMASE to test the hypothesis:
"FLAIR wins because Level is predictable (high autocorrelation)."

Usage:
    uv run python research/analysis/analyze_level_autocorr.py

Output:
    docs/flair_tech_report/fig_rank1_dominance.png  (updated 4-panel figure)
    results/rank1_dominance.csv  (updated with Level autocorrelation)
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


def compute_level_acf1(y, P):
    """Compute lag-1 autocorrelation of the Level series."""
    n = len(y)
    nc = n // P
    if nc < 5 or P < 2:
        return None
    usable = nc * P
    mat = y[-usable:].reshape(nc, P).T  # (P, nc)
    L = mat.sum(axis=0)  # (nc,)
    if len(L) < 5 or np.std(L) < 1e-10:
        return None
    # Lag-1 autocorrelation
    L_mean = L.mean()
    L_centered = L - L_mean
    var = np.sum(L_centered ** 2)
    if var < 1e-30:
        return None
    acf1 = float(np.sum(L_centered[:-1] * L_centered[1:]) / var)
    return acf1


def main():
    # Load existing rank-1 results
    existing = pd.read_csv("results/rank1_dominance.csv")
    existing_configs = set(existing["config"])

    # Load FLAIR relMASE
    flair_path = "results/15_gift_eval/all_results_flair_mdl_svdmodal.csv"
    sn_path = "/tmp/gift-eval-space/results/seasonal_naive/all_results.csv"
    flair_df = pd.read_csv(flair_path).set_index("dataset")
    sn_df = pd.read_csv(sn_path).set_index("dataset")

    acf_records = {}
    for ds_name, freq, terms in DATASET_CONFIGS:
        load_name = NAME_MAP.get(ds_name, ds_name)
        term = terms[0]
        config_name = f"{ds_name}/{freq}/{term}"
        print(f"  {config_name} ... ", end="", flush=True)

        ds_path = os.path.join(os.environ['GIFT_EVAL'], load_name)
        actual_name = f"{load_name}/{freq}" if os.path.isdir(os.path.join(ds_path, freq)) else load_name
        try:
            dataset = Dataset(name=actual_name, term=term, to_univariate=False)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        period = get_period(freq)
        acf1_list = []
        for i, (ts, _) in enumerate(dataset.test_data):
            target = ts["target"]
            y_arr = np.array(target, dtype=float)
            if y_arr.ndim == 2:
                series_list = [y_arr[v] for v in range(y_arr.shape[0])]
            else:
                series_list = [y_arr]

            for y in series_list:
                y = np.nan_to_num(y, nan=0.0)
                if len(y) < 10:
                    continue
                y_min = np.min(y)
                y_shift = max(1 - y_min, 1.0) if y_min <= 0 else 0.0
                y = y + y_shift
                P = select_period_bic(y, freq, period)
                a = compute_level_acf1(y, P)
                if a is not None:
                    acf1_list.append(a)

        if not acf1_list:
            print("no valid series")
            continue

        acf_records[config_name] = {
            "level_acf1_median": float(np.median(acf1_list)),
            "level_acf1_mean": float(np.mean(acf1_list)),
            "level_acf1_p10": float(np.percentile(acf1_list, 10)),
            "level_acf1_p90": float(np.percentile(acf1_list, 90)),
        }
        print(f"acf1_median={np.median(acf1_list):.3f}  (n={len(acf1_list)})")

    # Merge with existing
    for col in ["level_acf1_median", "level_acf1_mean", "level_acf1_p10", "level_acf1_p90"]:
        existing[col] = existing["config"].map(lambda c: acf_records.get(c, {}).get(col, None))

    existing.to_csv("results/rank1_dominance.csv", index=False)
    print(f"\nSaved: results/rank1_dominance.csv (updated with Level ACF)")

    # --- 4-panel visualization ---
    df = existing
    plt.rcParams['font.family'] = 'Hiragino Sans'
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # (a) Histogram of rank-1 ratios
    ax = axes[0, 0]
    ax.hist(df["rank1_median"], bins=20, color="#4a90d9", edgecolor="white", alpha=0.85)
    ax.axvline(x=df["rank1_median"].median(), color="#c0392b", linestyle="--",
               linewidth=2, label=f'Median = {df["rank1_median"].median():.1%}')
    ax.set_xlabel("Rank-1 variance ratio (σ₁² / Σσᵢ²)", fontsize=11)
    ax.set_ylabel("Number of configs", fontsize=11)
    ax.set_title("(a) Rank-1 dominance across GIFT-Eval", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # (b) Rank-1 ratio vs relMASE
    ax = axes[0, 1]
    valid = df.dropna(subset=["relMASE"])
    ax.scatter(valid["rank1_median"], valid["relMASE"],
               c="#4a90d9", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
    for _, row in valid.iterrows():
        if row["relMASE"] > 1.5 or row["rank1_median"] < 0.7:
            ax.annotate(row["config"], (row["rank1_median"], row["relMASE"]),
                        fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    corr_r1 = valid[["rank1_median", "relMASE"]].corr().iloc[0, 1]
    ax.set_xlabel("Rank-1 variance ratio", fontsize=11)
    ax.set_ylabel("FLAIR relMASE (lower = better)", fontsize=11)
    ax.set_title(f"(b) Rank-1 ratio vs accuracy (r={corr_r1:.2f})",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.15)

    # (c) Level ACF1 histogram
    ax = axes[1, 0]
    valid_acf = df.dropna(subset=["level_acf1_median"])
    ax.hist(valid_acf["level_acf1_median"], bins=20, color="#2ecc71", edgecolor="white", alpha=0.85)
    ax.axvline(x=valid_acf["level_acf1_median"].median(), color="#c0392b", linestyle="--",
               linewidth=2, label=f'Median = {valid_acf["level_acf1_median"].median():.2f}')
    ax.set_xlabel("Level lag-1 autocorrelation", fontsize=11)
    ax.set_ylabel("Number of configs", fontsize=11)
    ax.set_title("(c) Level predictability (ACF₁)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)

    # (d) Level ACF1 vs relMASE
    ax = axes[1, 1]
    valid_both = df.dropna(subset=["relMASE", "level_acf1_median"])
    ax.scatter(valid_both["level_acf1_median"], valid_both["relMASE"],
               c="#2ecc71", s=60, alpha=0.7, edgecolors="white", linewidths=0.5)
    for _, row in valid_both.iterrows():
        if row["relMASE"] > 1.5 or row["level_acf1_median"] < 0.3:
            ax.annotate(row["config"], (row["level_acf1_median"], row["relMASE"]),
                        fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    corr_acf = valid_both[["level_acf1_median", "relMASE"]].corr().iloc[0, 1]
    ax.set_xlabel("Level lag-1 autocorrelation", fontsize=11)
    ax.set_ylabel("FLAIR relMASE (lower = better)", fontsize=11)
    ax.set_title(f"(d) Level predictability vs accuracy (r={corr_acf:.2f})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)

    plt.suptitle(
        "Why FLAIR Works: Rank-1 structure + predictable Level dynamics",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("docs/flair_tech_report/fig_rank1_dominance.png",
                dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: docs/flair_tech_report/fig_rank1_dominance.png")
    plt.close()


if __name__ == "__main__":
    main()
