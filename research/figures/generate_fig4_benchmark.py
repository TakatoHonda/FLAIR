"""
Generate fig4_benchmark.png for FLAIR Technical Report.

Scatter plot: relMASE vs relCRPS for all GIFT-Eval methods
(excluding agentic methods and those with test data leakage).

Usage:
    python research/figures/generate_fig4_benchmark.py

Output:
    docs/flair_tech_report/fig4_benchmark.png
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GIFT_EVAL_RESULTS = "/tmp/gift-eval-space/results"
OUTPUT_PATH = "docs/flair_tech_report/fig4_benchmark.png"

# FLAIR scores (our method, not on the leaderboard)
FLAIR_SCORES = {"relMASE": 0.864, "relCRPS": 0.614}

# Local results not on the leaderboard
LOCAL_RESULTS = {
    "Prophet": "results/15_gift_eval/all_results_prophet.csv",
    "MFLES": "results/15_gift_eval/all_results_mfles.csv",
}

# Models to exclude (extreme outliers that distort axes)
# Extreme outliers + orchestration systems (not single-model methods)
EXCLUDE_MODELS = {"auto_ets", "crossformer", "VISIT-1.0", "DeOSAlphaTimeGPTPredictor-2025", "FLAIR"}


def load_model_configs(results_dir: str) -> dict:
    configs = {}
    for d in Path(results_dir).iterdir():
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                configs[d.name] = json.load(f)
    return configs


def _compute_one(name: str, df: pd.DataFrame, sn_df: pd.DataFrame) -> dict | None:
    common = df.index.intersection(sn_df.index)
    if len(common) < 10:
        return None
    mase_model = df.loc[common, "eval_metrics/MASE[0.5]"]
    mase_sn = sn_df.loc[common, "eval_metrics/MASE[0.5]"]
    crps_model = df.loc[common, "eval_metrics/mean_weighted_sum_quantile_loss"]
    crps_sn = sn_df.loc[common, "eval_metrics/mean_weighted_sum_quantile_loss"]
    valid = (mase_sn > 0) & (crps_sn > 0) & (mase_model > 0) & (crps_model > 0)
    valid = valid & mase_model.notna() & crps_model.notna()
    if valid.sum() < 10:
        return None
    rel_mase = mase_model[valid] / mase_sn[valid]
    rel_crps = crps_model[valid] / crps_sn[valid]
    return {
        "model": name,
        "relMASE": np.exp(np.mean(np.log(rel_mase))),
        "relCRPS": np.exp(np.mean(np.log(rel_crps))),
        "n_configs": int(valid.sum()),
    }


def compute_rel_scores(results_dir: str) -> pd.DataFrame:
    """Compute relMASE and relCRPS (geometric mean) for each model."""
    sn_path = Path(results_dir) / "seasonal_naive" / "all_results.csv"
    sn_df = pd.read_csv(sn_path).set_index("dataset")

    records = []
    # Leaderboard models
    for d in Path(results_dir).iterdir():
        if not d.is_dir():
            continue
        csv_path = d / "all_results.csv"
        if not csv_path.exists():
            continue
        name = d.name
        if name in ("seasonal_naive", "naive"):
            continue
        df = pd.read_csv(csv_path).set_index("dataset")
        rec = _compute_one(name, df, sn_df)
        if rec:
            records.append(rec)

    # Local models (Prophet, MFLES, etc.)
    for name, csv_path in LOCAL_RESULTS.items():
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path).set_index("dataset")
        rec = _compute_one(name, df, sn_df)
        if rec:
            records.append(rec)

    return pd.DataFrame(records)


def categorize_model(model_type: str) -> str:
    if model_type == "statistical":
        return "Statistical"
    elif model_type == "deep-learning":
        return "Deep Learning"
    elif model_type in ("pretrained", "zero-shot", "fine-tuned"):
        return "Foundation Model"
    elif model_type == "agentic":
        return "Agentic"
    return "Other"


def main():
    configs = load_model_configs(GIFT_EVAL_RESULTS)
    scores = compute_rel_scores(GIFT_EVAL_RESULTS)

    # Merge with configs
    # Manual overrides for local models not in leaderboard configs
    local_meta = {
        "Prophet": {"model_type": "statistical", "testdata_leakage": "No"},
        "MFLES":   {"model_type": "statistical", "testdata_leakage": "No"},
    }
    scores["model_type"] = scores["model"].map(
        lambda m: local_meta.get(m, configs.get(m, {})).get("model_type", "unknown")
    )
    scores["leakage"] = scores["model"].map(
        lambda m: local_meta.get(m, configs.get(m, {})).get("testdata_leakage", "No")
    )
    scores["category"] = scores["model_type"].map(categorize_model)

    # Filter: no agentic, no leakage, no excluded outliers
    mask = (
        (scores["category"] != "Agentic")
        & (scores["leakage"] != "Yes")
        & (~scores["model"].isin(EXCLUDE_MODELS))
    )
    filtered = scores[mask].copy()

    # Add SeasonalNaive at (1.0, 1.0)
    sn_row = pd.DataFrame([{
        "model": "SeasonalNaive",
        "relMASE": 1.0,
        "relCRPS": 1.0,
        "n_configs": 97,
        "model_type": "statistical",
        "leakage": "No",
        "category": "Statistical",
    }])
    filtered = pd.concat([filtered, sn_row], ignore_index=True)

    # Print summary
    print(f"Total models on leaderboard: {len(scores)}")
    print(f"After filtering (no agentic, no leakage): {len(filtered)}")
    print(f"\n{'Model':<30s} {'Category':<20s} {'relMASE':>8s} {'relCRPS':>8s} {'N':>4s}")
    print("-" * 75)
    for _, row in filtered.sort_values("relMASE").iterrows():
        print(f"{row['model']:<30s} {row['category']:<20s} {row['relMASE']:8.3f} {row['relCRPS']:8.3f} {row['n_configs']:4.0f}")

    # --- Plot ---
    plt.rcParams['font.family'] = 'Hiragino Sans'

    category_styles = {
        "Foundation Model": {"color": "#4a90d9", "marker": "s", "label": "Foundation Models (GPU, pre-trained)"},
        "Deep Learning":    {"color": "#d97b2a", "marker": "D", "label": "Deep Learning (GPU, trained per dataset)"},
        "Statistical":      {"color": "#7eb87e", "marker": "^", "label": "Statistical (CPU, no training)"},
    }

    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot each category
    for cat, style in category_styles.items():
        subset = filtered[filtered["category"] == cat]
        if subset.empty:
            continue
        ax.scatter(
            subset["relMASE"], subset["relCRPS"],
            c=style["color"], marker=style["marker"],
            s=80, alpha=0.85, label=style["label"],
            edgecolors="white", linewidths=0.5, zorder=3,
        )

    # Plot FLAIR as a star
    ax.scatter(
        FLAIR_SCORES["relMASE"], FLAIR_SCORES["relCRPS"],
        c="#c0392b", marker="*", s=350, zorder=5,
        label="FLAIR (CPU, one SVD, 0 hyperparams)",
        edgecolors="darkred", linewidths=0.5,
    )

    # Label all points
    label_map = {
        "auto_arima": "AutoARIMA",
        "auto_ets": "AutoETS",
        "auto_theta": "AutoTheta",
        "chronos-2": "Chronos-2",
        "chronos-2-synth": "Chronos-2-synth",
        "TimesFM-2.5": "TimesFM-2.5",
        "Toto_Open_Base_1.0": "Toto",
        "Moirai2": "Moirai2",
        "Moirai_base": "Moirai-base",
        "Moirai_large": "Moirai-large",
        "Moirai_small": "Moirai-small",
        "N-BEATS": "N-BEATS",
        "PatchTST": "PatchTST",
        "PatchTST-FM-r1": "PatchTST-FM",
        "FlowState-9.1M": "FlowState",
        "granite-flowstate-r1": "Granite-FS",
        "CleanTS-65M": "CleanTS",
        "Kairos_10m": "Kairos-10m",
        "Kairos_23m": "Kairos-23m",
        "Kairos_50m": "Kairos-50m",
        "Reverso": "Reverso",
        "Reverso-Small": "Reverso-S",
        "Reverso-Nano": "Reverso-N",
        "Xihe-max": "Xihe-max",
        "Xihe-ultra": "Xihe-ultra",
        "YingLong_300m": "YingLong-300m",
        "YingLong_110m": "YingLong-110m",
        "YingLong_50m": "YingLong-50m",
        "YingLong_6m": "YingLong-6m",
        "sundial_base_128m": "Sundial",
        "tabpfn_ts": "TabPFN-TS",
        "TempoPFN": "TempoPFN",
        "TiRex": "TiRex",
        "Timer-s1": "Timer-s1",
        "VISIT-1.0": "VISIT-1.0",
        "VISIT-2.0": "VISIT-2.0",
        "visionts": "VisionTS",
        "DeOSAlphaTimeGPTPredictor-2025": "DeOS-Alpha",
        "Lingjiang": "Lingjiang",
        "Migas-1.0": "Migas",
        "FFM": "FFM",
        "crossformer": "Crossformer",
        "deepar": "DeepAR",
        "iTransformer": "iTransformer",
        "tft": "TFT",
        "tide": "TiDE",
        "xLSTM-Mixer": "xLSTM-Mixer",
        "Samay": "Samay",
        "DLinear": "DLinear",
        "SeasonalNaive": "SeasonalNaive",
        "Prophet": "Prophet",
        "MFLES": "MFLES",
    }

    # Label ALL points
    for _, row in filtered.iterrows():
        m = row["model"]
        label_text = label_map.get(m, m)
        ax.annotate(
            label_text,
            (row["relMASE"], row["relCRPS"]),
            fontsize=6.5, alpha=0.75,
            textcoords="offset points", xytext=(6, 4),
        )

    # Label FLAIR
    ax.annotate(
        "FLAIR",
        (FLAIR_SCORES["relMASE"], FLAIR_SCORES["relCRPS"]),
        fontsize=10, fontweight="bold", color="#c0392b",
        textcoords="offset points", xytext=(8, -4),
    )

    # Reference lines at SeasonalNaive
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("relMASE (geometric mean, lower = better)", fontsize=11)
    ax.set_ylabel("relCRPS (geometric mean, lower = better)", fontsize=11)

    n_total = len(filtered) + 1  # +1 for FLAIR
    ax.set_title(
        f"GIFT-Eval Benchmark: {n_total} methods (non-agentic, no test leakage)",
        fontsize=13, fontweight="bold",
    )

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
