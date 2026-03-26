# FLAIR vs Chronos-Small — Monash/Chronos Benchmark

## Evaluation Protocol
- Chronos論文 (Ansari et al., 2024) と同一のzero-shotベンチマーク
- 25データセット (ETTh, ETTm除く)
- 評価指標: MASE (median), WQL (weighted quantile loss)
- Relative Score = method / Seasonal Naive (低いほど良い)
- Aggregate Relative Score = Geometric Mean of Relative Scores

## Results Summary

| Metric | FLAIR-MDL | Chronos-Small (46M, GPU) |
|--------|-----------|--------------------------|
| **Agg Relative MASE** | **0.704** | 0.839 |
| Agg Relative WQL | 0.850 | **0.675** |
| MASE wins | **15/25** | 10/25 |
| WQL wins | 7/25 | **18/25** |

## Key Findings

### FLAIR dominates on point forecast (MASE)
- Agg Relative MASE: **0.704 vs 0.839** (FLAIR 16% better)
- FLAIR wins 15/25 datasets on MASE
- Particularly strong on: m5 (0.375 vs 0.670), covid_deaths (0.090 vs 0.902), car_parts (0.292 vs 0.743)

### Chronos dominates on probabilistic forecast (WQL)
- Agg Relative WQL: 0.850 vs **0.675** (Chronos 21% better)
- Chronos wins 18/25 datasets on WQL
- FLAIR's conformal intervals are less calibrated than Chronos's learned quantiles

### FLAIR's strengths (MASE wins)
- **exchange_rate**: 0.876 vs 1.043 (Chronos > SN!)
- **m5**: 0.375 vs 0.670 (FLAIR nearly 2x better)
- **monash_tourism_monthly**: 0.960 vs 1.180 (Chronos > SN!)
- **monash_m1_yearly**: 0.707 vs 0.969

### FLAIR's weaknesses
- **WQL on most datasets**: conformal intervals are wider/less calibrated
- **monash_traffic**: 1.024 vs 0.761 (hourly traffic, FLAIRはSN以下)
- **monash_fred_md**: 0.778 vs 0.431 (Chronos much better)
- **nn5**: 0.651 vs 0.474

## Full Results Table

| Dataset | FLAIR relMASE | Chronos relMASE | Winner | FLAIR relWQL | Chronos relWQL | Winner |
|---------|--------------|-----------------|--------|-------------|----------------|--------|
| dominick | 0.650 | 0.931 | FLAIR | 1.263 | 0.744 | Chronos |
| ercot | 0.815 | 0.742 | Chronos | 0.568 | 0.432 | Chronos |
| exchange_rate | 0.876 | 1.043 | FLAIR | 0.923 | 1.077 | FLAIR |
| m4_quarterly | 0.899 | 0.775 | Chronos | 0.824 | 0.706 | Chronos |
| m4_yearly | 0.848 | 0.941 | FLAIR | 0.851 | 0.857 | FLAIR |
| m5 | 0.375 | 0.670 | FLAIR | 0.617 | 0.576 | Chronos |
| monash_australian_electricity | 0.540 | 0.977 | FLAIR | 1.357 | 0.833 | Chronos |
| monash_car_parts | 0.292 | 0.743 | FLAIR | 0.856 | 0.644 | Chronos |
| monash_cif_2016 | 0.784 | 0.791 | FLAIR | 1.933 | 1.000 | Chronos |
| monash_covid_deaths | 0.090 | 0.902 | FLAIR | 1.218 | 0.474 | Chronos |
| monash_fred_md | 0.778 | 0.431 | Chronos | 1.008 | 0.123 | Chronos |
| monash_hospital | 0.923 | 0.771 | Chronos | 0.877 | 0.781 | Chronos |
| monash_m1_monthly | 0.956 | 0.892 | Chronos | 0.806 | 0.723 | Chronos |
| monash_m1_quarterly | 0.880 | 0.870 | Chronos | 0.747 | 0.753 | FLAIR |
| monash_m1_yearly | 0.707 | 0.969 | FLAIR | 0.536 | 0.828 | FLAIR |
| monash_m3_monthly | 0.882 | 0.773 | Chronos | 0.758 | 0.671 | Chronos |
| monash_m3_quarterly | 1.023 | 0.898 | Chronos | 0.941 | 0.802 | Chronos |
| monash_m3_yearly | 0.921 | 1.066 | FLAIR | 0.964 | 0.940 | Chronos |
| monash_nn5_weekly | 0.768 | 0.873 | FLAIR | 0.675 | 0.732 | FLAIR |
| monash_tourism_monthly | 0.960 | 1.180 | FLAIR | 0.981 | 1.048 | FLAIR |
| monash_tourism_quarterly | 0.998 | 1.037 | FLAIR | 0.605 | 0.580 | Chronos |
| monash_tourism_yearly | 0.992 | 1.123 | FLAIR | 1.033 | 0.957 | Chronos |
| monash_traffic | 1.024 | 0.761 | Chronos | 0.931 | 0.710 | Chronos |
| monash_weather | 0.605 | 0.852 | FLAIR | 0.747 | 0.682 | Chronos |
| nn5 | 0.651 | 0.474 | Chronos | 0.393 | 0.395 | FLAIR |

## Implications for FLAIR paper
1. **Point forecast (MASE)**: FLAIR is significantly better than Chronos-Small
2. **Probabilistic forecast (WQL)**: FLAIR's conformal intervals need improvement
3. **Paper narrative**: "0-param CPU method beats 46M-param GPU model on point accuracy"
4. **Next step**: Improve conformal interval calibration to close the WQL gap
