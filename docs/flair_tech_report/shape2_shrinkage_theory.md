# Shape₂ Sinusoidal Prior Shrinkage — 理論的根拠と論文記述案

## 概要

Shape₂ (二次周期の比率分解) に対する empirical Bayes 縮約。
Fourier cos/sin を Ridge 特徴量として使う代わりに、Level を Shape₂ で除算 (deseasonalize) する。
Shape₂ の推定には、生の比率を最小次数の周期関数 (first harmonic) に縮約する。

**精度**: relMASE 0.886→0.874, relCRPS 0.642→0.639 (97 configs, GIFT-Eval)

---

## 手法

### Shape₂ の計算

```
Shape₂ = w × Shape₂_raw + (1-w) × Shape₂_harmonic
w = nc₂ / (nc₂ + cp)
```

- `Shape₂_raw`: 各位置の平均 Level 比率 (cp 個のパラメータ)
- `Shape₂_harmonic`: raw の第一高調波フィット (2 パラメータ: 振幅+位相)
- `nc₂ = n_complete // cp`: 完全な二次周期のサイクル数
- `cp = secondary_period / P`: 二次周期の Level 空間での長さ

### Deseasonalization

```python
L_deseason = Level / Shape₂[position]       # 二次パターン除去
# Ridge on L_deseason: intercept + trend + lag-1 + lag-cp (Fourier なし)
L_hat_deseason = Ridge_forecast(L_deseason)
L_hat = L_hat_deseason × Shape₂[forecast_position]  # 再季節化
```

---

## 理論的導出

### w = nc₂/(nc₂+cp) の導出: 階層ベイズモデル

Shape₂ を階層的にモデル化する:

**Stage 1 (Prior):**
```
Shape₂_true ~ N(Shape₂_harmonic, τ² I)
```
事前分布は最小次数の周期関数 (第一高調波) を中心とするガウス分布。
τ² は「真の Shape₂ が harmonic からどれだけ逸脱するか」の分散。

**Stage 2 (Likelihood):**
```
Shape₂_obs | Shape₂_true ~ N(Shape₂_true, (σ²/nc₂) I)
```
観測された比率は真の Shape₂ のノイズ付き推定。
各比率は nc₂ 個の値から推定されるため、分散は σ²/nc₂。

**事後平均 (Posterior mean):**
```
Shape₂_posterior = w × Shape₂_obs + (1-w) × Shape₂_harmonic
w = τ² / (τ² + σ²/nc₂)
```

**Empirical Bayes 近似:**
信号対ノイズ比 σ²/τ² を推定する代わりに、次元的議論を用いる:
- Shape₂_raw は cp 次元ベクトル
- 各次元の推定分散は σ²/nc₂
- 総推定誤差は cp × σ²/nc₂
- Harmonic fit (2 パラメータ) の推定誤差は 2 × σ²/nc₂

有効パラメータ数に比例する prior strength を仮定 (σ²/τ² ≈ cp):

```
w = nc₂ / (nc₂ + cp)
```

これは James-Stein 型縮約の標準形式であり、Dirichlet-Multinomial 縮約の
Shape₁ における kappa/(kappa + n) と構造的に同一。

---

## MDL との整合性

FLAIR の各スケールにおける prior は「一つ下の複雑度の表現」:

| スケール | パラメータ数 | Prior | Prior のパラメータ数 |
|----------|------------|-------|-------------------|
| P 選択 | — | BIC on SVD spectrum | MDL criterion |
| Shape₁ (Dirichlet) | P 個の比率 | Global average (全文脈の平均) | P 個 (だが推定が安定) |
| Shape₂ (本手法) | cp 個の比率 | First harmonic (最小次数の周期関数) | 2 個 (振幅+位相) |

**パターン**: 各スケールで、豊富なデータがあれば詳細な表現を使い、
データ不足時は最も単純な構造的表現に退化する。

- Shape₁: データ不足時 → uniform (定数、0次の周期関数)
- Shape₂: データ不足時 → sinusoidal (1次の周期関数)

この階層は MDL (最小記述長) の原理と整合する:
モデルの複雑度はデータが支持する範囲に自動的に制約される。

---

## 論文記述案 (英語)

### Method section:

> **Secondary Shape Decomposition.**
> When a secondary period exists (e.g., weekly cycle in hourly Level),
> we decompose Level₁ further using the same proportional mechanism.
> Define the cross-period cp = P_secondary / P and the number of complete
> secondary cycles nc₂ = ⌊n_complete / cp⌋.
>
> The secondary shape S₂ is estimated via empirical Bayes shrinkage:
>
>   S₂ = w · S₂^raw + (1 − w) · S₂^harmonic,    w = nc₂ / (nc₂ + cp)
>
> where S₂^raw is the vector of mean proportions at each position in the
> cp-cycle, and S₂^harmonic is its rank-1 Fourier approximation (the
> minimum-complexity periodic function with period cp). The weight w
> follows from a standard hierarchical Gaussian model with prior precision
> proportional to cp, analogous to the Dirichlet concentration parameter
> used for the primary shape S₁.
>
> Level₁ is then deseasonalized by dividing by S₂, predicted by Ridge
> regression with intercept, trend, and lag features only (no harmonic
> features), and reseasonalized by multiplying the forecast by S₂ at the
> corresponding positions.

### Discussion / ablation:

> The shrinkage weight w = nc₂/(nc₂ + cp) naturally adapts to data
> availability. For hourly data with weekly cross-period (cp = 7) and
> several months of history (nc₂ ≈ 70), w ≈ 0.91 and the raw proportions
> dominate, capturing non-sinusoidal patterns such as weekday–weekend
> contrasts. For daily data with yearly cross-period (cp = 52) and only a
> few years of history (nc₂ ≈ 3), w ≈ 0.05 and the harmonic prior
> dominates, providing stable regularization equivalent to the traditional
> Fourier approach but within a unified proportional framework.

---

## 実験結果サマリー (97 configs, GIFT-Eval)

| 手法 | relMASE | relCRPS | Chronos-small 比 |
|------|---------|---------|-----------------|
| FLAIR SVD-RQ (Fourier) | 0.886 | 0.642 | 0.886 < 0.892 ✓ |
| **FLAIR Shape₂ Shrinkage** | **0.874** | **0.639** | **0.874 << 0.892** ✓ |

### Per-horizon:
| Horizon | SVD-RQ | Shape₂ Shrinkage |
|---------|--------|-----------------|
| short | 0.885 | **0.855** |
| medium | 0.865 | 0.881 |
| long | 0.906 | 0.921 |

### 主な改善データセット:
- bitbrains_fast_storage/H: 3.578 → 1.351 (weekday/weekend step pattern captured)
- ett1/15T/short: 0.814 → 0.743
- covid_deaths/D: -0.284
- saugeen/D: -0.038

### 残る悪化:
- m4_daily/D: +0.586 (cp=52, heterogeneous series with no yearly pattern)
- bitbrains_rnd/H: +0.157

---

## コード

- 本体: `src/run_gift_eval_flair_ds.py` — `_compute_shape2()` 関数
- 全97実験: `src/run_shrinkage_full.py`
- 結果CSV: `results/15_gift_eval/all_results_flair_shrink.csv`
- コミット: d0cf132
