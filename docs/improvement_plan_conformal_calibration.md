# GIFT-Eval 総合改善計画

**作成日**: 2026-03-19
**更新**: クラスタリング+MinT再統合、4層アーキテクチャとの整合

---

## 1. 現状の問題

### 1.1 確率予測の退化（致命的）

| 指標 | 幾何平均 (97構成) | 算術平均 |
|------|-------------------|---------|
| CRPS (mean_weighted_sum_quantile_loss) | 0.2374 | 0.3807 |
| ND (Normalized Deviation) | 0.2649 | 0.4195 |
| MASE | 1.7191 | 2.0648 |
| **CRPS / ND 相関** | **0.9938** | - |
| **CRPS / ND 比率** | - | **mean=0.898** |

原因: `run_gift_eval.py` の確率予測生成が訓練残差σベースのbootstrap。
N_SAMPLES=20、等方的ガウスノイズ → 全quantileが点予測に張り付き、CRPS ≈ 0.9×ND に退化。

### 1.2 系列間構造の未活用（高）

GIFT-Evalのデータセットは数千〜数万系列を含む:

| データセット | 系列数 | 系列間の関連性 |
|---|---:|---|
| bitbrains_fast_storage | 22,500 | 同一DCのVM群 → 強い |
| electricity | 7,400 | 同一グリッドの家庭群 → 強い |
| loop_seattle | 6,137 | 同一高速道路のセンサー → 強い |
| m4_daily | 4,227 | 多様な系列 → 弱い |
| solar | 2,603 | 同一地域のパネル群 → 強い |

現パイプラインは全系列を**独立に**処理。クラスタリングもMinTも使っていない。
M5での実績（WP-MO: 0.778 vs 直接: 0.785）で効果は確認済みだが、GIFT-Evalに未適用。

### 1.3 モデル選択の不在（高）

ADPC と Theta の2モデルを両方実行して比較するだけ。
per-series/per-cluster でのCV自動選択がない。

### GIFT-Evalリーダーボードとの比較 (2026-03-19時点)

| モデル | タイプ | CRPS | CRPS_Rank | MASE |
|--------|--------|------|-----------|------|
| TSOrchestra | agentic | 0.468 | 9.206 | 0.677 |
| DeOSAlphaTimeGPTPredictor-2 | zero-shot | 0.466 | 9.381 | 0.682 |
| TiRex | zero-shot | 0.488 | 16.34 | 0.716 |
| **WP-MO-Theta (ours)** | **statistical** | **0.237 (退化値)** | **-** | **1.719** |

---

## 2. 改善の全体像: 4層アーキテクチャ

詳細は `docs/architecture_redesign.md` 参照。

```
[Layer 0: Structure]   系列群 → クラスタリング → 階層構築
[Layer 1: Forecast]    各レベルで最適モデルによる点予測
[Layer 2: Reconcile]   MinT WLS → 階層間coherency
[Layer 3: Calibrate]   Conformal Prediction → 較正済み区間
```

### 各層の改善効果

| 層 | 改善内容 | 効果 | 対象指標 |
|----|---------|------|---------|
| Layer 0 | クラスタリング+階層構築 | 系列間の構造活用 | MASE, CRPS |
| Layer 1 | 5予測器 + per-cluster CV選択 | 点予測精度向上 | **MASE** |
| Layer 2 | MinT WLS | ノイズ平滑化+coherency | **MASE** |
| Layer 3 | Conformal Prediction | 確率予測の正常化 | **CRPS** |

---

## 3. 改善計画: 段階的実装

### Phase 1: Conformal Calibration（最優先、1-2日）

**目標**: CRPS退化問題の修正

現在の点予測（ADPC/Theta）はそのまま維持し、確率予測だけを修正する。

#### アルゴリズム: Temporal Conformal Prediction

```
1. 時系列CVで残差を収集:
   各分割点tで: train=y[:t], val=y[t:t+h]
   残差 r_h = val[h] - pred[h] をhorizon step別に蓄積

2. 全訓練データで最終モデル学習 → 点予測

3. 各quantile level q、各horizon step h で:
   q_forecast[q, h] = point_fc[h] + quantile(residuals[h], q)

4. 非負制約 + isotonic regression (crossing防止)
```

#### ポイント

- **out-of-sample残差**: 訓練残差ではなく、CVで得たholdout残差を使う
- **Horizon-dependent**: h=1とh=28で異なる不確実性を捕捉
- **Distribution-free**: ガウス仮定不要。スパイク/間欠需要もそのまま処理
- **予測器非依存**: ADPCでもThetaでも同じCalibratorが使える

#### 期待効果

| 指標 | 現在 | 目標 |
|------|------|------|
| CRPS / ND 相関 | 0.994 | < 0.85 |
| CRPS / ND 比率 | 0.90 | 0.6-0.7 |
| CRPS (幾何平均) | 0.237 (退化値) | 0.35-0.50 (適切な値) |
| MASE | 1.719 | 変化なし |

**注意**: CRPS数値は上がる（悪化に見える）が、それが正しい。
0.237は壊れた値であり、リーダーボードと比較可能な値に修正することが目的。

### Phase 2: 予測器拡充 + Per-series選択（1-2日）

**目標**: 点予測精度（MASE）の改善

#### 追加予測器

| 予測器 | 追加理由 |
|--------|---------|
| AutoETS | 季節性+トレンドの自動選択。Thetaと相補的 |
| CrostonTSB | 間欠需要（car_parts等）に必須 |

#### Per-series/cluster CV選択

```python
candidates = [ADPC, Theta, AutoETS, SeasonalNaive, CrostonTSB]
# 直近horizon分をvalidationに使い、MAEで最良モデルを自動選択
# 失敗したモデルはscore=∞で自然脱落（fallback連鎖不要）
```

### Phase 3: クラスタリング + MinT統合（2-3日）

**目標**: 系列間の構造を活用した点予測のさらなる改善

#### Layer 0: ACFクラスタリング

```
1. 各系列のACF特徴量を計算（seasonal_period分のlag）
2. ACF距離 + Average linkage で階層クラスタリング
3. クラスタ数: silhouette or CRPS-based CV
4. 3レベル階層を構築: Total → K clusters → N series
```

#### Layer 2: MinT WLS

```
1. Total / Cluster / Bottom の各レベルで独立に点予測
2. MinT WLS で全レベル整合
   → Total = Σ(Clusters) = Σ(Bottom) を保証
3. 集約レベルの情報がbottom-level予測を補正
```

#### 適応的スキップ

- **系列数 < 30**: 階層なし → Layer 0, 2 をスキップ
- **系列数 ≥ 30**: クラスタリング + MinT を適用

#### M5での実績

- Middle-Out + MinT: WRMSSE 0.778
- 直接予測のみ:    WRMSSE 0.785
- 改善: 0.9%（統計手法の中では有意な差）

### Phase 4: 高度な較正（余裕があれば）

| 改善 | 効果 | 工数 |
|------|------|------|
| SPCI (Adaptive Conformal) | 非定常時系列への適応 | 3-5日 |
| Quantile reconciliation | 階層間のquantile coherency | 2-3日 |
| Heteroscedastic calibration | level-dependentな不確実性 | 1日 |

---

## 4. 実装ロードマップ

```
[Phase 1] Conformal Calibrator実装（1-2日）
  → calibrate.py 新規作成
  → 既存点予測を維持、ノイズ部分だけ差し替え
  → GIFT-Eval 97構成で CRPS 効果測定
  ↓
[Phase 2] 予測器拡充 + ModelSelector（1-2日）
  → AutoETS, CrostonTSB 追加
  → per-series CV自動選択
  → GIFT-Eval で MASE 効果測定
  ↓
[Phase 3] Structure層 + MinT統合（2-3日）
  → ACFクラスタリング + 階層構築
  → MinT WLS（既存コードから抽出）
  → Middle-Out按分
  → GIFT-Eval で MASE/CRPS 効果測定
  ↓
[Phase 4] 高度な較正（3-5日、Phase 1-3の結果次第）
  → SPCI, Quantile reconciliation等

合計: Phase 1-3 で 5-7日、Phase 4 含めて 8-12日
```

---

## 5. 成功指標

### Phase 1完了後

| 指標 | 現在 | 目標 |
|------|------|------|
| CRPS / ND 相関 | 0.994 | < 0.85 |
| CRPS (幾何平均) | 0.237 (退化) | 0.35-0.50 (正常化) |

### Phase 3完了後

| 指標 | 現在 | 目標 |
|------|------|------|
| MASE (幾何平均) | 1.719 | < 1.5 |
| MASE (算術平均) | 2.065 | < 1.8 |
| CRPS_Rank | - | < 25 (97構成での平均順位) |

---

## 6. 外部アイデアの取捨選択

Bitbrainsデータに対するハイブリッド統計モデル提案からの評価:

### 採用

| アイデア | 適用先 | 理由 |
|---------|--------|------|
| **Conformal Calibration (SPCI)** | Layer 3 | 確率予測品質を根本的に改善。全97構成に汎用 |
| **ACF距離クラスタリング** | Layer 0 | 形状類似性を直接捉える。計算量も軽い |
| **CRPS-basedクラスタ数選択** | Layer 0 | 評価指標と直結した最適化 |

### 不採用

| アイデア | 理由 |
|---------|------|
| HSMM regime detection | Bitbrains特化（4/97構成）。全体への影響微小 |
| Hawkes過程 | 同上。実装コストに見合わない |
| VAR（多変量） | sparse VMには不安定。大半はunivariate評価 |
| GARCH(1,1) | Conformalがdistribution-freeで代替。GARCH追加の必要なし |

---

## 7. 参考文献

- SPCI: Xu & Xie (2021) "Sequential Predictive Conformal Inference for Time Series"
- ACI: Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Conformal + Time Series: Zaffran et al. (2022) "Adaptive Conformal Predictions for Time Series"
- MinT: Wickramasuriya et al. (2019) "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization"
- GIFT-Eval: Aksu et al. (2024) "GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation" (arXiv:2410.10393)
