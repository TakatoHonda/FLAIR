# Ridge回帰ベース時系列予測の改善テクニック: FLAR向け調査レポート

**作成日**: 2026-03-21
**対象**: FLAR (Fourier-Lag Autoregressive Ridge) の改善候補

---

## 0. FLAR現構造の整理

```
特徴量: [intercept, linear_trend/n, cos(2pi*t/p), sin(2pi*t/p) x K harmonics, lag-1, seasonal-lag]
変換:   Box-Cox(y+1, lambda), lambda in [0,1] clipped
正則化: GCV alpha選択 (25点 logspace[-4,4])、soft-average variant
予測:   Recursive multi-step
不確実性: LOO残差のbootstrap + jitter
```

---

## 1. 特徴量設計の改善

### 1.1 時変トレンド (Time-Varying Trend)

**現状**: `trend = t/n` (線形トレンドのみ)

**改善案A: 区分線形トレンド (Piecewise Linear)**
- 時系列の後半に重みを置く区分線形基底を追加
- 例: `max(0, t - t_break)` を1-2本追加（t_breakは3/4地点など）
- Ridgeが不要な区分を自動的に縮退させる

**改善案B: 低次多項式トレンド**
- `t/n`, `(t/n)^2` の2次まで追加
- 過学習リスクはRidgeの正則化で制御可能
- ただし外挿時に発散リスクあり → recursive forecastで増幅される危険

**推奨**: 区分線形が安全。2次多項式はrecursive forecastで発散するリスクが高く、理論的にも区分線形の方がRidgeとの相性が良い。

### 1.2 Cross-Period Accumulation (2025年の最新研究)

**[An optimized ridge regression for forecasting time series with a fixed period (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002971)**

- 複数の効率的な累積操作を用いてクロスピリオド累積を実行
- 季節指標をモデル構造にパラメータとして埋め込み、データから直接推定
- FLARの `_focused_fourier_periods` を拡張する形で統合可能

**FLARへの適用**: 現在の固定Fourier harmonicsに加え、周期ごとの平均値（季節指標）を特徴量として追加。Ridgeが自動的に重要度を判断する。

### 1.3 Trend x Seasonality Interaction

**現状**: トレンドとFourier項は独立

**改善案**: `trend * cos(2pi*t/p)` と `trend * sin(2pi*t/p)` のinteraction項を追加
- 季節パターンが時間とともに変化する系列（振幅の増大/減少）を捕捉
- [時変季節性のモデリング](https://ecogambler.netlify.app/blog/time-varying-seasonality/) で有効性が確認されている
- 特徴量数が2倍になるが、Ridgeが不要な項を自動縮退

**注意**: v3 ablationでこのinteractionは全滅した（MEMORY.mdより）。ただし、他の改善（soft-averageなど）と組み合わせた場合に再検討の余地あり。

### 1.4 Multi-Period Calendar Table (v5で実装済み)

FLARv5の `FREQ_TO_PERIODS` テーブルで既に実装。例:
- `H → [24, 168]` (daily + weekly)
- `D → [7, 365]` (weekly + yearly)

v5 ablationの結果を踏まえて判断が必要。

### 1.5 Random Fourier Features / RBF基底 (非線形拡張)

- [Random Fourier Features for Kernel Ridge Regression (Avron et al.)](https://arxiv.org/abs/1804.09893) の理論的保証
- `cos(w_j^T x + b_j)` の形でD個のランダム特徴量を生成
- カーネルRidgeの近似として、計算量O(nD^2)で非線形性を導入
- **FLARへの適用**: lag特徴量に対してRFF変換を適用すれば、非線形なlag依存性を捕捉できる
- **懸念**: 特徴量空間の解釈性が失われる。FLARの「シンプルさ」という長所を損なう可能性

---

## 2. 正則化の改善

### 2.1 Generalized Ridge (構造化ペナルティ行列)

**[Generalized Ridge Regression (2024)](https://arxiv.org/pdf/2407.02583)**

通常のRidgeは `||beta||^2` で全係数を等しく縮退させるが、Generalized Ridgeは各係数に異なるペナルティを適用:

```
minimize ||y - X*beta||^2 + beta^T * Omega * beta
```

ここで `Omega` は対角行列ではなく、事前知識を反映した構造化ペナルティ行列。

**FLARへの具体的適用**:
- **intercept, trend**: 小さいペナルティ（常に必要）
- **Fourier harmonics**: 中程度のペナルティ（高調波ほど大きく）
- **lag-1, seasonal-lag**: 別個のペナルティ（autoregressive部分は重要度が系列依存）

**利点**: 方向ごとに異なる縮退率を設定でき、等方的Ridgeより厳密にMSEが小さくなる場合がある。

**実装**: GCVの拡張が必要だが、SVD分解を `Omega^{1/2}` で事前変換すれば既存のGCVフレームワークで対応可能。

### 2.2 Bayesian Ridge

**[scikit-learn BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)**

- alpha（正則化）とsigma^2（ノイズ分散）を同時にEM推定
- GCVよりも理論的に美しい: 事後分布からの推論
- 予測の不確実性が事後分散として自然に得られる（LOO近似不要）

**FLARへの適用**: LOO残差ベースの不確実性をBayesian事後分散に置き換え可能。ただしGCVの計算効率（closed-form）を失う。

### 2.3 Adaptive Ridge (L0近似)

**[An Adaptive Ridge Procedure for L0 Regularization (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4743917/)**

- 反復的に重み付きRidgeを解き、L0ペナルティに収束
- 不要な特徴量を自動的にゼロにする（変数選択効果）
- FLARでは：harmonics数が多い場合に不要な周波数成分を自動削除

### 2.4 GCVの改善: 相関データへの補正 (2024)

**[Risk and cross validation in ridge regression with correlated samples (2024)](https://arxiv.org/html/2408.04607)**

- 時系列データはi.i.d.でないため、標準GCVには漸近的バイアスがある
- sample-sample共分散のスペクトル統計量を推定して補正
- **FLARへの直接的な改善点**: 現在のGCVは時系列の自己相関を無視している

---

## 3. 予測方式の改善

### 3.1 Stratify Framework (2025年、最新)

**[Stratify: Unifying Multi-Step Forecasting Strategies (2025)](https://link.springer.com/article/10.1007/s10618-025-01135-1)**

全ての多段階予測戦略を統一するフレームワーク:

| 戦略 | 概要 | モデル数 |
|------|------|---------|
| Recursive (現FLAR) | 1-step modelをh回適用 | 1 |
| Direct | h個の独立モデル | h |
| DirRec | Direct + 前ステップの予測も入力 | h |
| MIMO | 1モデルでh個を同時出力 | 1 |
| **Stratify** | base予測 + 残差予測を足し合わせ | 2 |

**Stratifyの核心アイデア**:
1. Base strategyで予測を生成
2. 残差時系列 `r_t = y_t - hat{y}_t` を計算
3. 残差をさらに予測して、base予測に加算

1080実験の84%以上で既存戦略を上回る。

**FLARへの適用**:
- Phase 1: Recursive (現状) → Direct に変更。各horizon stepで独立したRidge回帰を学習
- Phase 2: Stratify化 — Recursive予測の残差をDirect modelで補正

### 3.2 Direct Multi-Step (v3で試行済み → 全滅)

v3 ablationで試行し、改善しなかった。ただし:
- v3ではDirect + Exponential Weighting + Trend x Fourier interactionを同時変更
- **Direct単独**での評価は未実施の可能性
- Stratifyのように「Recursive + Direct残差補正」なら異なる結果が得られる可能性

### 3.3 MIMO (Multiple-Input Multiple-Output)

- 1つのRidgeモデルでhorizon全体を同時に出力
- `Y = [y_{t+1}, ..., y_{t+h}]` を列ベクトルとして、行列Ridge回帰
- horizon間の依存性を保存しつつ、recursive errorの蓄積を回避
- **性能比較**: MIMO < DirMO < Direct < DirREC < Iterated (RMSE順) という報告もあり、一概に優れているわけではない

---

## 4. アンサンブルの改善

### 4.1 Feature-Set Ensemble

**現状**: 1つの特徴量セットでRidgeを1回学習

**改善案**: 複数の特徴量セットで独立にRidge回帰を学習し、soft-average:

```
特徴量セットA: [intercept, trend, Fourier(period), lag-1, seasonal-lag]  ← 現FLAR
特徴量セットB: [intercept, trend, Fourier(period, 2*period), lag-1]      ← more harmonics
特徴量セットC: [intercept, piecewise-trend, Fourier(period), lag-1, lag-2, seasonal-lag]
```

各セットのGCVスコアでsoftmax重み → 加重平均。
これは既にsoft-average alphaで採用している原理を特徴量セットに拡張したもの。

### 4.2 FFORMA的メタ学習 (Feature-based Forecast Model Averaging)

**[Montero-Manso et al. (2020)](https://robjhyndman.com/publications/fforma/)**

- 時系列の特徴量（tsfeatures: spectral entropy, ACF, trend strength, seasonal strength等）を抽出
- メタモデル（XGBoost）が各予測手法の重みを学習
- M4で2位

**FLARへの適用**:
- FLAR自体をFORMAの1手法として組み込む
- 系列の特徴量に基づいて、FLARのハイパーパラメータ（harmonics数、context長、alpha範囲）を自動調整
- **ただし**: 単一手法の改善には直接寄与しない。パイプライン全体の改善向き。

### 4.3 AutoForecast (2025): 評価不要のモデル選択

**[Evaluation-Free Time-Series Forecasting Model Selection via Meta-Learning (2025)](https://dl.acm.org/doi/10.1145/3715149)**

- Landmarker meta-features: 軽量モデルの予測パターンを特徴量として使用
- 348データセットのリポジトリで検証
- 新しいデータセットに対して実際に予測を評価せずにモデルを選択

---

## 5. 非線形拡張

### 5.1 Echo State Networks + Ridge (Reservoir Computing)

**[Locally Connected Echo State Networks (2024)](https://openreview.net/forum?id=KeRwLLwZaw)**

- ランダムに初期化したrecurrent reservoir（固定重み）の状態を特徴量として、出力層をRidgeで学習
- FLARのFourier+lag特徴量をreservoir状態に置き換える発想
- **利点**: 非線形ダイナミクスを捕捉、Ridge学習のみで高速
- **欠点**: interpretabilityの喪失、reservoir設計のハイパーパラメータ

### 5.2 DeepRRTime: 正則化INR基底 + Ridge (2025)

**[DeepRRTime (TMLR 2025)](https://openreview.net/forum?id=uDRzORdPT7)**

DeepTimeの改良版:
1. 時間インデックスを入力とするImplicit Neural Representation (INR) で基底関数を学習
2. 各時系列に対してRidge回帰で基底の重みを適応
3. **正則化項**: 基底要素が単位標準化され、互いに無相関になるよう促進 → Ridge回帰のconditioning改善

**FLARへの示唆**: FLARのFourier基底は固定的。もし基底関数をデータから学習できれば、Fourier基底では捕捉できないパターンも表現可能。ただし、meta-learningが必要で計算コストが大幅に増加。

### 5.3 ahead::ridge2f (Quasi-Randomized Functional Link NN)

**[ahead package (Moudiki)](https://github.com/Techtonique/ahead)**

- ランダム特徴量変換 + Ridge回帰の組み合わせ
- 多変量時系列に対応、C++実装で高速
- Functional Link Neural Network: 入力を非線形変換（sin, cos, sigm）してRidgeで学習
- **FLARとの類似性**: Fourier基底 → random非線形基底に一般化

---

## 6. 時系列固有の改善

### 6.1 Time-Varying Parameters as Ridge Regressions (2024)

**[Goulet Coulombe (2024, Journal of Forecasting)](https://www.sciencedirect.com/science/article/abs/pii/S0169207024000931)**

最も理論的にエレガントな改善候補:

- 時変パラメータモデル `y_t = x_t^T * beta_t` をRidge回帰として定式化
- `beta_t` が時間とともにランダムウォークで変動すると仮定
- **dual解が閉形式**: カルマンフィルタのMCMC/フィルタリングを完全に回避
- cross-validationで「時変の程度」を自動チューニング
- Two-Step Ridge Regression (2SRR): 残差のvolatility変動にも適応

**FLARへの適用方法**:

```python
# 現在: 全期間で固定beta
beta = ridge_solve(X, y, alpha)

# 改善: 時変beta
# ステップ1: カーネル行列を構築
K[i,j] = k(t_i, t_j) = exp(-|t_i - t_j|^2 / (2*bandwidth^2))
# ステップ2: dual Ridge
alpha_dual = (K + lambda*I)^{-1} * y
# ステップ3: 予測
beta_t = X^T * diag(k(t, t_i)) * alpha_dual
```

**利点**:
- Fourier係数が時間とともに変化する「振幅変調」を自然に捕捉
- トレンドの変化点を明示的にモデル化する必要がない
- GCVでチューニング可能 → FLARの既存フレームワークに統合しやすい
- 計算効率: dual解でO(n^3)だが、FLARのcontext window (<500)なら問題ない

### 6.2 Adaptive Windowing (ADWIN)

**[ADWIN Algorithm](https://epubs.siam.org/doi/10.1137/1.9781611972771.42)**

- 概念ドリフト検出: データ分布の変化を自動検出
- ウィンドウサイズを動的に調整
- FLARの `max_ctx = max(period*10, 500)` を適応的に決定できる

**FLARへの適用**: 固定の500点コンテキストではなく、分布変化を検出して適切な学習ウィンドウを自動決定。

### 6.3 VAR Ridgeによる多変量化 (2025)

**[Ridge regularized estimation of VAR models for inference (Ballarin, 2025)](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.12737)**

- Vector ARモデルにRidgeを適用、漸近理論を整備
- 柔軟なペナルティスキームで複雑な縮退効果を実現
- cross-validationによるペナルティ選択手順を導出

**FLARへの適用**: GIFT-Evalの多変量データセット（electricity, solarなど）で、系列間の相関を活用したVAR-Ridge拡張。

---

## 7. 不確実性定量化の改善

### 7.1 Conformal Prediction (LOO残差の正式化)

**現状**: LOO残差をbootstrap的にリサンプリング + jitter

**改善案A: Jackknife+**

[Barber et al. (2021)](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf):
- LOO残差 `|y_i - hat{y}_{-i}(x_i)|` から予測区間を構築
- 有限サンプル保証: coverage >= 1 - 2*alpha
- FLARは既にLOO残差を計算しているので、直接適用可能

```python
# 現在
drawn = np.random.choice(recent, size=(n_samples, horizon), replace=True)
samples = point_fc + drawn + jitter

# 改善: Jackknife+ conformal
sorted_resid = np.sort(np.abs(loo_orig))
q_alpha = sorted_resid[int(np.ceil((1-alpha) * (len(sorted_resid)+1)))]
lower = point_fc - q_alpha
upper = point_fc + q_alpha
```

**改善案B: CopulaCPTS (多段階Conformal)**

[Conformal multistep-ahead multivariate time-series forecasting (2024)](https://link.springer.com/article/10.1007/s10994-024-06722-9):
- Copulaを使ってhorizon間の依存性を保存
- 多段階予測の各ステップで個別にcoverageを保証

**改善案C: Residual Distribution Predictive Systems (2025)**

[hal-05232300](https://hal.science/hal-05232300/document):
- exchangeability仮定のもとで分布全体を出力
- classic conformal predictive systemsと同等のcalibration保証

### 7.2 LOO残差のhorizon依存スケーリング

**現状**: 全horizonで同一のLOO残差分布を使用

**改善案**: recursive forecastではhorizonが遠いほど誤差が蓄積するため、horizon依存のスケーリングを適用:

```python
# 不確実性のhorizon拡大
scale = np.sqrt(1 + 0.1 * np.arange(horizon))  # 経験的スケーリング
samples = point_fc + drawn * scale + jitter * scale
```

理論的背景: AR(1)モデルの予測分散は `sigma^2 * (1 + phi^2 + phi^4 + ... + phi^{2(h-1)})` で増大。

---

## 8. Box-Cox変換の改善

### 8.1 Back-transformation Bias Correction

**現状**: `_bc_inv(fc_t, lam)` は中央値を返す（平均ではない）

[Hyndman & Athanasopoulos, FPP3](https://otexts.com/fpp3/transformations.html):
- Box-Cox逆変換の点予測は中央値であり、平均ではない
- バイアス補正: `E[Y] = bc_inv(mu) * (1 + sigma^2 * g(lam, mu))` の形

```python
# バイアス補正付き逆変換
def _bc_inv_bias_corrected(z, l, sigma2):
    if l == 0.0:
        return np.exp(z + sigma2/2)  # log-normal mean
    else:
        return _bc_inv(z, l) * (1 + sigma2 * (1-l) / (2 * (z*l+1)**2))
```

### 8.2 Guerrero法によるlambda選択

現在の `scipy.stats.boxcox` はMLE。Guerrero法は季節性を考慮したlambda選択で、時系列に特化:
- 季節ごとのCV（変動係数）を均一化するlambdaを選択
- 季節パターンが強い系列でより適切

---

## 9. 総合的な優先順位付き改善ロードマップ

### Tier 1: 高確度・低リスク（理論的に美しく、実装も容易）

| # | 改善 | 期待効果 | 実装コスト | 根拠 |
|---|------|---------|-----------|------|
| 1 | **Generalized Ridge (構造化ペナルティ)** | MASE改善 | 低 | 特徴量グループ別の最適縮退 |
| 2 | **Jackknife+ Conformal** | CRPS改善 | 低 | 既にLOO残差を計算済み |
| 3 | **Horizon依存の不確実性スケーリング** | CRPS改善 | 極低 | recursive error蓄積の理論的対応 |
| 4 | **Box-Cox逆変換のバイアス補正** | MASE微改善 | 極低 | 統計学的に正しい平均予測 |

### Tier 2: 中確度・中リスク（理論的に動機付けられるが、効果は系列依存）

| # | 改善 | 期待効果 | 実装コスト | 根拠 |
|---|------|---------|-----------|------|
| 5 | **Time-Varying Parameters (Goulet Coulombe)** | MASE改善 | 中 | 非定常系列への本質的対応 |
| 6 | **GCV相関データ補正** | alpha選択改善 | 中 | 時系列の自己相関を正式に考慮 |
| 7 | **Stratify (Recursive + Direct残差補正)** | MASE改善 | 中 | 2025年の最新、84%改善 |
| 8 | **Adaptive Context Window** | MASE改善 | 中 | 分布変化への適応 |
| 9 | **Feature-Set Ensemble** | MASE/CRPS改善 | 低 | soft-average原理の自然な拡張 |

### Tier 3: 高リスク・高リターン（大幅な設計変更が必要）

| # | 改善 | 期待効果 | 実装コスト | 根拠 |
|---|------|---------|-----------|------|
| 10 | **Reservoir Computing + Ridge** | 非線形捕捉 | 高 | ESNの最新研究で94%改善例 |
| 11 | **VAR-Ridge多変量化** | MASE改善 | 高 | 系列間相関の活用 |
| 12 | **FFORMA/AutoForecast統合** | 全指標改善 | 高 | per-seriesモデル選択 |

---

## 10. 最も推奨する改善: Generalized Ridge + Conformal + TV-Parameters

FLARの強みである「理論的な美しさ」を維持しつつ、最大の効果が期待できる3つの改善:

### (A) Generalized Ridge (Tier 1, #1)

```python
# 現在: 等方的Ridge
# beta^T * alpha * I * beta

# 改善: 構造化ペナルティ
# beta^T * Omega * beta
# where Omega = diag([alpha_intercept, alpha_trend, alpha_fourier, ..., alpha_lag])

# 実装: X → X * Omega^{-1/2} に変換してから標準Ridgeを解く
# GCVも同じフレームワークで動作する
```

特徴量グループごとのペナルティ比率は固定（ハイパーパラメータ探索不要）:
- intercept/trend: `alpha * 0.01` (ほぼ縮退させない)
- Fourier harmonics: `alpha * 1.0` (標準)
- 高調波: `alpha * k` (k=harmonic次数、高次ほど強く縮退)
- lag features: `alpha * 0.5` (やや弱い縮退)

### (B) Jackknife+ Conformal (Tier 1, #2)

既存のLOO残差計算を活かし、理論的に正しい予測区間を構築。
現在のbootstrap+jitterよりも統計的に厳密で、有限サンプルcoverage保証付き。

### (C) Time-Varying Parameters (Tier 2, #5)

Goulet Coulombeの2024年論文は、FLARのようなRidge回帰モデルに対して最も自然な時変パラメータ拡張。
カルマンフィルタやMCMCを使わず、dual Ridge回帰の閉形式解で実装可能。
GCVで「時変の程度」を自動チューニングでき、FLARの設計哲学と完全に整合する。

---

## Sources

### 特徴量設計・正則化
- [An optimized ridge regression for forecasting time series with a fixed period (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002971)
- [Generalized Ridge Regression: Biased Estimation for... (2024)](https://arxiv.org/pdf/2407.02583)
- [g.ridge: An R Package for Generalized Ridge Regression (2024)](https://www.mdpi.com/2073-8994/16/2/223)
- [Risk and cross validation in ridge regression with correlated samples (2024)](https://arxiv.org/html/2408.04607)
- [An Adaptive Ridge Procedure for L0 Regularization (2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4743917/)
- [Random Fourier Features for Kernel Ridge Regression (Avron et al., 2017)](https://arxiv.org/abs/1804.09893)
- [BayesianRidge - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)

### 予測方式
- [Stratify: Unifying Multi-Step Forecasting Strategies (2025)](https://link.springer.com/article/10.1007/s10618-025-01135-1)
- [Recursive and direct multi-step forecasting: the best of both worlds (Taieb & Hyndman)](https://www.semanticscholar.org/paper/Recursive-and-direct-multi-step-forecasting:-the-of-Taieb-Hyndman/432bd2365c8cfebd16577990404d3ff9d05d7e7d)

### 時変パラメータ
- [Time-Varying Parameters as Ridge Regressions (Goulet Coulombe, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0169207024000931)
- [Ridge regularized estimation of VAR models for inference (Ballarin, 2025)](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.12737)

### 非線形拡張
- [DeepRRTime: Robust Time-series Forecasting with a Regularized INR Basis (TMLR 2025)](https://openreview.net/forum?id=uDRzORdPT7)
- [Locally Connected Echo State Networks (2024)](https://openreview.net/forum?id=KeRwLLwZaw)
- [ahead package - Dynamic Ridge Regression (Moudiki)](https://github.com/Techtonique/ahead)

### メタ学習・モデル選択
- [FFORMA: Feature-based forecast model averaging (Montero-Manso et al., 2020)](https://robjhyndman.com/publications/fforma/)
- [Evaluation-Free Time-Series Forecasting Model Selection via Meta-Learning (2025)](https://dl.acm.org/doi/10.1145/3715149)

### 不確実性定量化
- [Distribution-Free Predictive Inference for Regression (Barber et al.)](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)
- [Conformal multistep-ahead multivariate time-series forecasting (2024)](https://link.springer.com/article/10.1007/s10994-024-06722-9)
- [Conformal Prediction for Ensembles (2024)](https://arxiv.org/abs/2405.16246)

### 季節性・変換
- [Time-varying seasonality with Fourier-spline interaction](https://ecogambler.netlify.app/blog/time-varying-seasonality/)
- [Forecasting: Principles and Practice, 3rd ed - Transformations](https://otexts.com/fpp3/transformations.html)
- [Dynamic harmonic regression (FPP3)](https://otexts.com/fpp3/dhr.html)

### 適応的手法
- [Adaptive Regime-Switching Forecasts with Conformal Prediction (2024)](https://arxiv.org/html/2512.03298)
- [ADWIN: Learning from Time-Changing Data with Adaptive Windowing](https://epubs.siam.org/doi/10.1137/1.9781611972771.42)

### Competition研究
- [M4 Competition Results (Makridakis et al.)](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
- [M5 accuracy competition: Results, findings, and conclusions](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
