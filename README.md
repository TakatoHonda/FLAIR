# FLAIR

**Factored Level And Interleaved Ridge** — a single-equation time series forecasting method.

[日本語版はこちら](#flair-日本語)

## The Idea

Reshape a time series by its primary period, then separate *what* (level) from *how* (shape):

```
y(phase, period) = Level(period) × Shape(phase)
```

- **Level**: period totals, forecast by Ridge regression with soft-average GCV
- **Shape**: within-period proportions via Dirichlet-Multinomial empirical Bayes, with context derived from the secondary period structure
- **Location shift**: automatic handling of negative-valued series via shift-before-Box-Cox
- **P=1 degeneration**: no separate fallback model — one unified code path for all series

One SVD. Zero hyperparameters. No neural network.

## Usage

```python
import numpy as np
from flair import flair_forecast

y = np.random.rand(500) * 100  # your time series (can include negatives)
samples = flair_forecast(y, horizon=24, freq='H')
point_forecast = samples.mean(axis=0)
```

## Dependencies

- numpy
- scipy

## How It Works

1. **Location shift**: shift all values to be positive (handles negative-valued series like temperature)
2. **Reshape** the series by the primary calendar period (e.g., 24 for hourly). If fewer than 3 complete periods, degenerate to P=1 (Ridge on raw series)
3. **Shape** = Dirichlet posterior mean per context (`context = period_index % C`, where `C = secondary/primary` period). Shrinks toward the global average when context-specific data is scarce
4. **Level** = period totals → Box-Cox → NLinear → Ridge with soft-average GCV
5. **Forecast** Level for `ceil(H/P)` future periods via recursive prediction
6. **Reconstruct** `y_hat = Level_hat × Shape`, undo shift, and generate conformal prediction intervals

## Benchmark Results

### Chronos Zero-Shot Benchmark (25 datasets)

Evaluated on the exact same protocol as [Chronos](https://github.com/amazon-science/chronos-forecasting) (Ansari et al., 2024). Agg. Relative Score = geometric mean of per-dataset (method / Seasonal Naive). Lower is better.

Baseline results from [autogluon/fev](https://github.com/autogluon/fev/tree/main/benchmarks/chronos_zeroshot/results) and [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/evaluation/results).

| Model | Params | Agg. Rel. MASE | Agg. Rel. WQL | GPU |
|-------|--------|:--------------:|:-------------:|:---:|
| **FLAIR** | **~6** | **0.704** | 0.850 | **No** |
| Chronos-Bolt-Base | 205M | 0.791 | **0.624** | Yes |
| Moirai-Base | 311M | 0.812 | 0.637 | Yes |
| Chronos-T5-Base | 200M | 0.816 | 0.642 | Yes |
| Chronos-Bolt-Small | 48M | 0.819 | 0.636 | Yes |
| Chronos-T5-Large | 710M | 0.821 | 0.650 | Yes |
| Chronos-T5-Small | 46M | 0.830 | 0.665 | Yes |
| Chronos-T5-Mini | 20M | 0.841 | 0.689 | Yes |
| Chronos-Bolt-Tiny | 9M | 0.845 | 0.668 | Yes |
| AutoARIMA | - | 0.865 | 0.742 | No |
| Chronos-T5-Tiny | 8M | 0.870 | 0.711 | Yes |
| TimesFM | 200M | 0.879 | 0.711 | Yes |
| AutoETS | - | 0.937 | 0.812 | No |
| Seasonal Naive | - | 1.000 | 1.000 | No |

**FLAIR ranks #1 on point forecast accuracy (MASE) across all models** — including every Chronos variant (up to 710M params), Moirai, TimesFM, AutoARIMA, and AutoETS. No GPU. No pretraining. ~6 parameters per series.

On probabilistic forecasting (WQL), Chronos models retain an advantage due to their learned quantile distributions vs. FLAIR's conformal intervals.

### GIFT-Eval Benchmark (97 configs, 23 datasets)

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — 7 domains, short/medium/long horizons:

| Model | Type | relMASE | relCRPS | GPU |
|-------|------|---------|---------|-----|
| **FLAIR** | **Statistical** | **0.885** | **0.663** | **No** |
| Chronos-Small | Foundation | 0.892 | — | Yes |
| N-BEATS | Deep Learning | 0.938 | 0.816 | Yes |
| TFT | Deep Learning | 0.915 | 0.605 | Yes |
| SeasonalNaive | Baseline | 1.000 | 1.000 | No |
| AutoARIMA | Statistical | 1.074 | 0.912 | No |
| DeepAR | Deep Learning | 1.343 | 0.853 | Yes |
| Prophet | Statistical | 1.540 | 1.061 | No |

Per-horizon: short=0.885, medium=0.865, long=0.906.

## Citation

```
@misc{flair2026,
  title={FLAIR: Factored Level And Interleaved Ridge for Time Series Forecasting},
  year={2026}
}
```

## License

MIT

---

# FLAIR 日本語

**Factored Level And Interleaved Ridge** — 1つの式で完結する時系列予測手法。

## アイデア

時系列を主周期でリシェイプし、「何が」（水準）と「どのように」（形状）を分離する:

```
y(phase, period) = Level(period) × Shape(phase)
```

- **Level**: 周期合計。ソフト平均GCV付きRidge回帰で予測
- **Shape**: 周期内比率。Dirichlet-Multinomial経験ベイズにより、副周期構造から導出されたコンテキストごとに推定
- **Location shift**: 負の値を含む系列を自動処理（Box-Cox前にシフト）
- **P=1退化**: フォールバックモデル不要 — 全系列を1つのコードパスで処理

SVD 1回。ハイパーパラメータ 0個。ニューラルネットワーク不要。

## 使い方

```python
import numpy as np
from flair import flair_forecast

y = np.random.rand(500) * 100  # 時系列データ（負の値も可）
samples = flair_forecast(y, horizon=24, freq='H')
point_forecast = samples.mean(axis=0)
```

## 依存パッケージ

- numpy
- scipy

## 仕組み

1. **Location shift**: 全値を正にシフト（温度のような負値系列に対応）
2. **リシェイプ**: 主周期でリシェイプ（例: 時間データ → P=24）。完全周期が3未満の場合、P=1に退化（生系列でRidge）
3. **Shape**: コンテキスト別の Dirichlet 事後平均（`context = period_index % C`、`C = 副周期/主周期`）。データが少ないコンテキストでは全体平均に縮約
4. **Level**: 周期合計 → Box-Cox → NLinear → ソフト平均GCV Ridge
5. **予測**: Level を `ceil(H/P)` ステップ再帰予測
6. **復元**: `y_hat = Level_hat × Shape`、シフトを戻し、LOO conformal 予測区間を生成

## ベンチマーク結果

### Chronos Zero-Shot ベンチマーク（25データセット）

[Chronos](https://github.com/amazon-science/chronos-forecasting)（Ansari et al., 2024）と完全に同一のプロトコルで評価。Agg. Relative Score = 各データセットの（手法 / Seasonal Naive）の幾何平均。低いほど良い。

ベースライン結果は [autogluon/fev](https://github.com/autogluon/fev/tree/main/benchmarks/chronos_zeroshot/results) および [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/evaluation/results) から取得。

| モデル | パラメータ数 | Agg. Rel. MASE | Agg. Rel. WQL | GPU |
|-------|:----------:|:--------------:|:-------------:|:---:|
| **FLAIR** | **~6** | **0.704** | 0.850 | **不要** |
| Chronos-Bolt-Base | 205M | 0.791 | **0.624** | 必要 |
| Moirai-Base | 311M | 0.812 | 0.637 | 必要 |
| Chronos-T5-Base | 200M | 0.816 | 0.642 | 必要 |
| Chronos-Bolt-Small | 48M | 0.819 | 0.636 | 必要 |
| Chronos-T5-Large | 710M | 0.821 | 0.650 | 必要 |
| Chronos-T5-Small | 46M | 0.830 | 0.665 | 必要 |
| Chronos-T5-Mini | 20M | 0.841 | 0.689 | 必要 |
| Chronos-Bolt-Tiny | 9M | 0.845 | 0.668 | 必要 |
| AutoARIMA | - | 0.865 | 0.742 | 不要 |
| Chronos-T5-Tiny | 8M | 0.870 | 0.711 | 必要 |
| TimesFM | 200M | 0.879 | 0.711 | 必要 |
| AutoETS | - | 0.937 | 0.812 | 不要 |
| Seasonal Naive | - | 1.000 | 1.000 | 不要 |

**FLAIRは点予測精度（MASE）で全モデル中1位** — Chronos全バリアント（最大710Mパラメータ）、Moirai、TimesFM、AutoARIMA、AutoETSを上回る。GPU不要。事前学習不要。系列あたり約6パラメータ。

確率予測（WQL）では、Chronosの学習済み分位点分布がFLAIRの conformal intervals より優位。

### GIFT-Eval ベンチマーク（97設定、23データセット）

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) — 7ドメイン、short/medium/longホライズン:

| モデル | 種別 | relMASE | relCRPS | GPU |
|-------|------|---------|---------|-----|
| **FLAIR** | **統計** | **0.885** | **0.663** | **不要** |
| Chronos-Small | 基盤モデル | 0.892 | — | 必要 |
| N-BEATS | 深層学習 | 0.938 | 0.816 | 必要 |
| TFT | 深層学習 | 0.915 | 0.605 | 必要 |
| SeasonalNaive | ベースライン | 1.000 | 1.000 | 不要 |
| AutoARIMA | 統計 | 1.074 | 0.912 | 不要 |
| DeepAR | 深層学習 | 1.343 | 0.853 | 必要 |
| Prophet | 統計 | 1.540 | 1.061 | 不要 |

ホライズン別: short=0.885, medium=0.865, long=0.906。
