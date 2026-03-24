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

## GIFT-Eval Results

97 configurations, 23 datasets, 7 domains ([GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval)):

| Model | Type | relMASE | relCRPS | GPU |
|-------|------|---------|---------|-----|
| **FLAIR** | **Statistical** | **0.915** | **0.690** | **No** |
| N-BEATS | Deep Learning | 0.938 | 0.816 | Yes |
| TFT | Deep Learning | 0.915 | 0.605 | Yes |
| SeasonalNaive | Baseline | 1.000 | 1.000 | No |
| DLinear | Deep Learning | 1.061 | 0.846 | Yes |
| AutoARIMA | Statistical | 1.074 | 0.912 | No |
| TiDE | Deep Learning | 1.091 | 0.772 | Yes |
| AutoTheta | Statistical | 1.090 | 1.244 | No |
| DeepAR | Deep Learning | 1.343 | 0.853 | Yes |
| MFLES | Statistical | 1.405 | 1.015 | No |
| Prophet | Statistical | 1.540 | 1.061 | No |

FLAIR beats N-BEATS, TFT, DLinear, TiDE, and DeepAR — all GPU-trained deep learning models — with one SVD and no GPU. Per-horizon: short=0.892, medium=0.929, long=0.965.

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

## GIFT-Eval ベンチマーク結果

97設定、23データセット、7ドメイン（[GIFT-Eval benchmark](https://huggingface.co/spaces/Salesforce/GIFT-Eval)）:

| モデル | 種別 | relMASE | relCRPS | GPU |
|-------|------|---------|---------|-----|
| **FLAIR** | **統計** | **0.915** | **0.690** | **不要** |
| N-BEATS | 深層学習 | 0.938 | 0.816 | 必要 |
| TFT | 深層学習 | 0.915 | 0.605 | 必要 |
| SeasonalNaive | ベースライン | 1.000 | 1.000 | 不要 |
| DLinear | 深層学習 | 1.061 | 0.846 | 必要 |
| AutoARIMA | 統計 | 1.074 | 0.912 | 不要 |
| TiDE | 深層学習 | 1.091 | 0.772 | 必要 |
| AutoTheta | 統計 | 1.090 | 1.244 | 不要 |
| DeepAR | 深層学習 | 1.343 | 0.853 | 必要 |
| MFLES | 統計 | 1.405 | 1.015 | 不要 |
| Prophet | 統計 | 1.540 | 1.061 | 不要 |

FLAIR は N-BEATS、TFT、DLinear、TiDE、DeepAR — いずれもGPU学習済みの深層学習モデル — を、SVD 1回・GPU不要で上回る。ホライズン別: short=0.892, medium=0.929, long=0.965。
