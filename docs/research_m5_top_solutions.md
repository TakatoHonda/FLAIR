# M5 Kaggle コンペティション 上位解法調査

調査日: 2026-03-17

## コンペティション概要

- **主催**: Spyros Makridakis, Kaggle (2020年)
- **データ**: Walmart販売データ, 30,490個別商品時系列, 10店舗, 3州 (CA, TX, WI), 約5年間の日次データ
- **予測期間**: 28日先
- **2トラック**:
  - **Accuracy**: WRMSSE (Weighted Root Mean Scaled Squared Error) — 12階層レベルの加重指標
  - **Uncertainty**: WSPL (Weighted Scaled Pinball Loss) — 9分位点 × 12階層レベル
- **COVIDの影響**: データは2016年6月まで（COVID以前）。コンペ自体は2020年3-6月開催

### 12レベル階層構造

| レベル | 内容 | 系列数 |
|---|---|---|
| 1 | Total | 1 |
| 2 | State | 3 |
| 3 | Store | 10 |
| 4 | Category | 3 |
| 5 | Department | 7 |
| 6 | State × Category | 9 |
| 7 | State × Department | 21 |
| 8 | Store × Category | 30 |
| 9 | Store × Department | 70 |
| 10 | Item | 3,049 |
| 11 | State × Item | 9,147 |
| 12 | Item × Store (bottom) | 30,490 |

---

## Accuracy トラック 上位解法

### 1st Place: YeonJun In — "DRFAM" (スコア: 0.52060)

**論文**: "Simple averaging of direct and recursive forecasts via partial pooling using machine learning" (IJF, 2022, Vol 38(4), pp 1386-1399)

**コード**: https://github.com/Mcompetitions/M5-methods (A1)
https://github.com/YeonJun-IN/-kaggle-M5---Accuracy-1st-place-solution

**モデル**: 純粋LightGBM, NNなし, 外部データなし, Kaggle CPU上で実行

**アーキテクチャ — 6モデルアンサンブル (2戦略 × 3プーリングレベル)**:

| 戦略 | 説明 | モデル数 |
|---|---|---|
| Recursive | 1日ずつ予測し、予測値を次の入力に使用 | 110 (10+30+70) |
| Non-recursive (Direct) | 28日分を直接予測 | 110 (10+30+70) |

3つのプーリングレベル:
1. Store別 (10モデル)
2. Store × Category別 (30モデル)
3. Store × Department別 (70モデル)

**最終アンサンブル**: 6提出ファイルの単純算術平均

**LightGBMハイパーパラメータ (recursiveモデル)**:

```python
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'tweedie',
    'tweedie_variance_power': 1.1,
    'metric': 'rmse',
    'subsample': 0.5,
    'subsample_freq': 1,
    'learning_rate': 0.015,
    'num_leaves': 2047,       # 2^11 - 1
    'min_data_in_leaf': 4095, # 2^12 - 1
    'feature_fraction': 0.5,
    'max_bin': 100,
    'n_estimators': 3000,
    'boost_from_average': False,
}
```

Non-recursiveモデル: `num_leaves: 255`, `min_data_in_leaf: 255`

**特徴量エンジニアリング**:

| カテゴリ | 詳細 |
|---|---|
| **Sales lag** | lag_28〜lag_42 (15個, shift=28でリーク防止) |
| **Rolling stats** | mean/std: 窓サイズ 7, 14, 30, 60, 180日 |
| **Shifted rolling** | shift [1, 7, 14] → roll [7, 14, 30, 60] |
| **Price** | sell_price, max, min, std, mean, norm (price/max), nunique, momentum (週/月/年) |
| **Calendar** | event_name/type (1,2), SNAP flags, day of month, week of year, month, year, week of month, day of week, weekend |
| **Target encoding** | cat_id, dept_id, item_id, store_id等のmean/std |
| **その他** | release (商品初登場週), moon phase |

**主要トリック**:
- Tweedie損失 (power=1.1): ゼロ膨張カウントデータに必須
- Recursive + Directの組み合わせがデータの異なる側面を捕捉
- Early stoppingなし — 3000イテレーション固定
- **階層reconciliationは未使用**

---

### 2nd Place: Matthias Anderer (スコア: 0.52816)

**論文**: "Hierarchical forecasting with a top-down alignment of independent level forecasts" (IJF, 2022, arXiv: 2103.08250)

**コード**: https://github.com/matthiasanderer/m5-accuracy-competition

**アーキテクチャ — ハイブリッド階層手法**:
- **Bottom-level (Level 12)**: LightGBM (store別10モデル × 5 loss multiplier = 50モデル)
- **Top-level (Levels 1-5)**: N-BEATS (GluonTS, EnsembleEstimator, sMAPE損失)
- **アライメント**: Bottom-level予測をN-BEATSのtop-level予測にスケーリング調整

**最大の独自性**: **Top-downアライメント**
- N-BEATSで上位レベルの連続的な系列を予測
- LightGBMで下位レベルの間欠的な系列を予測
- loss multiplier (λ = 0.9, 0.93, 0.95, 0.97, 0.99) で調整 → 5モデルの等重み平均

**重要な洞察**: 歴史的売上特徴量（lag, rolling mean）をLightGBMで**意図的に不使用**
> 間欠的売上データでは短期lagはほぼノイズ。季節パターン（「木曜にペパーミントキャンディが売れる」等）はdatetime特徴量で捕捉できる。

**N-BEATS構成**:
- meta_context_length = [3, 5, 7], bagging_size = 3 → 9アンサンブルモデル
- 約4.5時間 (Colab GPU)

**所要時間**: 約5営業日

---

### 3rd Place: Jeon & Seong (devmofl) (スコア: ~0.532)

**論文**: "Robust recurrent network model for intermittent time-series forecasting" (IJF, 2022)

**コード**: https://github.com/devmofl/M5_Accuracy_3rd

**アーキテクチャ**: 修正DeepAR (LSTM) × 43NNアンサンブル

**最大の独自性**: **学習済み分布からのサンプリング**
- 訓練中、実際の過去値ではなく学習済み分布からサンプリングした値を入力
- ゼロ膨張の間欠需要パターンに特化
- Tweedie回帰 + cosine annealing

---

### 4th Place: Monsaraida (スコア: 0.53583)

**コード**: https://github.com/monsaraida/kaggle-m5-forecasting-accuracy-4th-place

**アーキテクチャ**: LightGBM × 40 (10 stores × 4予測horizon: 7, 14, 21, 28日)

- Non-recursiveのみ、アンサンブルなし、乗数補正なし
- 特徴量は1st placeとほぼ同一（kyakovlevベース）
- **シンプルさが戦略**
- 16時間訓練 + 10分予測

---

### 5th Place: Alan Lahoud (スコア: 0.53604)

**アーキテクチャ**: LightGBM (Poisson) × 7 (department別)

**最大の独自性**: **事後補正乗数**
- 生の予測が実際より3-4%低い傾向 → department-store別に補正係数を計算
- 補正係数: 0.92〜1.10の範囲
- これが最大の差別化要因

---

## Uncertainty トラック 上位解法

### 1st Place: "Everyday Low SPLices" (スコア: 0.15420)

**論文**: "Forecasting with gradient boosted trees: augmentation, tuning, and cross-validation strategies" (IJF, 2022)

**アーキテクチャ**: LightGBM quantile回帰, 各階層レベル別にモデル構築 (126モデル = 9分位点 × 14グループ)

**主要テクニック — Range-Blended Gradient Boosting**:
- Levels 1-9 (少数系列) でCV 5%改善、個別レベルで5-15%改善

**特徴量**:
- EWM (3, 7, 15, 30, 100日)
- 非ゼロ日数割合 (7, 14, 28, 56, 112日)
- Rolling stats (mean, median, stdev, skew, kurtosis)
- **外部データなし、価格データなし、item_idも不使用**

**CV**: LeaveOneGroupOut (年ベース), 休日月を除外

**重要な洞察**:
> コンペの差は Levels 1-9 (スパース, 低n系列) で決まった。Levels 10-12 はどの手法でも同じCV。10特徴量のスパースモデルで30分で17位相当のスコアが出る。

---

### 2nd Place: GoodsForecast (スコア: 0.15890)

**5ステップパイプライン**:
1. Base LightGBM点予測 (recursive)
2. SSA (特異スペクトル分析) による総量補正 (0.96係数)
3. 加重ヒストグラムによる確率分布構築 (forgetting factor + 年周期 + 週周期)
4. ヒストグラムをLGBM点予測の中央値に合わせてシフト
5. 分位点別の丸め戦略 (Floor/Round/Ceil)

---

### 3rd Place: Ouranos (スコア: 0.15892)

**アーキテクチャ**: LightGBM + Keras NN (加重幾何平均: `(LGBM³ × Keras)^(1/4)`)

**不確実性推定**:
- Levels 1-9: 正規分布で係数推定
- Levels 10-12: 歪正規分布で係数推定
- 上位分位点 (99.5%) に1.02-1.03の追加乗数（右裾の偏りに対応）

---

## 横断的知見

### 成功要因

1. **LightGBM + Tweedie (power=1.1)** が事実上の標準
2. **Cross-learning** (複数系列をプールして1モデルで学習) が統計手法にない強み
3. **特徴量はほぼ共通** (kyakovlevベース): lag28-42, rolling mean/std, 価格momentum, カレンダー
4. **Store別モデル** がほぼ全チームで採用
5. **外部データ不使用** — 全上位チームが提供データのみ
6. **アンサンブルの多様性** > 個別モデルの精度

### 階層の扱い

> **正式なreconciliation（MinT等）を使った上位チームは存在しない**

| 順位 | 階層の扱い |
|---|---|
| 1st Accuracy | 3レベルのプーリング → 単純平均（暗黙的） |
| 2nd Accuracy | **N-BEATS(上位) + LightGBM(下位) → top-downアライメント**（最もreconciliationに近い） |
| 5th Accuracy | 事後的に乗数で補正 |
| 1st Uncertainty | 各階層レベル別に独立モデル |

### 統計手法 vs ML

- **純粋な統計手法は上位に入れなかった**
- 統計手法が競争力を持つのは高度に集約されたレベル (1-5) のみ
- 下位レベル (10-12) ではcross-learningできるMLが圧倒的

---

## Conmcast手法への示唆

### 良いニュース
- M5でreconciliationが使われていないのは「試されていない」面が大きい
- 2nd placeが唯一の階層的手法で効果を示した → reconciliationの余地あり
- Uncertainty trackで階層レベル別に異なる分布仮定が有効 → 周期性ベースのクラスタリングと親和性

### 注意点
- M5での勝因は**LightGBMのcross-learning + Tweedie** → 純粋な統計手法単体では厳しい
- M5をベンチマークに使うなら、「LightGBM + クラスタリング + MinT」のハイブリッドが現実的
- ゼロ膨張の間欠需要データに対する周期性検出の精度が課題

### 参考リポジトリ
- 公式: https://github.com/Mcompetitions/M5-methods
- 1st place: https://github.com/YeonJun-IN/-kaggle-M5---Accuracy-1st-place-solution
- 2nd place: https://github.com/matthiasanderer/m5-accuracy-competition
- 3rd place: https://github.com/devmofl/M5_Accuracy_3rd
- 4th place: https://github.com/monsaraida/kaggle-m5-forecasting-accuracy-4th-place
- kyakovlev notebooks: Kaggle公開カーネル (m5-simple-fe, m5-lags-features, m5-custom-features)
