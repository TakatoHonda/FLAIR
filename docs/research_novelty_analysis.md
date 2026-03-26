# 新規性分析: 階層クラスタリング + 予測 + MinT reconciliation

調査日: 2026-03-17

## 概要

「階層クラスタリング → 予測 → MinT reconciliation」パイプラインの新規性を、先行研究との比較から分析した。

---

## 直撃する先行研究

### Zhang, Panagiotelis & Li (2024) — 最重要

- **論文**: "Constructing hierarchical time series through clustering: Is there an optimal way for forecasting?"
- **arXiv**: 2404.06064 / International Journal of Forecasting
- **内容**: 階層クラスタリング（Ward法）とk-medoidsでデータ駆動的に階層構造を構築 → MinT等でreconciliation
- **距離尺度**: Euclidean, DTW
- **系列表現**: 生系列, 予測誤差, 特徴量(56個, tsfeatures), 予測誤差特徴量
- **12組み合わせ** (2クラスタリング × 2距離 × 3表現) を体系的に比較
- **衝撃的な発見**:
  > クラスタリングの改善は「似た系列をグループにまとめた」ことではなく、**階層の構造そのもの**（深さ、中間レベル数）から来ている。ランダムに系列を割り当てても統計的に同等の結果。
- **推奨**: 複数クラスタリング階層の等重み平均が最良

### Cini et al. (ICML 2024) — HiGP

- **論文**: "Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting"
- **arXiv**: 2305.19183
- **内容**: GNNのtrainable graph poolingで階層構造をend-to-endで学習
- **特徴**: クラスタ割り当てが予測目的関数と連動（joint optimization）
- **差別化**: ブラックボックス（解釈不可能）

### その他の関連研究

| 論文 | 年 | 手法 | ドメイン |
|---|---|---|---|
| Li, Li, Lu & Panagiotelis | 2019 | 階層クラスタリング + reconciliation | 死因別死亡率 |
| Pang et al. (IJCAI) | 2018 | X-meansクラスタリング + reconciliation | 電力 |
| Pang et al. | 2022 | 複数クラスタリング + sparse penalty | 電力・太陽光 |
| Mattera et al. | 2024 | PAMクラスタリング + reconciliation | 株式市場 |
| Yang et al. | 2017 | 空間k-means + reconciliation | 太陽光発電 |
| Han et al. | 2022 | 多レベルクラスタリング（効率化目的） | 大規模階層 |

---

## MinT reconciliation の発展と限界

### 主要バリアント

| バリアント | 共分散仮定 | 用途 |
|---|---|---|
| OLS | 単位行列 | 最もシンプル |
| WLS (variance scaling) | 対角、in-sample分散 | 系列ごとの誤差分散が異なる場合 |
| WLS (structural scaling) | 対角、bottom-level系列数に比例 | 構造が分散を決める場合 |
| MinT(Sample) | 完全不偏標本共分散 | データが十分ある場合 |
| MinT(Shrink) | 縮小推定量 | 高次元（系列数 > 時系列長） |

### 既知の限界

1. **高次元での共分散推定**: 系列数 >> 時系列長で破綻
2. **不偏性仮定**: 基本予測が不偏であること（実際にはほぼ成立しない）
3. **計算コスト**: O(n³) の行列逆行列（大規模階層で非実用的）
4. **線形性**: 基本予測の線形結合のみ
5. **静的重み**: 固定的なreconciliation重み

### 主要な拡張手法

| 手法 | 年 | arXiv | 特徴 |
|---|---|---|---|
| ERM | 2019 | — | 不偏性仮定を緩和、バイアス-分散トレードオフ最適化 |
| EMinT | 2021 | — | 不偏制約なしでtrace最小化 |
| GTOP | 2015 | — | ゲーム理論的、改善保証 |
| FLAP | 2024 | — | 線形射影による分散削減、最適証明あり |
| MinTit | 2024 | 2409.18550 | 反復的trace最小化、短時系列に強い |
| FlowRec | 2025 | 2505.03955 | ネットワークフロー最適化、3-40x高速化 |
| NeuralReconciler | 2024 (WSDM) | — | Attention + Normalizing Flow、20%改善 |
| Robust (M-estimation) | 2026 | 2602.22694 | 外れ値頑健 |
| Billions-Scale | 2026 | 2602.05030 | 40億+予測値のreconciliation |

---

## 新規性ギャップ分析

### 「階層クラスタリング → 予測 → MinT」そのものは既出

Zhang et al. (2024) が直撃。コアアイデアの新規性は主張困難。

### 残存するギャップ

| # | ギャップ | 説明 | 新規性 |
|---|---|---|---|
| 1 | **Foundation Modelとの比較** | FM + reconciliation vs 統計手法 + reconciliationの体系的比較がゼロ | ★★★★★ |
| 2 | **周期性ベースの階層構築** | 周期性特徴でクラスタリングした研究はゼロ | ★★★★★ |
| 3 | **クラスタ構造 ↔ 共分散推定の理論** | Sの構造がMinTの共分散推定品質にどう影響するか未解明 | ★★★★☆ |
| 4 | **デンドログラムの最適pruning** | どの深さで切るのが最適かの理論的分析なし | ★★★☆☆ |
| 5 | **cross-temporal拡張** | 時間方向の集約とクラスタリング階層の組み合わせは未研究 | ★★★☆☆ |
| 6 | **動的再クラスタリング** | 時間とともにクラスタ構造を更新する手法は不存在 | ★★★☆☆ |
| 7 | **linkage手法の比較** | Ward法のみ検証済（complete, average, singleは未検証） | ★★☆☆☆ |
| 8 | **加重アンサンブル** | 複数階層の等重み平均のみ（加重は未試行） | ★★☆☆☆ |

---

## Foundation Model と階層予測の現状

### FMは階層的整合性を一切考慮しない

| モデル | 階層reconciliation対応 |
|---|---|
| Chronos / Chronos-2 | なし（group attentionはあるが整合性制約なし） |
| TimesFM | なし |
| Moirai / Moirai 2.0 | なし |
| TimeGPT (Nixtla) | 外部ライブラリで事後的にreconciliation可能（唯一） |
| Toto (Datadog) | なし |

### FM + reconciliation vs 統計手法 + reconciliation の比較論文: **ゼロ**

これは完全に空白の研究領域。

---

## 推奨する差別化の方向性

### A. 「統計手法 + clustering + reconciliation vs Foundation Models」

最も有望。以下のストーリーが構築可能:
> 「Foundation Modelはゼロショットで個別系列を予測するが、階層構造を活用しない。クラスタリング + MinTによる統計的手法が、構造を活用することでFMを上回れるか？」

### B. 周期性検出を用いた階層構築 → ギャップ#2

周期性自動検出と階層予測の交差領域はほぼ空白。詳細は `research_periodicity_hierarchy.md` を参照。

### C. 「なぜ構造が効くのか」の理論的解明 → ギャップ#3

Zhang et al.は「構造が重要」と示したが、**なぜそうなのか**は説明していない。MinTの共分散行列の条件数やデンドログラムの形状特性との関係を理論的に分析できれば強い。

---

## 参考文献

### 階層予測レビュー
- Athanasopoulos, Hyndman, Kourentzes, Panagiotelis (2024). "Forecast reconciliation: A review." IJF 40(2), 430-456.
- Awesome Forecast Reconciliation: https://github.com/danigiro/awesome-forecast-reconciliation

### ソフトウェア
| パッケージ | 言語 | 範囲 |
|---|---|---|
| FoReco | R | cross-sectional, temporal, cross-temporal reconciliation |
| FoRecoPy | Python | FoRecoのPython版 |
| HierarchicalForecast | Python (Nixtla) | Bottom-up, top-down, MinTrace, ERM |
| hts → fabletools | R | Hyndman et al.の最新エコシステム |
| HiGP | Python (PyTorch) | Graph-basedのend-to-end learned hierarchies |
