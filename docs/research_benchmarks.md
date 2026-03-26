# 時系列予測ベンチマーク調査

調査日: 2026-03-17

## 概要

時系列予測Foundation Modelの評価ベンチマークを調査し、統計的手法（階層クラスタリング + 予測 + MinT reconciliation）との比較に適したベンチマークを特定した。

---

## 主要ベンチマーク比較

| ベンチマーク | 年 | 採択先 | データセット数 | FM特化 | Zero-Shot | 非リーク事前学習 | 確率的評価 | 共変量 | リーダーボード |
|---|---|---|---|---|---|---|---|---|---|
| **GIFT-Eval** | 2024 | NeurIPS 2024 | 23-28 | Yes | Yes | Yes (230B pts) | Yes (CRPS) | Limited | Yes |
| **TIME** | 2026 | プレプリント | 50 (新規) | Yes | Yes (厳密) | N/A (新規データ) | Yes | Yes | Yes |
| **fev-bench** | 2025 | プレプリント | 96 | Yes | Yes | No | Yes | Yes (46タスク) | Yes |
| **TSFM-Bench** | 2024 | KDD 2025 | 21 | Yes | Yes | No | Yes | Limited | No |
| **TFB** | 2024 | PVLDB 2024 | 25 MV + 8K UV | No | No | No | Partial | No | No |
| **ProbTS** | 2023 | NeurIPS 2024 | 中規模 | Yes | Yes | No | Yes (core) | No | No |
| **Monash** | 2021 | NeurIPS 2021 | 30 (58 var.) | No | No | No | No | No | No |

---

## 各ベンチマーク詳細

### 1. GIFT-Eval (General TIme Series ForecasTing Model Evaluation)

- **発表**: 2024年10月 (arXiv: 2410.10393), NeurIPS 2024 Datasets & Benchmarks Track
- **著者**: Salesforce AI Research
- **規模**: 23データセット (28構成), 144,000+時系列, 1.77億データポイント
- **ドメイン**: 経済, 金融, 医療, 自然, 販売, 交通, Web/クラウド運用 (7ドメイン)
- **頻度**: 分単位〜年単位 (10種)
- **評価指標**: Median MAPE (点予測), CRPS (確率的予測), Seasonal Naiveで正規化
- **ベースライン**: 17モデル (統計, DL, Foundation Model)
- **最大の差別化**: 非リーク事前学習コーパス (~230B点, 88データセット) を提供
- **リソース**:
  - 論文: https://arxiv.org/abs/2410.10393
  - コード: https://github.com/SalesforceAIResearch/gift-eval
  - リーダーボード: https://huggingface.co/spaces/Salesforce/GIFT-Eval
- **強み**: FMの公平な評価に最適、多ドメイン・多頻度
- **弱み**: 予測タスクのみ、高頻度データでFMが苦戦、データセット数は中程度

### 2. TIME (Towards the Next Generation)

- **発表**: 2026年2月 (arXiv: 2602.12147)
- **規模**: 50の完全新規データセット, 98予測タスク
- **特徴**: Human-in-the-loop構築、パターンレベル評価、real-world aligned
- **強み**: データリーク完全排除、最も厳密
- **弱み**: 非常に新しい、zero-shot評価のみ

### 3. fev-bench

- **発表**: 2025年9月 (arXiv: 2509.26468)
- **著者**: AutoGluon team (Amazon)
- **規模**: 100予測タスク, 96データセット, 7ドメイン
- **特徴**: 46タスクが共変量付き、ブートストラップ信頼区間による統計的厳密性
- **評価**: win rateとskill scoreの2軸
- **リソース**:
  - コード: https://github.com/autogluon/fev
  - リーダーボード: https://huggingface.co/spaces/autogluon/fev-bench
- **強み**: 最大タスク数、共変量対応、統計的厳密性
- **弱み**: Amazon系、事前学習コーパスなし

### 4. TSFM-Bench (FoundTS)

- **発表**: 2024年10月 (arXiv: 2410.11802), KDD 2025
- **規模**: 21多変量データセット, 10ドメイン
- **特徴**: zero/few/full-shotの3評価レジーム、LLMベースモデルも含む14 TSFM
- **強み**: 適応能力の体系的比較
- **弱み**: 小規模、非リークコーパスなし

### 5. TFB

- **発表**: 2024年3月 (arXiv: 2403.20150), PVLDB 2024 (Best Paper Nomination)
- **規模**: 8,068単変量 + 25多変量、21手法
- **特徴**: 統計・ML・DLを公平に比較、「stereotype bias」排除
- **強み**: 統計手法への公平性
- **弱み**: FM特化ではない

### 6. ProbTS

- **発表**: 2023年10月 (arXiv: 2310.07446), NeurIPS 2024
- **著者**: Microsoft Research
- **特徴**: 点予測と確率的予測の相互作用を分析
- **強み**: 確率的予測の深い分析
- **弱み**: 焦点が狭い

---

## 統計手法 vs Foundation Model の比較に対する推奨

### メインベンチマーク: GIFT-Eval

- Foundation Model 17モデルの結果が既にある → 直接比較可能
- 非リーク事前学習コーパスによる公平性
- FMが苦戦するドメイン（交通、Web/Cloud）で統計手法の優位性を示せる可能性

### 補完ベンチマーク: fev-bench

- 共変量付きタスク46個 → 統計手法が共変量活用で差をつけられる
- win rate指標で特定ドメインの強みを主張しやすい
- ブートストラップ信頼区間で統計的有意性を担保
