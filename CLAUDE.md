# FLAIR

## Git ワークフロー

| ブランチ | 用途 | push |
|---|---|---|
| `main` | 公開用（flaircast/, README, LICENSE, tests/ のみ） | origin に push OK |
| `dev` | 開発用（パッケージの機能開発・テスト） | push しない |

### ルール
- **普段の作業は `dev` ブランチで行う**
- `main` への merge は公開リリース時のみ
- 研究スクリプト・実験結果は `../research/` リポジトリで管理（このリポジトリには含めない）

## FLAIR概要
- **Factored Level And Interleaved Ridge**: 周期時系列を Level × Shape に分解し、Level を Ridge で予測
- 1 SVD, 0 hyperparameters, CPU only
- GIFT-Eval: relMASE=0.857, relCRPS=0.610 (97 configs, 23 datasets)
- **Chronos-small (46M params, GPU) を大幅に超えた** (0.892)
- Per-horizon: short=0.854, medium=0.843, long=0.878

## 最重要ルール: やっても無駄なこと（絶対に再試行するな）

### Frozen Shape Theorem
**Shape を改善する試みは 17 回全て失敗した。Shape は触るな。**
- EWMA weighting, Fourier fitting, Rank-r SVD, exponential smoothing
- James-Stein shrinkage, curvature-regularized Ridge, lag damping
- Deseasonalize-first, multi-variate tensor, hierarchical reconciliation
- Phase-specific intervals, horizon-scaled bands, Fourier interaction
- Rank-1 Quality Gate (sigmoid blend to uniform)
- K=5 average of raw proportions が最適

### Level Ridge の特徴量追加は短い系列で破綻する
- **Velocity feature (lag差分)**: hospital +0.18, m4_monthly +0.12。Ridge の L2 penalty は個別特徴量を縮約できない
- **Feature standardization**: electricity/H/long +0.23。既存特徴量のバランスが崩れる
- **FLAIR-I (integration closure Z=(I-Π_X)CX)**: hospital +0.33, m4_monthly +0.44。特徴量が倍増し過学習
- **Ordered QR Ridge**: singular matrix errors。改善なし
- **Quadratic trend**: recursive forecast で発散

### Location Shift の除去/置換は全て失敗 (2026-03-27, n_samples=200 公平比較)
**Location shift は Shape の暗黙的正則化として機能している。除去するな。**

1. **IHS 置換 (Box-Cox → arcsinh/sinh)**: 全体 +11.5%。Box-Cox の adaptive λ が分散安定化に不可欠。IHS は大きい値で常に log 圧縮し、λ≈1 (identity) や λ≈0.5 (sqrt) が最適な系列を壊す
2. **Conditional shift (非負系列で shift=0)**: 全体 +1.1%。electricity -1% 改善するが m4_hourly +8.6%, solar +2-5% 退行。shift 除去で Shape にハードゼロが生じ、forecast = L×0 = 0 になる
3. **MDL Transform Selection (BIC で identity vs Yeo-Johnson 選択)**: 全体 +1.8%。エレガントだが Shape ゼロ問題は Level 変換では解決不可能
4. **Level-only shift (Box-Cox 前に Level だけ shift)**: P-floor でも 1-floor でも同じ。問題は Level でなく Shape
5. **EWRR (指数重み付き Ridge, H=n/2)**: 全体 +0.8%。m4_yearly -1.7%, m4_monthly -1.8% は改善するが saugeen +6.8% 退行

**根本的発見**: shift `y += max(1-min(y), 1)` は以下の3つの役割を同時に果たしている:
- (a) Box-Cox に正値を保証 — Level shift で代替可能
- (b) Shape の proportions をゼロから遠ざける — **代替手段なし**
- (c) Phase noise の分母 (fitted = L×S) をゼロから遠ざける — L/P floor で部分的に代替可能
特に (b) が重要: shift は乗法復元で自己相殺するため bias は小さいが、Shape ゼロを防ぐ正則化効果は不可欠

**ベンチマーク注意**: DS_RESULTS (n_samples=20 の固定値) は run 間分散が大きい（m4_hourly: 1.18 vs fresh 1.94）。改善判定には必ず n_samples=200 で DS-fresh と同時実行すること

### その他の失敗
- 差分次数ソフト平均 (d=1 vs d=2): ad hoc model averaging。論文に書けない
- Multi-scale soft-average over P: 美しくない（just model averaging）
- Forward-fill NaN: Level dynamics を破壊（electricity/D 1.90→3.17）
- Median NaN fill: 改善するが nan_to_num(0) より悪い

## 現在のアーキテクチャ (FLAIR-MDL)
1. **Location shift**: y += max(1-min(y), 1) → 全値正にして Box-Cox 対応
2. **MDL period selection**: カレンダーテーブルの候補から BIC (on SVD spectrum) で P を選択
3. **P=1 degeneration**: n_complete < 3 → P=1 で Ridge on raw series
4. **Reshape**: mat = y.reshape(n_complete, P).T
5. **Dirichlet Shape**: context = k % C (C = secondary/P)、K×C recency window
6. **Box-Cox → NLinear → Ridge SA**: 1 SVD、25 α のソフト平均
7. **Recursive forecast**: m = ceil(H/P) steps
8. **Conformal intervals**: LOO residuals → bootstrap samples

## 既知のバグ / 未解決問題

### Location shift のトレードオフ (RESOLVED — 除去は不可)
- **理論的バイアス**: `y += shift` で L = L_orig + P×shift。復元時に shift を1回引くため、位相ごとのバイアス = shift×(P×S[p]−1)
- **しかし実験で判明**: shift 除去は全体で +1.1% 退行（n_samples=200, 19 configs 公平比較）。electricity は -1% 改善するが m4_hourly +8.6%, solar +2-5% 退行
- **原因**: shift は Shape のゼロ防止正則化として不可欠。除去すると forecast = L×0 = 0 になる位相が生じる
- **結論**: 理論的バイアスは存在するが、Shape 正則化の便益が上回る。shift はそのまま維持。詳細は「やっても無駄なこと」セクション参照

### bitbrains_fast_storage/H (relMASE=2.27-2.95)
- MDL が P=168 を選択するが nc=3 → Shape/Level が不安定
- BIC の nc >= 5 制約で P=24 に矯正可能だが、location shift の問題と複合

### electricity/D (relMASE=1.6)
- V5 除去による random state ずれが原因の一部
- location shift の `L + P` vs `L + 1` が主因

## 成功した改善（段階的）
| Version | Change | relMASE | Delta |
|---------|--------|---------|-------|
| V9 | Level×Shape, Ridge SA | 1.028 | — |
| +n_samples=200 | conformal samples 増 | 1.014 | -1.4% |
| +Dirichlet Shape | context=k%C, K×C window | 0.983 | -3.1% |
| +Location Shift | y += max(1-min(y),1) | 0.920 | -6.4% |
| +V5 removal | P=1 degeneration | 0.915 | -0.5% |
| +MDL Period | BIC on SVD spectrum | 0.885 | -3.3% |
| +LSR1 diff-target | random walk reparameterization | 0.857 | -3.2% |

## M5 実験結果 (2026-03-24)

### 精度比較 (Level 12 WRMSSE, 30,490 series)
| Method | WRMSSE | vs SN | Time |
|--------|--------|-------|------|
| SeasonalNaive | 1.305 | baseline | 0.1s |
| FLAIR-DS (pure) | 1.122 | -14.1% | 43s |
| FLAIR-Routed (Croston TSB for sparse) | 1.093 | -16.3% | 60s |
| FLAIR-TopDown (store×dept aggregate) | 1.088 | -16.7% | 17s |
| **FLAIR-Exog+Price (calendar+price in Ridge)** | **1.049** | **-19.6%** | **17s** |
| Theta | 1.001 | -23.4% | 285s |
| Kaggle 1st (LightGBM ensemble) | 0.521 | -60.1% | hours |

### 何が効いたか
- **カレンダー変数注入 (SNAP, event, month_sin/cos)**: 最大の改善要因。Level Ridge に特徴量追加するだけで閉形式解を維持
- **Intermittency Router**: 間欠需要 (zero_ratio>0.3) → Croston TSB にルーティング。M5は 84% が間欠

### 何が効かなかったか
- **価格変数**: M5の価格変動が28日horizon内で小さいため微小効果
- **TopDown Disaggregation**: 集約予測は良いが比率分解で情報損失
- **Rank-1 Quality Gate**: Shapeをuniformにblendしたら悪化。Frozen Shape Theorem の追加証拠
- **Hierarchical Hybrid**: TopDown + FLAIR routing の組み合わせも改善せず

### FLAIRがM5で弱い根本原因
1. **P=7 は圧縮が弱い**: P=24 だと24倍圧縮、P=7 は7倍のみ
2. **外部変数ゼロ**: M5は価格・SNAP・イベントが売上を支配
3. **間欠需要**: 30-40% の系列が50%以上ゼロ。Level×Shape は連続需要向け

## GIFT-Eval ベンチマーク図 (fig4)
- `../research/figures/generate_fig4_benchmark.py` で生成
- フィルタ: agentic除外, test leakage除外, orchestration除外 (DeOS), 外れ値除外 (AutoETS, Crossformer, VISIT-1.0)
- Prophet, MFLES はローカル結果 (`../research/results/15_gift_eval/`)
- リーダーボードデータ: `/tmp/gift-eval-space/results/` (HF space clone)

## ファイル構成
- `flaircast/` — パッケージソース（公開用）
- `tests/` — テスト
- `examples/` — 使用例
- 研究用スクリプトは `../research/` を参照
