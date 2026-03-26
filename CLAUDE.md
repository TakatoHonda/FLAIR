# CompCast Research — FLAIR

## FLAIR概要
- **Factored Level And Interleaved Ridge**: 周期時系列を Level × Shape に分解し、Level を Ridge で予測
- 1 SVD, 0 hyperparameters, CPU only
- GIFT-Eval: relMASE=0.885, relCRPS=0.663 (97 configs, 23 datasets)
- **Chronos-small (46M params, GPU) を超えた** (0.892)
- Per-horizon: short=0.885, medium=0.865, long=0.906

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

### Location shift が非負系列を壊す問題 (CRITICAL)
- **原因**: `y += shift` で L = sum(y+shift) = L_orig + P×shift。V9 は `L+1` だが MDL は `L+P`
- **影響**: electricity/D が 1.84→3.17、bitbrains_fast_storage/H が V9 比で悪化
- **未修正**: 負値系列は shift 必須（ETT2）、非負系列は shift 有害。if/else で分岐すると非負系列は V9 互換に戻るが、MDL の SVD 計算との相互作用でまだ差が出る
- **根本解決案**: shift を raw y ではなく Level に適用する方式。ただし Shape との乗法復元で `(L-shift)*S ≠ L*S - shift` のため不整合

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
- `src/generate_fig4_benchmark.py` で生成
- フィルタ: agentic除外, test leakage除外, orchestration除外 (DeOS), 外れ値除外 (AutoETS, Crossformer, VISIT-1.0)
- Prophet, MFLES はローカル結果 (`results/15_gift_eval/`)
- リーダーボードデータ: `/tmp/gift-eval-space/results/` (HF space clone)

## ファイル構成
- `flair.py` — コアアルゴリズム（公開用）
- `research/benchmarks/run_gift_eval_flair_ds.py` — FLAIR-DS 本体 (Dirichlet Shape)
- `research/benchmarks/run_gift_eval_flar9.py` — V9 / Ridge GCV-SA / 共通ユーティリティ
- `research/benchmarks/evaluate_m5_flair_quick.py` — M5 評価 (Router 付き)
- `research/benchmarks/evaluate_m5_flair_exog.py` — M5 評価 (外部変数注入)
- `research/benchmarks/evaluate_m5_wrmsse.py` — M5 WRMSSE + 既存統計手法
- `research/figures/generate_fig4_benchmark.py` — ベンチマーク散布図生成
- `research/figures/visualize_flair_pipeline.py` — パイプライン可視化
- `docs/flair_tech_report/` — テクニカルレポート + 図
