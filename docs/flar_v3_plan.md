# FLAR v3: Direct Multi-Step Ridge with Exponential Weighting

## 概要

FLAR B (MASE=0.911, CRPS=0.773) の3つの弱点を同時に解消する改良:

1. **Recursive forecast の誤差蓄積** → Direct multi-step で除去
2. **全データ点の等重み** → 指数重み付けで非定常適応
3. **固定振幅の季節性** → Trend×Fourier 交互作用で変動季節性を捕捉

全て線形代数の範囲内。if/else は追加しない。

## 現状 (FLAR B) の構造

```python
# 1. Box-Cox λ (MLE)
# 2. 特徴量: [1, t/T, cos/sin(focused_periods), y[t-1], y[t-period]]
# 3. Ridge GCV-α (SVD)
# 4. Recursive forecast (for h in range(H): predict → feed back)
# 5. LOO Conformal (全 horizon step で同じ残差プール)
```

## FLAR v3 の構造

```python
# 1. Box-Cox λ (MLE)                    ← 変更なし
# 2. 特徴量: [1, t/T, cos/sin, t/T×cos/sin, y[t-1], y[t-period]]  ← 交互作用追加
# 3. 指数重み付き Ridge GCV-α (Weighted SVD)   ← 重み追加
# 4. Direct multi-step: Y = [y[t+1], ..., y[t+H]]  ← recursive 廃止
# 5. LOO Conformal per-horizon                       ← h ごとに独立較正
```

## 実装詳細

### 2. 特徴量: Trend×Fourier 交互作用

```python
# 現在
cols = [1, t/T, cos(2πt/p), sin(2πt/p), ...]

# v3: 交互作用項を追加
cols = [1, t/T,
        cos(2πt/p), sin(2πt/p),         # 季節性
        (t/T)*cos(2πt/p), (t/T)*sin(2πt/p),  # 振幅変動 (NEW)
        y[t-1], y[t-period]]
```

特徴量数: n_base + 2*n_fourier (交互作用) + n_lag
focused Fourier で periods ≈ 3-4 → 交互作用 6-8 列追加 → 合計 ~20 特徴量。
Ridge の正則化が不要な交互作用を自動でゼロに押す。

### 3. 指数重み付き Ridge

```python
# 半減期: horizon の 2 倍 (直近データを重視しつつ過去も活用)
half_life = horizon * 2
decay = np.log(2) / half_life
weights = np.exp(-decay * np.arange(n_train-1, -1, -1))  # 新しい方が大きい
W_sqrt = np.sqrt(weights)

# Weighted Ridge = 通常の Ridge に W_sqrt を掛けるだけ
X_w = X * W_sqrt[:, np.newaxis]
y_w = y * W_sqrt

# あとは通常の SVD-Ridge
U, s, Vt = svd(X_w)
...
```

追加コスト: O(n × p) の要素積のみ。無視できる。

### 4. Direct Multi-Step (核心の改良)

**現在 (recursive):**
```python
y_ext = concat(y_t, zeros(H))
for h in range(H):
    x = build_features(t=n+h, lag1=y_ext[n+h-1], lag_p=y_ext[n+h-period])
    pred = x @ beta
    y_ext[n+h] = pred  # ← 自分の予測をフィードバック (誤差蓄積)
```

**v3 (direct):**
```python
# 訓練データ: 各 horizon step h に対して別の目的変数
# X は全 step で共通、Y だけが異なる
n_usable = n - start - H + 1  # 使える訓練サンプル数
X_train = build_features(y_t, start, n_usable)  # (n_usable, n_feat)
Y_train = np.column_stack([
    y_t[start+h : start+h+n_usable] for h in range(H)
])  # (n_usable, H)

# Weighted Ridge: 1回の分解で H 個の右辺を解く
X_w = X_train * W_sqrt[:, np.newaxis]
Y_w = Y_train * W_sqrt[:, np.newaxis]

U, s, Vt = svd(X_w)  # ONE SVD
s2 = s**2
# GCV で最適 α を選択 (平均 GCV across all H steps)
...

# Beta: (n_feat, H) — 各列が h ステップ先の回帰係数
d = s2 / (s2 + alpha)
Uty = U.T @ Y_w  # (p, H)
Beta = Vt.T @ (d[:, np.newaxis] * Uty / s[:, np.newaxis])  # (n_feat, H)

# 予測: ループなし
x_future = build_features_future(y_t, n, H)  # (n_feat,) — lag は実データのみ使用
fc_t = x_future @ Beta  # (H,) — 全 step を一発で予測
```

**lag 特徴量の扱い (重要):**
- Direct multi-step では lag-1 は `y[n-1]`（最終観測値）で固定
- lag-period は `y[n-period]` で固定
- 予測値のフィードバックが不要 → 誤差蓄積ゼロ
- 但し h が大きいほど lag の情報が古くなる → Ridge が自動で lag 係数を小さくして対応

### 5. LOO Conformal per-horizon

```python
# 各 horizon step h に独立した LOO 残差
H_diag = (U**2) @ d  # (n_usable,) — 共通

for h in range(H):
    resid_h = Y_w[:, h] - X_w @ Beta[:, h]
    loo_resid_h = resid_h / (1 - H_diag)
    # → 各 h に固有の残差分布

# サンプル生成: step h の予測 ± step h の LOO 残差から draw
for s in range(n_samples):
    for h in range(H):
        samples[s, h] = fc[h] + random.choice(loo_resid_h_orig[h])
```

これにより、短い horizon では狭い区間、長い horizon では広い区間が自然に構築される。
現在の「全 step 同じ残差プール」より原理的に正しい。

## 計算量分析

| 操作 | FLAR B (recursive) | FLAR v3 (direct) |
|---|---|---|
| SVD | O(n × p²) | O(n × p²) — 同じ |
| GCV α探索 | O(25 × p) | O(25 × p × H) |
| 係数計算 | O(p²) | O(p² × H) |
| 予測 | O(H × p) recursive | O(p × H) direct — 同じ |
| LOO残差 | O(n) | O(n × H) |
| **合計** | O(n×p² + H×p) | O(n×p² + p²×H + n×H) |

p ≈ 20, H ≈ 48, n ≈ 500 の場合:
- FLAR B: 500×400 + 48×20 ≈ 201,000
- FLAR v3: 500×400 + 400×48 + 500×48 ≈ 243,200

**+21% の計算量増加** — 無視できるレベル。

## 期待される改善

| 指標 | FLAR B | FLAR v3 (期待) | 根拠 |
|---|---|---|---|
| MASE | 0.911 | 0.88-0.90 | Direct で長ホライズン改善 + 指数重みで非定常改善 |
| CRPS | 0.773 | 0.72-0.75 | per-horizon LOO で較正改善 |
| 速度 | 349s | ~400s | 計算量 +21% |
| コード量 | ~150行 | ~170行 | 交互作用 + direct の分 |

## 変更ファイル

`src/flar_ab_test.py` の `flar_B` を改良して `flar_v3` とする。
または新規 `src/run_gift_eval_flar3.py` を作成。

## 実装ステップ

### Step 1: Direct Multi-Step 基盤
- `_build_direct_targets(y_t, start, H)` → Y 行列 (n_usable, H) を構築
- `_ridge_gcv_loo_multi(X_w, Y_w)` → GCV α + H 個の Beta + per-h LOO
- recursive forecast ループを `x_future @ Beta` に置換

### Step 2: 指数重み付け
- `_build_weights(n_train, half_life)` → W_sqrt
- X_w, Y_w の構築

### Step 3: Trend×Fourier 交互作用
- `_build_features` に `t/T * cos`, `t/T * sin` を追加

### Step 4: per-horizon LOO Conformal
- 各 h の LOO 残差を独立に変換・サンプリング

### Step 5: Quick test
```bash
python -u src/run_gift_eval_flar3.py
```
- MASE < 0.911 が目標
- CRPS < 0.773 が目標
- 速度 < 600s が目標

## リスクと対策

| リスク | 対策 |
|---|---|
| Direct で lag 情報が薄れ精度低下 | lag 係数を Ridge が自動調整。悪化したら lag-h (= y[n-h]) を追加特徴量として検討 |
| 指数重みで過去データを捨てすぎ | half_life = horizon*2 で緩やかな減衰。GCV が最適 α で正則化 |
| 交互作用で特徴量過多 | focused Fourier なので追加は 6-8 列のみ。Ridge が pruning |
| per-horizon LOO で残差サンプル不足 | 各 h の LOO は n_usable 個あるので十分（n_usable ≈ 450+） |

## 美しさの評価

```
FLAR B:  Box-Cox → Features → Ridge(SVD) → Recursive → LOO
FLAR v3: Box-Cox → Features' → Weighted Ridge(SVD) → x@Beta → LOO/h

変更点:
  - Recursive ループ → 行列積 (美しい)
  - 等重み → 指数減衰 (自然)
  - 固定季節振幅 → 交互作用 (表現力)
  - 全step同一LOO → per-h LOO (原理的に正しい)

不変:
  - if/else ゼロ
  - モデル選択ゼロ
  - ハイパラ調整ゼロ (GCV が全自動)
  - 1つの SVD 分解から全てが導出
```
