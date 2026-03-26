# Shifted-Log Poisson FLAIR 実装プラン

## 目標

FLAIR の乗法分解 `M ≈ L × S` を Shifted-Log Poisson NMF に拡張し、
連続データと間欠データを **1パラメータ c** で統一的に扱う。

**論文**: Willwerscheid (2026), arXiv:2601.05845
"A New Family of Poisson NMF Methods Using the Shifted Log Link"

## 数学的定式化

### 現行 FLAIR（c → 0 の極限）
```
mat[i,j] ≈ L[j] × S[i]       (rank-1 multiplicative)
```

### Shifted-Log FLAIR
```
mat[i,j] ~ Poisson(λ[i,j])
α_c · log(1 + λ[i,j] / c) = l[j] + s[i]    (rank-1 in link space)

λ[i,j] = c × (exp((l[j] + s[i]) / α_c) - 1)

where α_c = max(1, c)
```

**挙動**:
- c → 0:  `λ = exp(l) × exp(s) = L × S`  → 現行FLAIR（乗法）
- c → ∞:  `λ = l + s = L + S`              → 加法モデル（ゼロに強い）

## 実装ステップ

### Step 1: rank-1 Shifted-Log NMF solver（新規関数）

ファイル: `src/shifted_log_nmf.py`（新規作成）

```python
def shifted_log_rank1(mat, c, max_iter=20, tol=1e-6):
    """Rank-1 Poisson NMF with shifted-log link.

    Solves: α_c · log(1 + λ/c) = l + s  where λ = outer(exp(s), exp(l))
    via alternating weighted least squares (Taylor approx at zeros).

    Args:
        mat: (P, n_complete) non-negative matrix
        c: shifted-log parameter (float > 0)
        max_iter: max alternating iterations
        tol: convergence tolerance on loglik change

    Returns:
        L: (n_complete,) level vector (in original space, λ scale)
        S: (P,) shape vector (in original space, λ scale, sums to 1)
        loglik: Poisson log-likelihood at convergence
    """
```

**アルゴリズム**:

1. 初期化: 現行FLAIR と同じ（`L = mat.sum(axis=0)`, `S = proportions.mean(axis=1)`）
2. Link transform: `theta = α_c × log(1 + mat/c)`  (ゼロは `theta=0`)
3. Alternating step:
   - Fix s, solve for l:  `l[j] = weighted_mean(theta[:,j] - s)`
   - Fix l, solve for s:  `s[i] = weighted_mean(theta[i,:] - l)`
   - 重み: Poisson IRLS weight `w[i,j] = λ[i,j] / (α_c × (1 + λ[i,j]/c))`
4. 逆変換: `λ[i,j] = c × (exp((l[j] + s[i])/α_c) - 1)`
5. 収束判定: Poisson log-likelihood の変化 < tol

**注意**: c が小さい時 (c < 0.1)、数値安定性のため `log1p` を使用。

### Step 2: BIC による c の自動選択

```python
def select_c_bic(mat, c_candidates=[0.01, 0.1, 1.0, 10.0, 100.0]):
    """Select shifted-log parameter c by BIC (MDL principle).

    BIC = -2 × loglik + k × log(n)
    k = P + n_complete - 1  (rank-1 のパラメータ数)
    n = P × n_complete      (観測数)
    """
    best_c, best_bic = c_candidates[0], np.inf
    for c in c_candidates:
        L, S, loglik = shifted_log_rank1(mat, c)
        P, nc = mat.shape
        k = P + nc - 1
        n = P * nc
        bic = -2 * loglik + k * np.log(n)
        if bic < best_bic:
            best_c, best_bic = c, bic
    return best_c
```

### Step 3: flair_ds への統合

ファイル: `src/experiment_shifted_log.py`（新規作成）

`flair_ds()` をコピーして `flair_shifted_log()` を作成。変更箇所:

#### 3a. Location shift の除去/変更（L95-100）
```python
# 現行: y_shift = max(1 - y_floor, 1.0)  # 全値を正にする
# 提案: Poisson なので非負で十分。ゼロはそのまま残す。
y_shift = max(-y_floor, 0.0)  # min(y) >= 0 にするだけ
```

#### 3b. Shape/Level の計算を Shifted-Log NMF に置換（L155-168）
```python
# 現行:
#   mat = y_trim.reshape(n_complete, P).T
#   S_global = proportions.mean(axis=1)
#   L = mat.sum(axis=0)

# 提案:
mat = y_trim.reshape(n_complete, P).T  # (P, n_complete) — 変更なし
c_opt = select_c_bic(mat)
L, S_global, _ = shifted_log_rank1(mat, c_opt)
S_global = S_global / S_global.sum()  # 正規化
```

#### 3c. Box-Cox の条件付き適用（L231-233）
```python
# c が大きい場合（加法モデル寄り）、Level に多くのゼロが残る可能性
# Box-Cox は正値が前提なので、ゼロがある場合はスキップ
if np.min(L_work) > 0:
    lam = _bc_lambda(L_work)
    L_bc = _bc(L_work, lam)
else:
    lam = 1.0  # identity transform
    L_bc = L_work
```

#### 3d. 乗法復元の修正（L318-324）
```python
# 現行: samples = L_hat × S × (1 + phase_noise) - y_shift
# 提案（c が大きい場合は加法復元も検討）:
if c_opt < 1.0:
    # 乗法復元（現行と同じ）
    samples = L_hat_all[:, step_idx] * S_h[np.newaxis, :] * (1 + phase_noise) - y_shift
else:
    # 加法復元（Shifted-Log の逆変換）
    # λ = c × (exp((l + s) / α_c) - 1)
    # ただし簡易版として: samples = L_hat + S_additive を使ってもよい
    samples = L_hat_all[:, step_idx] * S_h[np.newaxis, :] * (1 + phase_noise) - y_shift
    # NOTE: 初回実験では乗法復元のまま。c選択の効果だけ見る。
```

### Step 4: Quick evaluation（12 config）

```python
QUICK_CONFIGS = [
    # 連続（FLAIRが得意）— c≈0 が選ばれるはず
    ("electricity", "H", ["short"]),
    ("ett1", "H", ["short"]),
    ("solar", "H", ["short"]),
    ("loop_seattle", "H", ["short"]),
    # 間欠/ノイジー（FLAIRが苦手）— c > 1 が選ばれるはず
    ("hierarchical_sales", "D", ["short"]),
    ("restaurant", "D", ["short"]),
    ("kdd_cup_2018", "D", ["short"]),
    ("car_parts", "M", ["short"]),
    # 中間
    ("m4_monthly", "M", ["short"]),
    ("m4_hourly", "H", ["short"]),
    ("hospital", "M", ["short"]),
    ("temperature_rain", "D", ["short"]),
]
```

### Step 5: 比較指標

| 指標 | 意味 |
|------|------|
| relMASE | 点予測精度（SeasonalNaive比） |
| relCRPS | 確率予測精度（SeasonalNaive比） |
| c_selected | BICが選んだ c の値（データセットごと） |
| time_sec | 実行時間（SVD 1回と比べてどれだけ遅いか） |

## 成功基準

1. **連続データで回帰しない**: electricity, solar 等で現行FLAIR（relMASE=0.866）と同等以下
2. **間欠データで改善**: hierarchical_sales, restaurant, car_parts で relMASE 改善
3. **c の自動選択が妥当**: 連続データで c ≈ 0.01、間欠データで c >> 1
4. **計算時間**: 現行の 3倍以内（alternating iteration が追加されるため）

## リスクと対策

| リスク | 対策 |
|--------|------|
| alternating iteration が収束しない | max_iter=20 で打ち切り。初期値を現行FLAIR にする |
| c の BIC 選択が不安定 | c_candidates を粗く [0.01, 1, 100] の3点に絞る |
| 計算が遅い | Quick eval（12 config）で先にスモークテスト |
| 連続データで精度低下 | c=0.01 が現行FLAIR とほぼ等価であることを確認 |
| Poisson 仮定が実数値データに合わない | 整数丸めするか、quasi-Poisson（分散スケーリング）を使う |

## ファイル構成

```
src/
  shifted_log_nmf.py           # Step 1-2: NMF solver + BIC selector
  experiment_shifted_log.py     # Step 3-4: flair_shifted_log() + quick eval
```

## 依存関係

- numpy, scipy のみ（新規ライブラリ不要）
- `src/run_gift_eval_flair_ds.py` の `flair_ds()` をベースにコピー
- `src/run_gift_eval_flar9.py` のユーティリティ（`_bc`, `_ridge_gcv_loo_softavg` 等）

## 実行コマンド

```bash
# Step 1: NMF solver の単体テスト
uv run python -c "from shifted_log_nmf import shifted_log_rank1; ..."

# Step 2-4: Quick evaluation
uv run python -u src/experiment_shifted_log.py

# Step 5: 結果が良ければフル 97 config
uv run python -u src/experiment_shifted_log.py --full
```
