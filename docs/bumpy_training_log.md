# 凹凸地形（bumpy）歩行学習ログ

## 概要

BlockyWalkerEnvで凹凸地形（heightfield）上の二足歩行を学習するための試行錯誤の記録。

---

## 問題と解決策

### 1. std爆発（最重要問題）

**症状**
- `train/std` が学習開始から指数関数的に増加（1.0 → 36.9 @ 15M）
- `clip_fraction` が 0.5〜0.8 に張り付く（健全値は 0.1〜0.3）
- `approx_kl` が際限なく上昇（0.08 → 0.8）
- 結果として bang-bang制御（アクションが常に±1の極値）になり、歩行不能

**原因**
- `target_kl` 未設定のため、PPOの更新幅に上限がなかった
- 1ロールアウトあたり10エポックの更新で方策が動きすぎていた

**解決策**
```python
model = PPO(..., target_kl=0.02, ...)
```

**検証結果（3Mステップ比較）**

| ステップ | std（旧） | std（target_kl=0.02） |
|---------|----------|-----------------------|
| 1M      | 1.03     | 1.01                  |
| 3M      | 1.19     | 1.02                  |
| 9M      | 3.01     | 1.04                  |

---

### 2. 足のオフセット問題（つま先立ち）

**症状**
- 常につま先立ちになる
- 前方への転倒時に無理やり速度を出そうとする

**原因**
```xml
<!-- 変更前: 足首ジョイントから前方4cmにオフセット -->
<geom name="left_foot_geom" size="0.13 0.08 0.04" pos="0.04 0 -0.04"/>
```
足首ジョイントの前方17cm・後方9cmの非対称配置により、前進報酬で前傾するとつま先が最適戦略になる。

**解決策**
```xml
<!-- 変更後: 前後対称 -->
<geom name="left_foot_geom" size="0.13 0.08 0.04" pos="0.0 0 -0.04"/>
```

**効果**
- 自然な足裏接地が可能に
- best reward が 136.7 → 463.6 に向上

---

### 3. 転倒ペナルティの追加

**症状**
- エピソード終了直前に意図的に前方へ倒れ込む行動（報酬ハッキング）
- 転倒時 vx が急上昇し大きな速度報酬を受け取ってから終了

**原因**
エピソード終了は暗黙のペナルティだが、終了ステップの報酬スパイクは防げない。

**解決策**
```python
if not healthy:
    reward -= 10.0
```

---

### 4. 速度報酬の上限設定（tanh）

**症状**
- 線形速度報酬 `vx * 2.0` により「できるだけ速く走る」戦略を学習
- 実測速度: 平均1.79 m/s、最大3.52 m/s（転倒スパイク）

**解決策**
```python
V_TARGET = 1.5  # m/s
reward = float(np.tanh(vx / V_TARGET)) * 2.0
```

**目標速度の調整経緯**

| V_TARGET | 結果 |
|----------|------|
| なし（線形） | 急いで転倒。平均1.79 m/s |
| 0.8 m/s  | 遅すぎ。すり足でほぼ進まない |
| 1.2 m/s  | やや遅い |
| 1.5 m/s  | 適切。バンプで足を上げる動作が出現 |

---

### 5. 初期姿勢の固定

**変更**
```python
# 変更前: ランダム初期化（膝 0〜0.4rad など）
for i, (lo, hi) in enumerate(joint_ranges):
    self.data.qpos[7 + i] = self.np_random.uniform(lo, hi)

# 変更後: 全関節0（直立固定）、速度ゼロ
self.data.qpos[7:17] = 0.0
self.data.qvel[:] = 0.0
```

---

### 6. 視覚情報（terrain_vision）の常時有効化

```python
# train_slope.py: --bumpy 時に自動で vision=True
if args.bumpy:
    args.vision = True
```

地形の高さサンプル10次元を観測に追加。バンプで足を持ち上げる行動が観察された。

---

### 7. ランダム地形による汎化学習

固定地形（seed=0）での学習は特定の地形を記憶するだけになる。
エピソードごとに異なる地形で学習することで真の汎化を狙う。

```python
# walker_env.py reset() 内
if self._bumpy:
    self._regenerate_terrain(self.np_random)  # seed固定 → ランダム化
```

---

## ハイパーパラメータ変更まとめ

| パラメータ | 変更前 | 変更後 | 理由 |
|-----------|-------|-------|------|
| `target_kl` | 未設定 | 0.02 | std爆発防止（最重要） |
| `learning_rate` | 3e-4 | 1e-4 | 更新幅を抑制 |
| `V_TARGET` | なし（線形） | 1.5 | 過度な速度追求を抑制 |

---

## 学習の推移

| フェーズ | 累計ステップ | best reward | 主な変更点 |
|---------|------------|-------------|-----------|
| std爆発期 | 〜15M | 136.7 | target_klなし、足オフセットあり |
| 修正後 | 9M | 463.6 | target_kl=0.02, 足オフセット修正, 転倒ペナルティ |
| tanh報酬 | 9M | 164.1 | V_TARGET=1.5、固定地形 |
| ランダム地形継続 | 18M | 312.9 | エピソードごとランダム地形 |
| 継続中 | 48M（予定） | - | - |

---

## 継続学習の実装

`models/bumpy/best/best_model.zip` が存在すれば自動的に継続学習する。

```bash
python train_slope.py --bumpy --steps 9000000          # 継続（自動検出）
python train_slope.py --bumpy --steps 9000000 --from-scratch  # ゼロから
```

---

## 現在の環境設定（walker_env.py）

```python
# 速度報酬（tanh、上限あり）
V_TARGET = 1.5
reward = float(np.tanh(vx / V_TARGET)) * 2.0

# ペナルティ
if not healthy:           reward -= 10.0              # 転倒
if self._shin_on_ground(): reward -= 1.0              # 脛接地
if inter_leg_angle > 100*pi/180:                       # 足間角度超過
    reward -= (inter_leg_angle - INTER_LEG_LIMIT) * 2.0
if height_above_floor < 0.75:                          # 腰落ち
    reward -= (0.75 - height_above_floor) * 2.0
```

```xml
<!-- 足geom（左右共通）: 前後対称 -->
<geom name="left_foot_geom" type="box" size="0.13 0.08 0.04"
      pos="0.0 0 -0.04"/>
```
