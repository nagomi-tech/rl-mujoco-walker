# 学習経過ログ

## プロジェクト概要

MuJoCo の二足歩行ロボット（BlockyWalker）を PPO（Stable-Baselines3）で強化学習する。
カリキュラム学習として段階的に地形を複雑化し、視覚観測（前方地形の高さ情報）を活用した汎用歩行を目指す。

---

## 環境設定

| 項目 | 内容 |
|------|------|
| アルゴリズム | PPO (Stable-Baselines3) |
| ポリシー | MlpPolicy [256, 256] × 2（pi / vf） |
| 並列環境数 | 8（SubprocVecEnv） |
| 観測次元 | 35次元（基本）+ 10次元（地形視覚）= 45次元 |
| 観測正規化 | VecNormalize（clip_obs=10.0） |

---

## 報酬設計（現行）

```python
# 前進速度報酬
vx = float(self.data.qvel[0])
reward = vx * 2.0

# 直立維持報酬（±10度の遊びを許容）
torso_mat = self.data.xmat[self._torso_id].reshape(3, 3)
upright = float(torso_mat[2, 2])  # cos(傾き角)
_upright_threshold = 0.9848  # cos(10°)
if upright < _upright_threshold:
    reward += (upright - _upright_threshold) * 0.5

# ゴール到達ボーナス
if torso_x >= self._goal_x:
    reward += 200.0
```

**変更履歴:**
- 初期: ゴール進捗（距離変化 × 50）のみ
- 変更1: 前進速度（vx × 2.0）に変更
- 変更2: 直立維持報酬を追加（upright × 0.5）
- 変更3: ±10度の遊びを設けてペナルティ方式に変更

---

## リセット設計（現行）

```python
# 胴体位置・姿勢を固定
self.data.qpos[0:3] = [0.0, 0.0, 1.05]
self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

# 関節角度をランダム初期化（有効範囲の半分程度）
joint_ranges = [
    (-0.3,  0.3),   # left_shoulder
    (-0.4,  0.0),   # left_elbow
    (-0.3,  0.3),   # right_shoulder
    (-0.4,  0.0),   # right_elbow
    (-0.3,  0.1),   # left_hip
    ( 0.0,  0.4),   # left_knee
    (-0.15, 0.15),  # left_ankle
    (-0.3,  0.1),   # right_hip
    ( 0.0,  0.4),   # right_knee
    (-0.15, 0.15),  # right_ankle
]

# 速度をランダム化（小さめ）
self.data.qvel[:] = np.random.uniform(-0.05, 0.05, nv)
```

**変更履歴:**
- 初期: 全ジョイントに微小ノイズ（±0.005）
- 変更: 各関節の物理的有効範囲の約半分でランダム初期化（学習安定化のため）

---

## 学習経過

### Phase 1: 坂道 5度 × vision_flat（失敗）

- `train_slope.py --vision`（`--slope` 未指定 → デフォルト 5.0度）
- `vision_flat` ラベルを使用したが、実際は **5度坂道**でのトレーニング
- 6M → 30M ステップまで学習

**問題:**
- 18M以降、eval報酬は伸びず（~900 → 変化なし）
- 動画確認で「最初の一歩が大きすぎて前傾転倒」するパターンを確認
- 5度坂を上るために大きく足を上げる行動が平坦地でも固定化された

**対策:**
1. 直立維持報酬の追加
2. 初期関節角度のランダム範囲を縮小
3. clip_range=0.1、lr=1e-4 でポリシー崩壊を抑制

**過去の崩壊:**
- default hyperparameter（clip_range=0.2, lr=3e-4）での15M以上の学習で std が 1.6 → 103 → 15,200 に爆発
- clip_range=0.1 + lr=1e-4 で std=3.56 まで抑制成功（15M時点）

---

### Phase 2: 坂道 5度 × vision × 報酬改善版（6M→15M）【廃棄】

- `--vision --slope 0 (誤: 5度) --start-m 6 --lr 1e-4 --clip-range 0.1`
- 直立維持報酬（upright × 0.5）を追加
- さらに ±10度の遊びを追加してペナルティ方式に変更
- eval報酬: 6M時 345 → 15M時 608

**廃棄理由:** 依然として 5度坂道での学習であり、カリキュラム設計の根本的な見直しが必要

---

### Phase 3: 平坦地 × vision × ゼロから学習（現行）

**設定:**

| 項目 | 値 |
|------|-----|
| slope_deg | 0.0（平坦地） |
| terrain_vision | True |
| lr | 3e-4（デフォルト） |
| clip_range | 0.2（デフォルト） |
| 初期化 | ランダム（ゼロから） |
| 目標 | 6M → 継続判断 |

**結果（6M時点）:**

| 指標 | 値 |
|------|-----|
| eval報酬 | **3,248**（±358） |
| ep_len_mean | 944ステップ |
| std | 2.0（安定） |
| approx_kl | 0.23（やや高め） |

動画（results/vision_flat_v2_6m.mp4）でエピソード1は1,000ステップ上限まで歩行継続を確認。

---

## 今後のカリキュラム計画（案）

1. **フェーズ1（現在）**: 平坦地 slope=0 で基本歩行を習得
2. **フェーズ2**: 緩い坂道（slope=3°程度）に転移学習
3. **フェーズ3**: 中程度の坂道（slope=5°〜8°）
4. **フェーズ4**: ドメインランダマイゼーション（slope=0°〜8° をランダム選択）
   - 視覚観測が地形判断に活用されるようになることを期待

---

## ファイル構成

```
rl-mujoco-walker/
├── envs/
│   └── walker_env.py        # 環境定義（報酬・リセット・観測）
├── models/
│   └── vision_flat/         # 現行学習モデル（slope=0 + vision）
│       ├── ckpt_03m/        # 3Mステップ時のベストモデル
│       ├── ckpt_06m/        # 6Mステップ時のベストモデル ← 現在地
│       ├── best/            # EvalCallback が更新するベスト
│       ├── final.zip        # 最終チャンクのモデル
│       └── vecnorm.pkl      # 観測正規化統計
├── train_slope.py           # 坂道・vision学習スクリプト
├── train.py                 # カリキュラム学習（level 0→1→2）
├── train_loop.py            # チャンク学習ループ
├── evaluate.py              # 評価スクリプト
├── view.py                  # リアルタイム可視化
└── results/
    └── vision_flat_v2_6m.mp4  # 6M時点の動画（平坦地）
```

---

## 動画確認コマンド

```bash
# リアルタイム視聴（macOS: mjpython 必要）
mjpython view.py --model models/vision_flat/ckpt_06m/best_model --vision --speed 0.5

# 動画ファイル生成
arch -arm64 venv/bin/python train_slope.py --vision --slope 0 \
    --start-m 6 --steps 3000000 --lr 0.0001 --clip-range 0.1
```

---

## 重要な知見

- `vision_flat` ラベルは「--vision フラグを使う際のラベル名」であり、地形が平坦であることを保証しない。`--slope 0` を明示的に指定すること。
- VecNormalize の `.pkl` はモデルと対になっている。チェックポイントの `vecnorm.pkl` と `best_model.zip` は同タイミングで保存されたものを使うこと（不一致だと推論精度が大幅低下する）。
- `approx_kl > 0.2` はポリシー崩壊の予兆。`clip_range=0.1` + `lr=1e-4` で抑制可能。
- ゼロから平坦地学習した場合、6M で eval 報酬 3,248（坂道15M の 608 と比べて大幅優位）。
