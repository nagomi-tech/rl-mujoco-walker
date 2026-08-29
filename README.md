# rl-mujoco-walker

MuJoCo を使ったブロック型ヒューマノイドロボットの強化学習プロジェクト。

PPO (Proximal Policy Optimization) でフラット地形・坂道・階段の歩行を学習する。

## 環境

- MuJoCo 3.x
- Stable-Baselines3
- Python 3.11+

## セットアップ

```bash
python -m venv venv
source venv/bin/activate
pip install mujoco stable-baselines3 gymnasium imageio
```

## 学習

```bash
# フラット地形（視覚なし）
python train_slope.py --from-scratch --steps 9000000

# 階段（視覚なし）
python train_slope.py --stairs --from-scratch --steps 9000000

# フラット地形（地形視覚観測あり）
python train_slope.py --vision --from-scratch --steps 9000000
```

## 可視化

```bash
# macOS (mjpython 必須)
mjpython view.py --model models/stairs/ckpt_12m/best_model --stairs --speed 0.5
mjpython view.py --vision --speed 0.5
```

## 結果

`results/` ディレクトリに各チェックポイントの動画を保存。

| モデル | 最高報酬 |
|--------|---------|
| stairs_3m  | 207.8 |
| stairs_6m  | 238.5 |
| stairs_9m  | 299.6 |
| stairs_12m | 306.8 |
| stairs_15m | 207.0 |
| stairs_18m | 268.5 |

## ドキュメント

- [docs/baseline_stairs.md](docs/baseline_stairs.md) - 視覚なし階段ベースライン
