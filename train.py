"""
カリキュラム学習: Level 0 → 1 → 2 の順に段階的に訓練する。
各レベルの学習済み重みを次のレベルの初期値として引き継ぐ。

使い方:
    python train.py                         # Level 0 から全レベル学習
    python train.py --start-level 1         # Level 1 から開始（Level 0 モデル必要）
    python train.py --level 0 --timesteps 200000  # Level 0 のみ短時間で試す
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))
from envs.walker_env import BlockyWalkerEnv

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")
N_ENVS = 8

# レベルごとのデフォルト学習ステップ数
DEFAULT_TIMESTEPS = {
    0: 600_000,   # 平地歩行
    1: 500_000,   # 壁の隙間を通過
    2: 500_000,   # キューブ回避
}


def make_train_env(level: int, prev_vecnorm_path: Path | None = None):
    venv = make_vec_env(
        lambda: BlockyWalkerEnv(level=level),
        n_envs=N_ENVS,
    )
    # obs4 の意味がレベルごとに異なる（Level 2 はキューブ情報を含む）ため、
    # 前レベルの統計を引き継ぐと obs4[2,3] の分散=0 問題が起きる。
    # Level 0→1 は obs4[0] のみ変化するため引き継ぎ可能。
    # Level 2 は obs4 全体が変わるため必ず新規作成する。
    if level < 2 and prev_vecnorm_path and prev_vecnorm_path.exists():
        vn = VecNormalize.load(str(prev_vecnorm_path), venv)
        vn.training = True
        vn.norm_reward = True
        return vn
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)


def make_eval_env(level: int, prev_vecnorm_path: Path | None = None):
    venv = make_vec_env(
        lambda: BlockyWalkerEnv(level=level),
        n_envs=1,
    )
    if level < 2 and prev_vecnorm_path and prev_vecnorm_path.exists():
        vn = VecNormalize.load(str(prev_vecnorm_path), venv)
        vn.training = False
        vn.norm_reward = False
        return vn
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def build_model(env, level: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        tensorboard_log=str(LOGS_DIR / f"level{level}"),
    )


def train_level(level: int, timesteps: int,
                resume_model_path: Path | None = None,
                prev_vecnorm_path: Path | None = None) -> Path:
    """
    指定レベルを学習する。
    - resume_model_path: 前レベルのモデル重みを引き継ぐ
    - prev_vecnorm_path: 前レベルの VecNormalize 統計を引き継ぐ（重要）
    完成したモデルのパスを返す。
    """
    level_name = BlockyWalkerEnv.LEVEL_NAMES[level]
    print(f"\n{'='*55}")
    print(f"  Level {level}: {level_name}  ({timesteps:,} ステップ)")
    if prev_vecnorm_path and prev_vecnorm_path.exists():
        print(f"  VecNormalize 引き継ぎ: {prev_vecnorm_path}")
    print(f"{'='*55}")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    train_env = make_train_env(level, prev_vecnorm_path)
    eval_env = make_eval_env(level, prev_vecnorm_path)

    # コールバック
    save_freq = max(1, 50_000 // N_ENVS)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(MODELS_DIR / f"level{level}"),
        name_prefix=f"ppo_l{level}",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / f"level{level}" / "best"),
        log_path=str(LOGS_DIR / f"level{level}"),
        eval_freq=save_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # モデル作成（前レベルから重みを引き継ぐ場合）
    if resume_model_path and resume_model_path.exists():
        print(f"  モデル重み引き継ぎ: {resume_model_path}")
        model = PPO.load(
            str(resume_model_path),
            env=train_env,
            tensorboard_log=str(LOGS_DIR / f"level{level}"),
        )
    else:
        model = build_model(train_env, level)

    model.learn(
        total_timesteps=timesteps,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # 最終モデルを保存
    final_path = MODELS_DIR / f"level{level}_final"
    model.save(str(final_path))
    train_env.save(str(MODELS_DIR / f"level{level}_vecnorm.pkl"))
    eval_env.close()
    train_env.close()

    print(f"\n  Level {level} 完了: {final_path}.zip")
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-level", type=int, default=0, choices=[0, 1, 2],
                        help="開始レベル（前レベルのモデルが必要）")
    parser.add_argument("--level", type=int, default=None, choices=[0, 1, 2],
                        help="このレベルだけ学習（--start-level より優先）")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="学習ステップ数（未指定ならレベルごとのデフォルト値）")
    args = parser.parse_args()

    if args.level is not None:
        levels_to_train = [args.level]
    else:
        levels_to_train = list(range(args.start_level, 3))

    prev_model_path = None
    prev_vn_path = None

    for lvl in levels_to_train:
        # 前レベルのモデルと VecNormalize 統計を探す
        if lvl > 0:
            model_cand = MODELS_DIR / f"level{lvl - 1}_final.zip"
            vn_cand = MODELS_DIR / f"level{lvl - 1}_vecnorm.pkl"
            if model_cand.exists():
                prev_model_path = model_cand.with_suffix("")
            else:
                print(f"警告: Level {lvl-1} のモデルが見つかりません。ランダム初期化で開始します。")
            if vn_cand.exists():
                prev_vn_path = vn_cand
            else:
                print(f"警告: Level {lvl-1} の VecNormalize が見つかりません。新規統計で開始します。")

        ts = args.timesteps if args.timesteps else DEFAULT_TIMESTEPS[lvl]
        saved_path = train_level(lvl, ts,
                                 resume_model_path=prev_model_path,
                                 prev_vecnorm_path=prev_vn_path)
        prev_model_path = saved_path
        prev_vn_path = MODELS_DIR / f"level{lvl}_vecnorm.pkl"

    print("\n全レベルの学習が完了しました。")
    print("次のステップ: python evaluate.py --level 2")


if __name__ == "__main__":
    main()
