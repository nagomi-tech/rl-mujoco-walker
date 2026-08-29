"""
3Mステップずつ学習し、毎回動画を保存するループスクリプト。
目標: 合計15Mステップ

使い方:
    python train_loop.py
    python train_loop.py --start-at 6  # 6M済み→9M,12M,15Mのみ
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.walker_env import BlockyWalkerEnv

MODELS_DIR = Path("models")
LOGS_DIR   = Path("logs")
RESULTS_DIR = Path("results")
N_ENVS = 8
CHUNK_STEPS = 3_000_000


def evaluate_and_save_video(step_millions: int,
                             model_path: Path,
                             vecnorm_path: Path | None,
                             n_episodes: int = 3,
                             max_steps: int = 1500):
    """ベストモデルで評価し動画を保存する。"""
    import imageio

    label = f"{step_millions}M"
    video_path = RESULTS_DIR / f"level0_{label}.mp4"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== 動画生成: {video_path} (モデル: {model_path}) ===")

    env = DummyVecEnv([lambda: BlockyWalkerEnv(level=0, render_mode="rgb_array")])
    if vecnorm_path and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    best_frames = []
    best_reward = -np.inf

    for ep in range(n_episodes):
        obs = env.reset()
        frames = []
        total_reward = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            frame = env.envs[0].render()
            if frame is not None:
                frames.append(frame)
            if done[0]:
                break
        print(f"  Episode {ep+1}: 報酬={total_reward:.1f}, フレーム数={len(frames)}")
        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    env.close()

    if best_frames:
        with imageio.get_writer(str(video_path), fps=30) as writer:
            for frame in best_frames:
                writer.append_data(frame)
        print(f"  => 保存完了: {video_path}")
    else:
        print("  警告: フレームが取得できませんでした")


def train_chunk(chunk_idx: int,
                resume_model_path: Path,
                vecnorm_path: Path | None) -> tuple[Path, Path]:
    """3Mステップ学習して、finalモデルパスとvecnorm pathを返す。"""
    level = 0
    print(f"\n{'='*60}")
    print(f"  Chunk {chunk_idx}: {(chunk_idx)*3}M → {(chunk_idx+1)*3}M ステップ")
    print(f"  再開モデル: {resume_model_path}")
    print(f"{'='*60}")

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # 学習環境
    venv = make_vec_env(lambda: BlockyWalkerEnv(level=level), n_envs=N_ENVS)
    if vecnorm_path and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), venv)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 評価環境
    eval_venv = make_vec_env(lambda: BlockyWalkerEnv(level=level), n_envs=1)
    if vecnorm_path and vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_venv)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

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

    # モデル読み込み
    model = PPO.load(
        str(resume_model_path),
        env=train_env,
        tensorboard_log=str(LOGS_DIR / f"level{level}"),
    )

    model.learn(
        total_timesteps=CHUNK_STEPS,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # 保存
    final_path = MODELS_DIR / f"level{level}_final"
    new_vecnorm_path = MODELS_DIR / f"level{level}_vecnorm.pkl"
    model.save(str(final_path))
    train_env.save(str(new_vecnorm_path))

    eval_env.close()
    train_env.close()

    print(f"\n  Chunk {chunk_idx} 完了: {final_path}.zip")
    return final_path, new_vecnorm_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-at", type=int, default=3,
                        help="現在の累積Mステップ数（デフォルト=3: 3M済みから再開）")
    args = parser.parse_args()

    current_m = args.start_at  # 現在の累積ステップ数（M単位）
    target_m = 15

    # 現在のモデルで最初の動画を保存
    best_model = MODELS_DIR / "level0" / "best" / "best_model.zip"
    vecnorm = MODELS_DIR / "level0_vecnorm.pkl"
    if not vecnorm.exists():
        vecnorm = MODELS_DIR / "level0" / "best" / "best_vecnormalize.pkl"

    print(f"=== 現在 {current_m}M ステップ済み → {target_m}M まで学習 ===")
    print(f"ベストモデル: {best_model}")
    print(f"VecNormalize: {vecnorm}")

    # まず現在のモデルで動画保存
    if best_model.exists():
        evaluate_and_save_video(current_m, best_model, vecnorm if vecnorm.exists() else None)
    else:
        print(f"警告: {best_model} が見つかりません。動画生成をスキップします。")

    # 学習ループ
    resume_model = MODELS_DIR / "level0_final"
    if not (resume_model.with_suffix(".zip")).exists():
        # final がなければ best model から再開
        resume_model = MODELS_DIR / "level0" / "best" / "best_model"

    chunk_idx = current_m // 3  # 何チャンク目か

    while current_m < target_m:
        final_path, vecnorm = train_chunk(chunk_idx, resume_model, vecnorm)
        current_m += 3
        chunk_idx += 1

        # ベストモデルで動画保存
        best_model_after = MODELS_DIR / "level0" / "best" / "best_model.zip"
        vn_best = MODELS_DIR / "level0" / "best" / "best_vecnormalize.pkl"
        vn_for_video = vn_best if vn_best.exists() else vecnorm
        evaluate_and_save_video(current_m, best_model_after, vn_for_video)

        # 次チャンクの再開元をfinalモデルに更新
        resume_model = final_path

    print(f"\n=== 全チャンク完了: {target_m}M ステップ ===")
    print("動画は results/ フォルダに保存されました。")


if __name__ == "__main__":
    main()
