"""
学習済みモデルを評価し、3つのレベルの動画を生成する。

使い方:
    python evaluate.py                    # 全レベルの動画を生成
    python evaluate.py --level 0          # Level 0 のみ
    python evaluate.py --no-video         # 動画なしで報酬だけ確認
    python evaluate.py --level 2 --episodes 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from envs.walker_env import BlockyWalkerEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("stable-baselines3 が見つかりません。pip install stable-baselines3")
    sys.exit(1)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def find_model(level: int) -> Path | None:
    """利用可能な学習済みモデルを探す。"""
    candidates = [
        MODELS_DIR / f"level{level}" / "best" / "best_model.zip",
        MODELS_DIR / f"level{level}_final.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def find_vecnorm(level: int) -> Path | None:
    candidates = [
        MODELS_DIR / f"level{level}" / "best" / "best_vecnormalize.pkl",
        MODELS_DIR / f"level{level}_vecnorm.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_model_and_env(level: int):
    model_path = find_model(level)
    if model_path is None:
        print(f"Level {level} のモデルが見つかりません（先に train.py を実行してください）")
        return None, None

    env = DummyVecEnv([lambda lv=level: BlockyWalkerEnv(level=lv, render_mode="rgb_array")])

    vn_path = find_vecnorm(level)
    if vn_path:
        env = VecNormalize.load(str(vn_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"Level {level}: VecNormalize ファイルが見つかりません（精度が落ちる可能性）")

    model = PPO.load(str(model_path), env=env)
    print(f"Level {level} モデル読み込み: {model_path}")
    return model, env


def run_episode(model, env, max_steps: int = 1000):
    """1エピソードを実行しフレームと累積報酬を返す。"""
    obs = env.reset()
    frames = []
    total_reward = 0.0
    done = False

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])

        frame = env.envs[0].render()
        if frame is not None:
            frames.append(frame)

        if done[0]:
            break

    return frames, total_reward


def save_video(frames: list, path: Path, fps: int = 30):
    try:
        import imageio
    except ImportError:
        print("imageio がインストールされていません")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  動画保存: {path}")


def evaluate_level(level: int, n_episodes: int, save_video_flag: bool):
    level_name = BlockyWalkerEnv.LEVEL_NAMES[level]
    print(f"\n--- Level {level}: {level_name} ---")

    model, env = load_model_and_env(level)
    if model is None:
        return

    rewards = []
    best_frames = []
    best_reward = -np.inf

    for ep in range(n_episodes):
        frames, rew = run_episode(model, env)
        rewards.append(rew)
        print(f"  Episode {ep+1}: 累積報酬 = {rew:.1f}")
        if rew > best_reward:
            best_reward = rew
            best_frames = frames

    print(f"  平均: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}  最高: {np.max(rewards):.1f}")

    if save_video_flag and best_frames:
        video_path = RESULTS_DIR / f"level{level}_{level_name.lower()}.mp4"
        save_video(best_frames, video_path)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=None, choices=[0, 1, 2],
                        help="評価するレベル（未指定で全レベル）")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else [0, 1, 2]

    for lvl in levels:
        evaluate_level(lvl, args.episodes, not args.no_video)

    print("\n評価完了。動画は results/ フォルダに保存されました。")


if __name__ == "__main__":
    main()
