"""
combined モデルの動画を生成する。

使い方:
    python record_combined.py
    python record_combined.py --episodes 5 --output results/combined.mp4
    python record_combined.py --vecnorm models/combined/ppo_slope_vecnormalize_61850000_steps.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from envs.walker_env import BlockyWalkerEnv

try:
    import mujoco
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("依存パッケージが見つかりません")
    sys.exit(1)

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def make_env():
    return BlockyWalkerEnv(
        level=0,
        render_mode="rgb_array",
        terrain_vision=True,
        combined=True,
    )


def render_frame(env_inner, renderer, camera):
    """カスタムカメラでフレームをレンダリングする。"""
    # ロボットのtorso位置を取得してカメラのlookat更新
    torso_id = mujoco.mj_name2id(env_inner.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_pos = env_inner.data.xpos[torso_id]
    camera.lookat[:] = torso_pos
    renderer.update_scene(env_inner.data, camera=camera)
    return renderer.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/combined/best/best_model.zip")
    parser.add_argument("--vecnorm", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="results/combined_best.mp4")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # vecnorm を探す
    if args.vecnorm:
        vn_path = Path(args.vecnorm)
    else:
        vn_path = MODELS_DIR / "combined" / "vecnorm.pkl"

    print(f"モデル: {args.model}")
    print(f"vecnorm: {vn_path}")

    vec_env = DummyVecEnv([make_env])

    if vn_path and vn_path.exists():
        vec_env = VecNormalize.load(str(vn_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("警告: VecNormalize ファイルが見つかりません")

    model = PPO.load(args.model, env=vec_env)

    # カスタムカメラ設定（サイドビュー、ロボット追従）
    env_inner = vec_env.envs[0]
    renderer = mujoco.Renderer(env_inner.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90      # 真横から（X軸方向の移動が見える）
    camera.elevation = -15   # 少し見下ろす
    camera.distance = 8.0    # 全体が見える距離

    best_frames = []
    best_reward = -np.inf

    for ep in range(args.episodes):
        obs = vec_env.reset()
        frames = []
        total_reward = 0.0

        for _ in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += float(reward[0])

            frame = render_frame(env_inner, renderer, camera)
            frames.append(frame)

            if done[0]:
                break

        print(f"Episode {ep+1}: 報酬={total_reward:.1f}, steps={len(frames)}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_frames = frames

    renderer.close()
    vec_env.close()

    print(f"\n最高報酬: {best_reward:.1f} ({len(best_frames)} frames)")
    print("動画を保存中...")

    try:
        import imageio
    except ImportError:
        print("imageio をインストールしてください: pip install imageio[ffmpeg]")
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=args.fps) as writer:
        for frame in best_frames:
            writer.append_data(frame)

    print(f"保存完了: {out_path}")


if __name__ == "__main__":
    main()
