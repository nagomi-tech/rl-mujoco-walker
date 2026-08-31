"""
学習済みモデルをMuJoCoインタラクティブビューアで表示する。

マウス操作:
  左ドラッグ  : 視点回転
  右ドラッグ  : パン
  スクロール  : ズーム
  Spaceキー   : 一時停止
  Escキー     : 終了

使い方:
    python view.py                                      # デフォルト（交互歩行モデル）
    python view.py --model models/level0/best/best_model
    python view.py --model models/slope5deg/best/best_model --slope 5
    python view.py --speed 0.5                          # スロー再生
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.walker_env import BlockyWalkerEnv

MODELS_DIR = Path("models")


def find_vecnorm(model_path: Path, slope_deg: float, stairs: bool = False,
                 vision: bool = False, bumpy: bool = False) -> Path | None:
    candidates = []
    # チェックポイント専用vecnormを最優先
    candidates += [
        model_path.parent / "vecnorm.pkl",
        model_path.parent / "best_vecnormalize.pkl",
        model_path.parent / "vecnormalize.pkl",
    ]
    if bumpy:
        candidates += [
            MODELS_DIR / "bumpy" / "best" / "best_vecnormalize.pkl",
            MODELS_DIR / "bumpy" / "vecnorm.pkl",
        ]
    if vision:
        candidates += [
            MODELS_DIR / "vision_flat" / "best" / "best_vecnormalize.pkl",
            MODELS_DIR / "vision_flat" / "vecnorm.pkl",
        ]
    if stairs:
        candidates += [
            MODELS_DIR / "stairs" / "best" / "best_vecnormalize.pkl",
            MODELS_DIR / "stairs" / "vecnorm.pkl",
        ]
    slope_label = f"slope{int(slope_deg)}deg" if slope_deg > 0 else None
    if slope_label:
        candidates += [
            MODELS_DIR / slope_label / "best" / "best_vecnormalize.pkl",
            MODELS_DIR / slope_label / "vecnorm.pkl",
        ]
    candidates += [
        MODELS_DIR / "level0" / "best" / "best_vecnormalize.pkl",
        MODELS_DIR / "level0_vecnorm.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="models/slope0deg/best/best_model",
                        help="モデルパス（.zipなし）")
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--slope", type=float, default=0.0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--stairs", action="store_true", help="階段環境で表示")
    parser.add_argument("--vision", action="store_true", help="地形視覚観測ありモデルで表示")
    parser.add_argument("--bumpy", action="store_true", help="凹凸地形モデルで表示")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="再生速度倍率（0.5=スロー、2.0=高速）")
    args = parser.parse_args()

    default_model = "models/slope0deg/best/best_model"
    if args.bumpy and args.model == default_model:
        args.model = "models/bumpy/best/best_model"
    elif args.stairs and args.model == default_model:
        args.model = "models/stairs/best/best_model"
    elif args.vision and args.model == default_model:
        args.model = "models/vision_flat/best/best_model"

    model_path = Path(args.model)
    if not model_path.with_suffix(".zip").exists():
        print(f"モデルが見つかりません: {model_path}.zip")
        sys.exit(1)

    vn_path = Path(args.vecnorm) if args.vecnorm else find_vecnorm(
        model_path, args.slope, args.stairs, args.vision, args.bumpy)

    # render_mode=None で環境作成（ビューアは手動で開く）
    env = DummyVecEnv([lambda: BlockyWalkerEnv(
        level=args.level,
        slope_deg=args.slope,
        stairs=args.stairs,
        terrain_vision=args.vision,
        bumpy=args.bumpy,
        render_mode=None,
    )])

    if vn_path and vn_path.exists():
        print(f"VecNormalize読み込み: {vn_path}")
        env = VecNormalize.load(str(vn_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print("VecNormalizeなし")

    model = PPO.load(str(model_path), env=env)
    print(f"モデル読み込み: {model_path}.zip")

    # 内部のMuJoCoモデル・データに直接アクセス
    base_env = env.envs[0]
    mj_model = base_env.model
    mj_data  = base_env.data

    dt = mj_model.opt.timestep  # シミュレーションのタイムステップ
    render_interval = dt / args.speed  # 実時間での1ステップの長さ

    print("--- MuJoCoビューアを開いています ---")
    print("左ドラッグ=回転 / 右ドラッグ=パン / スクロール=ズーム / Space=一時停止 / Esc=終了")

    paused = False

    def key_callback(keycode):
        nonlocal paused
        if keycode == 32:  # Space
            paused = not paused
            print("一時停止" if paused else "再開")

    ep = 0
    done_arr = [False]
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        obs = env.reset()
        total_reward = 0.0
        step_count = 0

        while viewer.is_running():
            t0 = time.perf_counter()

            if not paused:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, info = env.step(action)
                total_reward += float(reward[0])
                step_count += 1

            viewer.sync()

            # 速度調整のためのスリープ
            elapsed = time.perf_counter() - t0
            sleep_time = render_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if not paused and done_arr[0]:
                ep += 1
                print(f"Episode {ep}: 報酬={total_reward:.1f}, steps={step_count}")
                total_reward = 0.0
                step_count = 0
                obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
