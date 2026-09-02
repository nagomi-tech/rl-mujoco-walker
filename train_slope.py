"""
坂道歩行の学習スクリプト。
平地6Mモデルをベースに、傾斜 slope_deg の坂道を上る歩行を学習する。

使い方:
    python train_slope.py                      # 5° 坂道、6M ステップ
    python train_slope.py --slope 8 --steps 9000000
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs.walker_env import BlockyWalkerEnv

MODELS_DIR  = Path("models")
LOGS_DIR    = Path("logs")
RESULTS_DIR = Path("results")
N_ENVS = 8
CHUNK_STEPS = 3_000_000
MILESTONE_STEPS = 3_000_000  # ckpt_XXm を保存する間隔


class MilestoneCallback(BaseCallback):
    """3Mステップごとに ckpt_XXm を保存するコールバック。"""

    def __init__(self, slope_dir: Path, label: str, train_env: VecNormalize,
                 milestone_steps: int = MILESTONE_STEPS, verbose: int = 0):
        super().__init__(verbose)
        self.slope_dir = slope_dir
        self.label = label
        self.train_env = train_env
        self.milestone_steps = milestone_steps
        self.next_milestone = milestone_steps  # _on_training_start で調整される

    def _on_training_start(self) -> None:
        # 再開時: 現在のステップ数を超える最初のマイルストーンから開始
        while self.next_milestone <= self.model.num_timesteps:
            self.next_milestone += self.milestone_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_milestone:
            total_m = self.next_milestone // 1_000_000
            ckpt_dir = self.slope_dir / f"ckpt_{total_m:02d}m"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # best モデルをコピー
            best_src = self.slope_dir / "best"
            for fname in ["best_model.zip", "best_vecnormalize.pkl"]:
                src = best_src / fname
                if src.exists():
                    shutil.copy2(src, ckpt_dir / fname)

            # 現在の VecNormalize を保存
            vecnorm_path = ckpt_dir / "vecnorm.pkl"
            self.train_env.save(str(vecnorm_path))

            print(f"\n--- チェックポイント {total_m}M 保存: {ckpt_dir} ---")
            self.next_milestone += self.milestone_steps
        return True


def make_slope_env(slope_deg: float, n_envs: int = N_ENVS, render: bool = False,
                   stairs: bool = False, terrain_vision: bool = False,
                   bumpy: bool = False, combined: bool = False):
    def _make():
        return BlockyWalkerEnv(level=0,
                               slope_deg=slope_deg,
                               stairs=stairs,
                               terrain_vision=terrain_vision,
                               bumpy=bumpy,
                               combined=combined,
                               render_mode="rgb_array" if render else None)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(_make, n_envs=n_envs, vec_env_cls=vec_cls)


def evaluate_and_save_video(label: str, model_path: Path, vecnorm_path: Path | None,
                             slope_deg: float, n_episodes: int = 3, max_steps: int = 1500):
    import imageio

    video_path = RESULTS_DIR / f"slope{int(slope_deg)}deg_{label}.mp4"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== 動画生成: {video_path} ===")

    env = DummyVecEnv([lambda: BlockyWalkerEnv(level=0, slope_deg=slope_deg,
                                                render_mode="rgb_array")])
    if vecnorm_path and vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    best_frames, best_reward = [], -np.inf
    for ep in range(n_episodes):
        obs = env.reset()
        frames, total_reward = [], 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            frame = env.envs[0].render()
            if frame is not None:
                frames.append(frame)
            if done[0]:
                break
        print(f"  Episode {ep+1}: 報酬={total_reward:.1f}, steps={len(frames)}")
        if total_reward > best_reward:
            best_reward, best_frames = total_reward, frames

    env.close()
    if best_frames:
        with imageio.get_writer(str(video_path), fps=30) as writer:
            for f in best_frames:
                writer.append_data(f)
        print(f"  => 保存: {video_path}")


def train_chunk(chunk_idx: int, slope_deg: float,
                resume_model_path: Path | None, vecnorm_path: Path | None,
                n_envs: int = N_ENVS, stairs: bool = False,
                terrain_vision: bool = False, bumpy: bool = False,
                lr: float = 3e-4, clip_range: float = 0.2) -> tuple[Path, Path]:
    if bumpy:
        label = "bumpy"
    elif terrain_vision:
        label = "vision_flat"
    elif stairs:
        label = "stairs"
    else:
        label = f"slope{int(slope_deg)}deg"
    slope_dir = MODELS_DIR / label
    slope_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    env_desc = "凹凸地形" if bumpy else ("視覚フラット" if terrain_vision else ("階段" if stairs else f"坂道 {slope_deg}°"))
    print(f"  {env_desc} | Chunk {chunk_idx}: "
          f"{(chunk_idx-1)*3}M → {chunk_idx*3}M ステップ")
    print(f"  再開モデル: {resume_model_path if resume_model_path else 'ランダム初期化'}")
    print(f"  並列環境数: {n_envs}")
    print(f"{'='*60}")

    train_venv = make_slope_env(slope_deg, n_envs=n_envs, stairs=stairs,
                                terrain_vision=terrain_vision, bumpy=bumpy)
    if vecnorm_path and vecnorm_path.exists():
        train_env = VecNormalize.load(str(vecnorm_path), train_venv)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_venv = make_slope_env(slope_deg, n_envs=1, stairs=stairs,
                               terrain_vision=terrain_vision, bumpy=bumpy)
    if vecnorm_path and vecnorm_path.exists():
        eval_env = VecNormalize.load(str(vecnorm_path), eval_venv)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    save_freq = max(1, 50_000 // n_envs)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(slope_dir),
        name_prefix="ppo_slope",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(slope_dir / "best"),
        log_path=str(LOGS_DIR / label),
        eval_freq=save_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    if resume_model_path and (resume_model_path.with_suffix(".zip")).exists():
        model = PPO.load(
            str(resume_model_path),
            env=train_env,
            tensorboard_log=str(LOGS_DIR / label),
            learning_rate=lr,
            clip_range=clip_range,
        )
    else:
        # ゼロから学習
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1,
            tensorboard_log=str(LOGS_DIR / label),
        )

    model.learn(
        total_timesteps=CHUNK_STEPS,
        callback=[ckpt_cb, eval_cb],
        reset_num_timesteps=True,
        progress_bar=True,
    )

    final_path = slope_dir / "final"
    new_vecnorm = slope_dir / "vecnorm.pkl"
    model.save(str(final_path))
    train_env.save(str(new_vecnorm))

    eval_env.close()
    train_env.close()
    print(f"\n  Chunk {chunk_idx} 完了: {final_path}.zip")
    return final_path, new_vecnorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope", type=float, default=5.0, help="坂道角度（度）")
    parser.add_argument("--steps", type=int, default=9_000_000, help="今回追加する学習ステップ数")
    parser.add_argument("--base-model", type=str,
                        default="models/level0/best/best_model",
                        help="転移元モデルパス（.zip なし）")
    parser.add_argument("--base-vecnorm", type=str,
                        default="models/level0_vecnorm.pkl",
                        help="転移元VecNormalizeパス")
    parser.add_argument("--start-m", type=int, default=0,
                        help="再開時の累積ステップ数（M単位）。例: 9 なら9M済みから再開")
    parser.add_argument("--from-scratch", action="store_true",
                        help="既存モデルを無視してゼロから学習する")
    parser.add_argument("--n-envs", type=int, default=N_ENVS,
                        help=f"並列環境数（デフォルト: {N_ENVS}）")
    parser.add_argument("--no-video", action="store_true",
                        help="動画生成をスキップ（クラウド実行時など）")
    parser.add_argument("--stairs", action="store_true",
                        help="階段環境で学習する")
    parser.add_argument("--vision", action="store_true",
                        help="地形視覚観測を追加する（OBS +10次元）")
    parser.add_argument("--bumpy", action="store_true",
                        help="凹凸地形（heightfield）で学習する")
    parser.add_argument("--combined", action="store_true",
                        help="合成地形（平坦→バンプ→階段、ゴール30m）で学習する")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学習率（デフォルト: 1e-4）")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPOクリップ範囲（デフォルト: 0.2）")
    args = parser.parse_args()

    slope_deg = args.slope
    if args.combined:
        args.vision = True  # combined時は視覚情報を必ず有効化
        label = "combined"
    elif args.bumpy:
        args.vision = True  # bumpy時は視覚情報を必ず有効化
        label = "bumpy"
    elif args.vision:
        label = "vision_flat"
    elif args.stairs:
        label = "stairs"
    else:
        label = f"slope{int(slope_deg)}deg"

    total_steps = args.steps
    slope_dir = MODELS_DIR / label
    slope_dir.mkdir(parents=True, exist_ok=True)

    # 再開モデルの自動検出
    resume_model  = slope_dir / "best" / "best_model"
    resume_vecnorm = slope_dir / "vecnorm.pkl"
    resuming = (not args.from_scratch
                and resume_model.with_suffix(".zip").exists()
                and resume_vecnorm.exists())

    if resuming:
        print(f"=== 継続学習: {label} / +{total_steps//1_000_000}M ステップ ===")
        print(f"  再開モデル: {resume_model}.zip")
    else:
        print(f"=== 新規学習: {label} / {total_steps//1_000_000}M ステップ ===")
    print(f"  並列環境数: {args.n_envs}")

    combined = getattr(args, 'combined', False)
    train_venv = make_slope_env(slope_deg, n_envs=args.n_envs, stairs=args.stairs,
                                terrain_vision=args.vision, bumpy=args.bumpy,
                                combined=combined)
    if resuming:
        train_env = VecNormalize.load(str(resume_vecnorm), train_venv)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(train_venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_venv = make_slope_env(slope_deg, n_envs=1, stairs=args.stairs,
                               terrain_vision=args.vision, bumpy=args.bumpy,
                               combined=combined)
    if resuming:
        eval_env = VecNormalize.load(str(resume_vecnorm), eval_venv)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        eval_env = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    save_freq = max(1, 50_000 // args.n_envs)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(slope_dir),
        name_prefix="ppo_slope",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(slope_dir / "best"),
        log_path=str(LOGS_DIR / label),
        eval_freq=save_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    milestone_cb = MilestoneCallback(slope_dir, label, train_env)

    if resuming:
        model = PPO.load(
            str(resume_model),
            env=train_env,
            learning_rate=args.lr,
            clip_range=args.clip_range,
            target_kl=0.02,
            tensorboard_log=str(LOGS_DIR / label),
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            target_kl=0.02,
            ent_coef=0.005,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=1,
            tensorboard_log=str(LOGS_DIR / label),
        )

    model.learn(
        total_timesteps=total_steps,
        callback=[ckpt_cb, eval_cb, milestone_cb],
        reset_num_timesteps=not resuming,
        progress_bar=True,
    )

    final_path = slope_dir / "final"
    model.save(str(final_path))
    train_env.save(str(slope_dir / "vecnorm.pkl"))
    eval_env.close()
    train_env.close()

    print(f"\n=== 学習完了: {label} / {total_steps//1_000_000}M ステップ ===")


if __name__ == "__main__":
    main()
