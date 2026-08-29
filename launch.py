"""
学習ランチャー。ローカルまたはGCPクラウドで学習を実行する。

使い方:
    # ローカル実行
    python launch.py --slope 0 --steps 3000000 --from-scratch

    # GCPクラウド実行（自動でVMを作成・学習・結果取得・停止）
    python launch.py --slope 0 --steps 3000000 --from-scratch --target cloud

    # クラウド実行時のオプション
    python launch.py --slope 0 --steps 9000000 --from-scratch --target cloud \
        --gcp-project my-project --gcp-zone asia-northeast1-a --n-envs 16

事前準備（クラウド実行時）:
    1. gcloud CLI をインストール: https://cloud.google.com/sdk/docs/install
    2. gcloud auth login
    3. gcloud config set project YOUR_PROJECT_ID
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ---- GCP デフォルト設定 ----
GCP_INSTANCE   = "rl-trainer"
GCP_ZONE       = "us-central1-a"
GCP_MACHINE    = "c4-standard-16"
GCP_DISK_SIZE  = "50GB"
GCP_IMAGE_FAM  = "ubuntu-2204-lts"
GCP_IMAGE_PROJ = "ubuntu-os-cloud"

# クラウドに転送するファイル・ディレクトリ
UPLOAD_FILES = [
    "train_slope.py",
    "requirements.txt",
    "envs/",
]


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def gcloud(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(f"gcloud {cmd}", check=check)


# ------------------------------------------------------------------ #
#  ローカル実行                                                         #
# ------------------------------------------------------------------ #
def run_local(train_args: list[str]):
    cmd = [sys.executable, "train_slope.py"] + train_args
    print(f"\n=== ローカル実行: {' '.join(cmd)} ===\n")
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------ #
#  GCP クラウド実行                                                     #
# ------------------------------------------------------------------ #
def create_instance(zone: str, machine: str, project: str | None):
    proj = f"--project {project}" if project else ""
    print(f"\n=== GCP VM を作成中: {GCP_INSTANCE} ({machine}) ===")
    gcloud(
        f"compute instances create {GCP_INSTANCE} "
        f"--zone={zone} "
        f"--machine-type={machine} "
        f"--provisioning-model=SPOT "
        f"--instance-termination-action=STOP "
        f"--image-family={GCP_IMAGE_FAM} "
        f"--image-project={GCP_IMAGE_PROJ} "
        f"--boot-disk-size={GCP_DISK_SIZE} "
        f"--boot-disk-type=pd-ssd "
        f"{proj}"
    )
    print("VM 起動待ち (30秒)...")
    time.sleep(30)


def upload_code(zone: str, project: str | None, include_models: bool = False):
    proj = f"--project {project}" if project else ""
    print("\n=== コードを転送中 ===")
    gcloud(f'compute ssh {GCP_INSTANCE} --zone={zone} {proj} --command="mkdir -p ~/rl/envs"')
    for f in UPLOAD_FILES:
        if f.endswith("/"):
            gcloud(f"compute scp --recurse {f} {GCP_INSTANCE}:~/rl/ --zone={zone} {proj}")
        else:
            gcloud(f"compute scp {f} {GCP_INSTANCE}:~/rl/{f} --zone={zone} {proj}")
    if include_models and Path("models").exists():
        print("=== モデルファイルを転送中（再開用）===")
        gcloud(f"compute scp --recurse models/ {GCP_INSTANCE}:~/rl/ --zone={zone} {proj}")


def setup_env(zone: str, project: str | None):
    proj = f"--project {project}" if project else ""
    print("\n=== VM 環境セットアップ中 ===")
    setup_cmd = (
        "sudo apt-get update -q && "
        "sudo apt-get install -y -q python3-pip python3-venv libosmesa6-dev && "
        "cd ~/rl && "
        "python3 -m venv venv && "
        "source venv/bin/activate && "
        "pip install --quiet -r requirements.txt"
    )
    gcloud(f'compute ssh {GCP_INSTANCE} --zone={zone} {proj} --command="{setup_cmd}"')


def run_training_on_vm(zone: str, project: str | None, train_args: list[str]):
    proj = f"--project {project}" if project else ""
    args_str = " ".join(train_args)
    print(f"\n=== クラウドで学習開始: {args_str} ===")

    train_cmd = (
        "cd ~/rl && "
        "source venv/bin/activate && "
        f"MUJOCO_GL=osmesa python train_slope.py {args_str} --no-video "
        "> ~/train.log 2>&1"
    )
    # バックグラウンドで実行（SSH切断後も継続）
    gcloud(
        f'compute ssh {GCP_INSTANCE} --zone={zone} {proj} '
        f'--command="nohup bash -c \'{train_cmd}\' &"'
    )
    print("\n学習開始しました。進捗確認:")
    print(f"  gcloud compute ssh {GCP_INSTANCE} --zone={zone} --command='tail -f ~/train.log'")
    print("\n完了後の結果取得:")
    print(f"  python launch.py --fetch --target cloud --gcp-zone {zone}")


def wait_and_fetch(zone: str, project: str | None):
    proj = f"--project {project}" if project else ""
    print("\n=== 学習完了を待機中 (Ctrl+C で中断して後で --fetch できます) ===")
    try:
        while True:
            result = subprocess.run(
                f"gcloud compute ssh {GCP_INSTANCE} --zone={zone} {proj} "
                f"--command='pgrep -f train_slope.py > /dev/null && echo running || echo done'",
                shell=True, capture_output=True, text=True
            )
            status = result.stdout.strip()
            print(f"  状態: {status} ({time.strftime('%H:%M:%S')})")
            if status == "done":
                break
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n監視を中断しました。後で --fetch で結果を取得できます。")
        return

    fetch_results(zone, project)


def fetch_results(zone: str, project: str | None):
    proj = f"--project {project}" if project else ""
    print("\n=== 結果をダウンロード中 ===")
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    gcloud(f"compute scp --recurse {GCP_INSTANCE}:~/rl/models/ . --zone={zone} {proj}")
    # ログも取得
    gcloud(f"compute scp {GCP_INSTANCE}:~/train.log ./cloud_train.log --zone={zone} {proj}")
    print("  => models/ にダウンロード完了")
    print("  => cloud_train.log にログ保存")


def stop_instance(zone: str, project: str | None):
    proj = f"--project {project}" if project else ""
    print(f"\n=== VM を停止中: {GCP_INSTANCE} ===")
    gcloud(f"compute instances stop {GCP_INSTANCE} --zone={zone} {proj}", check=False)


def run_cloud(train_args: list[str], zone: str, machine: str,
              project: str | None, fetch_only: bool, resume: bool = False):
    if fetch_only:
        fetch_results(zone, project)
        return

    create_instance(zone, machine, project)
    upload_code(zone, project, include_models=resume)
    setup_env(zone, project)
    run_training_on_vm(zone, project, train_args)
    wait_and_fetch(zone, project)
    stop_instance(zone, project)


# ------------------------------------------------------------------ #
#  エントリポイント                                                      #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="学習ランチャー（ローカル / GCPクラウド）"
    )
    parser.add_argument("--target", choices=["local", "cloud"], default="local")
    parser.add_argument("--fetch", action="store_true",
                        help="クラウドから結果だけ取得する（学習済み）")

    # GCP 設定
    parser.add_argument("--gcp-zone",    default=GCP_ZONE)
    parser.add_argument("--gcp-machine", default=GCP_MACHINE)
    parser.add_argument("--gcp-project", default=None, help="GCPプロジェクトID")

    # train_slope.py に渡す引数（残りはすべて転送）
    parser.add_argument("--slope",       type=float, default=0.0)
    parser.add_argument("--steps",       type=int,   default=3_000_000)
    parser.add_argument("--n-envs",      type=int,   default=8)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--start-m",     type=int,   default=0)
    parser.add_argument("--base-model",  type=str,   default="models/level0/best/best_model")
    args = parser.parse_args()

    # train_slope.py に渡す引数リストを構築
    train_args = [
        f"--slope {args.slope}",
        f"--steps {args.steps}",
        f"--n-envs {args.n_envs}",
    ]
    if args.from_scratch:
        train_args.append("--from-scratch")
    if args.start_m > 0:
        train_args.append(f"--start-m {args.start_m}")

    if args.target == "local":
        run_local(train_args)
    else:
        run_cloud(
            train_args=train_args,
            zone=args.gcp_zone,
            machine=args.gcp_machine,
            project=args.gcp_project,
            fetch_only=args.fetch,
            resume=(args.start_m > 0),
        )


if __name__ == "__main__":
    main()
