import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    ret = subprocess.call(cmd)
    if ret != 0:
        print("Command failed:", " ".join(cmd))
        sys.exit(ret)


def main():
    parser = argparse.ArgumentParser(
        description="Run full Deepfake Benchmark (hfreq + DIRE)"
    )

    parser.add_argument(
        "--jpeg",
        action="store_true",
        help="Run JPEG-90 evaluation only (NO training / NO calibration)",
    )

    parser.add_argument(
        "--dire_ckpt",
        required=True,
        help="Path to DIRE checkpoint (.pth)",
    )
    parser.add_argument("--dire_arch", default="resnet50")
    parser.add_argument("--dire_cpu", action="store_true")

    args = parser.parse_args()

    # Ensure project root is cwd
    project_root = Path(__file__).resolve().parents[1]
    os_env = dict(**dict(**__import__("os").environ))
    os_env["PYTHONPATH"] = str(project_root)

    # ------------------------------------------------------------
    # hfreq
    # ------------------------------------------------------------
    hfreq_cmd = [
        sys.executable,
        "-m",
        "runners.run_hfreq",
    ]
    if args.jpeg:
        hfreq_cmd.append("--jpeg")

    run(hfreq_cmd)

    # ------------------------------------------------------------
    # DIRE
    # ------------------------------------------------------------
    dire_cmd = [
        sys.executable,
        "-m",
        "runners.run_dire",
        "--model_path",
        args.dire_ckpt,
        "--arch",
        args.dire_arch,
    ]
    if args.dire_cpu:
        dire_cmd.append("--use_cpu")
    if args.jpeg:
        dire_cmd.append("--jpeg")

    run(dire_cmd)

    print("\n FULL PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
