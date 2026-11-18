"""Wait for a running training job to finish an epoch, stop it, then start another training command."""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def count_val_acc(log_path: Path) -> int:
    """Count the validation lines written to the log so far."""
    if not log_path.exists():
        return 0
    try:
        return sum(1 for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if "val acc:" in line)
    except Exception:
        return 0


def stop_training_processes():
    """Stop the running training processes."""
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*train_audio_spectro*' -or $_.CommandLine -like '*full_pipeline.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-Command", cmd], check=False)


def main():
    """Wait, stop the current job and launch the next one."""
    ap = argparse.ArgumentParser(description="Stop current training after next epoch, then run short training (image first).")
    ap.add_argument("--log", type=str, required=True, help="Path to pipeline.log to watch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--audio-archs", type=str, default="vgg16,mobilenet_v3_small")
    ap.add_argument("--image-archs", type=str, default="densenet121,alexnet")
    ap.add_argument("--image-first", action="store_true", default=True)
    ap.add_argument("--run_root", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    log_path = (root / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    target = count_val_acc(log_path) + 1

    while True:
        time.sleep(20)
        if count_val_acc(log_path) >= target:
            break

    stop_training_processes()
    time.sleep(10)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) if args.run_root else root / "runs" / f"pipeline_short_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    out_log = run_root / "pipeline.log"

    cmd = [
        sys.executable, "scripts/full_pipeline.py",
        "--skip-download",
        "--epochs", str(args.epochs),
        "--audio-archs", args.audio_archs,
        "--image-archs", args.image_archs,
        "--no-streamlit",
        "--run_root", str(run_root),
    ]
    if args.image_first:
        cmd.append("--image-first")

    with out_log.open("a", encoding="utf-8") as f:
        f.write(f"Short pipeline start: {ts}\n")
        f.flush()
        subprocess.run(cmd, cwd=str(root), stdout=f, stderr=f, check=True)


if __name__ == "__main__":
    main()
