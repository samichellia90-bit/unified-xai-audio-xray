"""Prepare the datasets, then train every audio and image model, logging to a run folder."""

from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd, log_file):
    """Run a command and append its output to the log file."""
    msg = f"$ {' '.join(str(c) for c in cmd)}"
    log_file.write(msg + "\n")
    log_file.flush()
    subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file)


def main():
    """Run the data preparation and the training commands in sequence."""
    ap = argparse.ArgumentParser(description="Full pipeline: download full datasets, train all models, start Streamlit.")
    ap.add_argument("--max-per-label", type=int, default=0, help="0 = full dataset (default)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--audio-archs", type=str, default="resnet18,vgg16,mobilenet_v3_small")
    ap.add_argument("--image-archs", type=str, default="densenet121,alexnet")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-streamlit", action="store_true")
    ap.add_argument("--image-first", action="store_true", help="Train image models before audio")
    ap.add_argument("--run_root", type=str, default=None, help="Base folder for logs (default: runs/pipeline_YYYYMMDD_HHMMSS)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root) if args.run_root else Path("runs") / f"pipeline_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "pipeline.log"

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"Pipeline start: {ts}\n")
        logf.flush()

        if not args.skip_download:
            run_cmd(
                [sys.executable, "scripts/prepare_real_datasets.py", "--max-per-label", str(args.max_per_label)],
                logf,
            )

        if not args.skip_train:
            audio_archs = [a.strip() for a in args.audio_archs.split(",") if a.strip()]
            image_archs = [a.strip() for a in args.image_archs.split(",") if a.strip()]

            def run_audio(arch: str):
                run_dir = run_root / f"audio_{arch}"
                run_cmd(
                    [
                        sys.executable, "-m", "src.train.train_audio_spectro",
                        "--data_dir", "data/audio",
                        "--arch", arch,
                        "--epochs", str(args.epochs),
                        "--run_dir", str(run_dir),
                    ],
                    logf,
                )

            def run_image(arch: str):
                run_dir = run_root / f"image_{arch}"
                run_cmd(
                    [
                        sys.executable, "-m", "src.train.train_xray",
                        "--data_dir", "data/xray",
                        "--arch", arch,
                        "--epochs", str(args.epochs),
                        "--run_dir", str(run_dir),
                    ],
                    logf,
                )

            if args.image_first:
                for arch in image_archs:
                    run_image(arch)
                for arch in audio_archs:
                    run_audio(arch)
            else:
                for arch in audio_archs:
                    run_audio(arch)
                for arch in image_archs:
                    run_image(arch)

        if not args.no_streamlit:
            logf.write("Starting Streamlit...\n")
            logf.flush()
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py"],
                stdout=logf,
                stderr=logf,
            )

        logf.write("Pipeline done.\n")
        logf.flush()


if __name__ == "__main__":
    main()
