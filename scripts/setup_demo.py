"""Download the demo files and pre-load the pretrained models."""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.demo_setup import download_demo_files, warmup_models

def main():
    """Run the demo setup."""
    print("Downloading demo files...")
    downloaded = download_demo_files()
    if downloaded:
        for p in downloaded:
            print(f"Downloaded: {p}")
    else:
        print("Demo files already exist.")

    print("Warming up pretrained models (cache download)...")
    logs = warmup_models()
    for line in logs:
        print(line)
    print("Done.")

if __name__ == "__main__":
    main()
