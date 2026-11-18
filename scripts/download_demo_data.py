"""Download the public demo files listed in src/utils/demo_setup.py."""

from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.demo_setup import download_demo_files

def main():
    """Download the demo files into data/demo."""
    dest = Path("data/demo")
    files = download_demo_files(dest=dest, force=False)
    if not files:
        print("Demo files already exist.")
    else:
        for f in files:
            print(f"Downloaded: {f}")
    print("Done. Demo files in data/demo/")

if __name__ == "__main__":
    main()
