"""Save a few labelled audio clips and chest X-rays from Hugging Face datasets as demo files."""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse
import json

from datasets import load_dataset, Audio, Image


def _label_name(label, label_names, fallback_map):
    if isinstance(label, str):
        key = label.lower()
        return fallback_map.get(key, key)
    if label_names:
        key = str(label_names[label]).lower()
        return fallback_map.get(key, key)
    return str(label)


def _write_bytes(path: Path, data: bytes, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.write_bytes(data)
    return True


def download_audio(dest: Path, per_label: int, force: bool):
    """Stream the audio dataset and save a fixed number of clips per label."""
    name = "DynamicSuperb/SpoofDetection_ASVspoof2017"
    ds = load_dataset(name, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    label_names = getattr(ds.features.get("label"), "names", None)
    map_audio = {
        "spoofed": "faux",
        "spoof": "faux",
        "authentic": "reel",
        "bonafide": "reel",
        "genuine": "reel",
    }
    allowed = {"reel", "faux"}
    counts = defaultdict(int)
    saved = []

    for ex in ds:
        label = _label_name(ex.get("label"), label_names, map_audio)
        if label not in allowed:
            continue
        if counts[label] >= per_label:
            continue
        audio = ex.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            continue
        out = dest / f"audio_{label}_{counts[label]+1:02d}.wav"
        if _write_bytes(out, data, force):
            saved.append(out)
        counts[label] += 1
        if all(counts[l] >= per_label for l in allowed):
            break

    return {
        "dataset": name,
        "saved": [p.name for p in saved],
        "counts": dict(counts),
    }


def download_images(dest: Path, per_label: int, force: bool):
    """Stream the X-ray dataset and save a fixed number of images per label."""
    name = "hf-vision/chest-xray-pneumonia"
    ds = load_dataset(name, split="train", streaming=True)
    ds = ds.cast_column("image", Image(decode=False))

    label_names = getattr(ds.features.get("label"), "names", None)
    map_img = {
        "normal": "normal",
        "pneumonia": "pneumonie",
    }
    allowed = {"normal", "pneumonie"}
    counts = defaultdict(int)
    saved = []

    for ex in ds:
        label = _label_name(ex.get("label"), label_names, map_img)
        if label not in allowed:
            continue
        if counts[label] >= per_label:
            continue
        img = ex.get("image") or {}
        data = img.get("bytes")
        if not data:
            continue
        out = dest / f"xray_{label}_{counts[label]+1:02d}.jpg"
        if _write_bytes(out, data, force):
            saved.append(out)
        counts[label] += 1
        if all(counts[l] >= per_label for l in allowed):
            break

    return {
        "dataset": name,
        "saved": [p.name for p in saved],
        "counts": dict(counts),
    }


def main():
    """Download both demo sets and write a manifest."""
    parser = argparse.ArgumentParser(description="Download a pro demo pack (audio + xray) from HF datasets.")
    parser.add_argument("--out", default="data/demo", help="Output folder (default: data/demo)")
    parser.add_argument("--per-label", type=int, default=4, help="Samples per label (default: 4)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(dest),
        "per_label": args.per_label,
        "audio": download_audio(dest, args.per_label, args.force),
        "xray": download_images(dest, args.per_label, args.force),
    }

    manifest = dest / "demo_pro_manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Pro demo pack ready.")
    print(f"Audio: {report['audio']['counts']}")
    print(f"Xray: {report['xray']['counts']}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
