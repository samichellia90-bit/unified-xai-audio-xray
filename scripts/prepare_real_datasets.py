"""Build the training folders (files and labels.csv) from Hugging Face datasets."""

from __future__ import annotations
from collections import defaultdict
import itertools
from pathlib import Path
import argparse
import csv

from datasets import load_dataset, Audio, Image


def _label_to_int(label, label_names, mapping):
    if isinstance(label, str):
        key = label.lower()
    else:
        key = str(label_names[label]).lower() if label_names else str(label).lower()
    return mapping.get(key)


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        used.add(name)
        return name
    stem = Path(name).stem
    suffix = Path(name).suffix
    i = 2
    while True:
        cand = f"{stem}_{i}{suffix}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1


def _parse_splits(split: str):
    if split.lower() == "all":
        return ["train", "validation", "test"]
    raw = split.replace("+", ",")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _load_streaming_splits(name: str, split: str):
    splits = _parse_splits(split)
    datasets = []
    errors = []
    for sp in splits:
        try:
            ds = load_dataset(name, split=sp, streaming=True)
            datasets.append(ds)
        except Exception as e:
            errors.append(f"{sp}: {e}")
    if not datasets:
        raise ValueError(f"No valid splits for {name} (requested: {split}). Errors: {errors}")
    return datasets, splits


def prepare_audio(out_dir: Path, max_per_label: int, force: bool, split: str) -> dict:
    """Save the audio clips and their labels."""
    name = "Bisher/ASVspoof_2019_LA"
    dsets, splits = _load_streaming_splits(name, split)
    dsets = [ds.cast_column("audio", Audio(decode=False)) for ds in dsets]

    label_names = dsets[0].features["key"].names
    mapping = {"bonafide": 0, "spoof": 1}

    wav_root = out_dir / "wav"
    wav_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"

    counts = defaultdict(int)
    used = set()
    rows = []

    for ex in itertools.chain.from_iterable(dsets):
        label_int = _label_to_int(ex.get("key"), label_names, mapping)
        if label_int is None:
            continue
        if max_per_label > 0 and counts[label_int] >= max_per_label:
            continue
        audio = ex.get("audio") or {}
        data = audio.get("bytes")
        if not data:
            continue
        fname = ex.get("audio_file_name") or (audio.get("path") or f"audio_{len(rows)+1}.wav")
        fname = _unique_name(Path(fname).name, used)
        out = wav_root / fname
        if not out.exists() or force:
            out.write_bytes(data)
        rows.append((fname, label_int))
        counts[label_int] += 1
        if max_per_label > 0 and all(counts[i] >= max_per_label for i in mapping.values()):
            break

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for r in rows:
            w.writerow(r)

    return {
        "dataset": name,
        "split": split,
        "resolved_splits": splits,
        "counts": dict(counts),
        "labels_csv": str(csv_path),
        "wav_root": str(wav_root),
    }


def prepare_images(out_dir: Path, max_per_label: int, force: bool, split: str) -> dict:
    """Save the X-ray images and their labels."""
    name = "hf-vision/chest-xray-pneumonia"
    dsets, splits = _load_streaming_splits(name, split)
    dsets = [ds.cast_column("image", Image(decode=False)) for ds in dsets]

    label_names = dsets[0].features["label"].names
    mapping = {"normal": 0, "pneumonia": 1}

    img_root = out_dir / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "labels.csv"

    counts = defaultdict(int)
    used = set()
    rows = []

    for ex in itertools.chain.from_iterable(dsets):
        label_int = _label_to_int(ex.get("label"), label_names, mapping)
        if label_int is None:
            continue
        if max_per_label > 0 and counts[label_int] >= max_per_label:
            continue
        img = ex.get("image") or {}
        data = img.get("bytes")
        if not data:
            continue
        fname = img.get("path") or f"image_{len(rows)+1}.jpg"
        fname = _unique_name(Path(fname).name, used)
        out = img_root / fname
        if not out.exists() or force:
            out.write_bytes(data)
        rows.append((fname, label_int))
        counts[label_int] += 1
        if max_per_label > 0 and all(counts[i] >= max_per_label for i in mapping.values()):
            break

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label"])
        for r in rows:
            w.writerow(r)

    return {
        "dataset": name,
        "split": split,
        "resolved_splits": splits,
        "counts": dict(counts),
        "labels_csv": str(csv_path),
        "img_root": str(img_root),
    }


def main():
    """Prepare the audio and X-ray training sets."""
    ap = argparse.ArgumentParser(description="Prepare real datasets for training (audio + xray).")
    ap.add_argument("--max-per-label", type=int, default=2000, help="0 = no limit (full dataset)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    ap.add_argument("--audio-split", type=str, default="train+validation")
    ap.add_argument("--image-split", type=str, default="train")
    ap.add_argument("--audio-out", type=str, default="data/audio")
    ap.add_argument("--image-out", type=str, default="data/xray")
    ap.add_argument("--only", choices=["audio", "images"], default=None, help="Prepare one modality only")
    args = ap.parse_args()

    if args.only != "images":
        print("Prepared audio:", prepare_audio(Path(args.audio_out), args.max_per_label, args.force, args.audio_split))
    if args.only != "audio":
        print("Prepared images:", prepare_images(Path(args.image_out), args.max_per_label, args.force, args.image_split))


if __name__ == "__main__":
    main()
