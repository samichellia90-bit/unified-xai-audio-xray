"""Quantitative comparison of the image explanation methods.

For each image and each method, the heatmap ranks the pixels. Two perturbation
curves follow that ranking (Petsiuk et al., 2018):

- deletion: the most important pixels are removed first; a faithful heatmap
  makes the predicted probability fall quickly (low area under the curve);
- insertion: the most important pixels are revealed first on a blurred image;
  a faithful heatmap makes the probability rise quickly (high area).

Two random heatmaps give the reference levels: independent pixels, and 16-pixel
blocks, which is the fair reference for methods that rank regions rather than
pixels. The border share measures how much of the top-ranked area lies in the
outer band of the image.

Usage:
    python -m src.eval.faithfulness --arch densenet121 --weights weights/image/densenet121_xray.pth \
        --data_dir data/xray_test --per_class 50 --out results/faithfulness/densenet121
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image

from ..config import DEFAULT_LIME_SAMPLES, DEFAULT_SHAP_EVALS
from ..train.common import Item, read_labels_csv
from ..xai import lime_image, occlusion_image, shap_image
from ..xai.gradcam_common import find_last_conv, gradcam_2d
from .evaluate_xray import load_model

METHODS = ("gradcam", "occlusion", "lime", "shap", "random", "random_blocks")
PredictFn = Callable[[np.ndarray], np.ndarray]


def make_predict_fn(model: torch.nn.Module, device: str, batch: int = 64) -> PredictFn:
    """Return a function mapping float RGB images in [0, 1] (N, H, W, 3) to class probabilities."""

    def predict(images: np.ndarray) -> np.ndarray:
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), batch):
                chunk = torch.from_numpy(np.ascontiguousarray(images[start : start + batch], dtype=np.float32))
                chunk = ((chunk - 0.5) / 0.5).permute(0, 3, 1, 2).to(device)
                out.append(torch.softmax(model(chunk), dim=-1).cpu().numpy())
        return np.concatenate(out)

    return predict


def perturbation_curve(
    image: np.ndarray, heat: np.ndarray, predict_fn: PredictFn, target: int, mode: str, steps: int = 20
) -> np.ndarray:
    """Probability of the target class while pixels are removed or revealed in heatmap order."""
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be 'deletion' or 'insertion'")
    height, width, _ = image.shape
    order = np.argsort(-heat.ravel(), kind="stable")
    if mode == "deletion":
        start, end = image, np.broadcast_to(image.mean(axis=(0, 1)), image.shape)
    else:
        start, end = cv2.GaussianBlur(image, (0, 0), sigmaX=11), image
    frames = np.repeat(start[np.newaxis], steps + 1, axis=0).reshape(steps + 1, height * width, 3)
    flat_end = np.asarray(end).reshape(height * width, 3)
    for step in range(1, steps + 1):
        changed = order[: round(step * height * width / steps)]
        frames[step, changed] = flat_end[changed]
    return predict_fn(frames.reshape(steps + 1, height, width, 3))[:, target]


def curve_auc(curve: np.ndarray) -> float:
    """Area under a curve sampled at equal steps over [0, 1]."""
    return float((curve[:-1] + curve[1:]).sum() / (2 * (len(curve) - 1)))


def border_share(heat: np.ndarray, top_fraction: float = 0.10, band: float = 0.10) -> float:
    """Share of the top-ranked pixels lying in the outer band of the image.

    With a band of 10 % on each side, a heatmap unrelated to position gives 0.36.
    """
    height, width = heat.shape
    top = np.argsort(-heat.ravel(), kind="stable")[: round(top_fraction * heat.size)]
    rows, cols = np.unravel_index(top, heat.shape)
    margin_r, margin_c = band * height, band * width
    outside = (rows < margin_r) | (rows >= height - margin_r) | (cols < margin_c) | (cols >= width - margin_c)
    return float(outside.mean())


def explain(method: str, model: torch.nn.Module, image: np.ndarray, predict_fn: PredictFn, target: int,
            device: str, rng: np.random.Generator) -> np.ndarray:
    """Return the heatmap (H, W) of one method for the target class."""
    if method == "gradcam":
        x = torch.from_numpy((image - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).float().to(device)
        return gradcam_2d(model, x, target, conv=find_last_conv(model, 2)).heat
    if method == "occlusion":
        return occlusion_image.explain(image, predict_fn, target, patch=32, stride=16, batch=64).heat
    if method == "lime":
        as_uint8 = (image * 255).astype(np.uint8)
        return lime_image.explain(as_uint8, lambda b: predict_fn(b.astype(np.float32) / 255.0), DEFAULT_LIME_SAMPLES).mask
    if method == "shap":
        return shap_image.explain(image, predict_fn, DEFAULT_SHAP_EVALS).heat
    if method == "random":
        return rng.random(image.shape[:2], dtype=np.float32)
    if method == "random_blocks":
        height, width = image.shape[:2]
        coarse = rng.random((height // 16, width // 16), dtype=np.float32)
        return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_NEAREST)
    raise ValueError(f"unknown method: {method}")


def sample_items(items: list[Item], per_class: int, seed: int) -> list[Item]:
    """Draw the same number of images from each class, reproducibly."""
    rng = np.random.default_rng(seed)
    chosen: list[Item] = []
    for label in sorted({item.label for item in items}):
        of_class = [item for item in items if item.label == label]
        picks = rng.choice(len(of_class), size=min(per_class, len(of_class)), replace=False)
        chosen += [of_class[i] for i in sorted(picks)]
    return chosen


def main() -> None:
    """Run the comparison and write per-image scores, mean curves and a summary."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True, choices=["alexnet", "densenet121"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True, help="folder with images/ and labels.csv")
    ap.add_argument("--per_class", type=int, default=50)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.arch, args.weights, device)
    predict_fn = make_predict_fn(model, device)
    items = sample_items(read_labels_csv(args.data_dir / "labels.csv", args.data_dir / "images"), args.per_class, args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    curves = {m: {"deletion": [], "insertion": []} for m in METHODS}
    for index, item in enumerate(items, start=1):
        image = np.asarray(Image.open(item.path).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        probs = predict_fn(image[np.newaxis])[0]
        target = int(np.argmax(probs))
        for method in METHODS:
            heat = np.asarray(explain(method, model, image, predict_fn, target, device, rng), dtype=np.float32)
            deletion = perturbation_curve(image, heat, predict_fn, target, "deletion", args.steps)
            insertion = perturbation_curve(image, heat, predict_fn, target, "insertion", args.steps)
            curves[method]["deletion"].append(deletion)
            curves[method]["insertion"].append(insertion)
            rows.append({
                "file": item.path.name, "label": item.label, "predicted": target, "confidence": float(probs[target]),
                "method": method, "deletion_auc": curve_auc(deletion), "insertion_auc": curve_auc(insertion),
                "border_share": border_share(heat),
            })
        print(f"[{index}/{len(items)}] {item.path.name}", flush=True)
        # Rewritten after every image so that an interrupted run keeps its finished images.
        with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {"arch": args.arch, "n_images": len(items), "per_class": args.per_class, "steps": args.steps,
               "seed": args.seed, "lime_samples": DEFAULT_LIME_SAMPLES, "shap_evals": DEFAULT_SHAP_EVALS, "methods": {}}
    for method in METHODS:
        scores = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            **{f"{key}_{stat}": float(getattr(np, stat)([r[key] for r in scores]))
               for key in ("deletion_auc", "insertion_auc", "border_share") for stat in ("mean", "std")},
            "deletion_curve": np.mean(curves[method]["deletion"], axis=0).round(4).tolist(),
            "insertion_curve": np.mean(curves[method]["insertion"], axis=0).round(4).tolist(),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for method, stats in summary["methods"].items():
        print(f"{method:<10} deletion {stats['deletion_auc_mean']:.3f}  insertion {stats['insertion_auc_mean']:.3f}  "
              f"border {stats['border_share_mean']:.3f}")


if __name__ == "__main__":
    main()
