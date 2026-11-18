"""Evaluate a trained chest X-ray classifier on a labelled folder (for example the test split).

Usage:
    python -m src.eval.evaluate_xray --arch densenet121 --weights weights/image/densenet121_xray.pth \
        --data_dir data/xray_test --out results/test/densenet121.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..train.common import read_labels_csv
from ..train.train_xray import XRayDataset, build_model


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve from the rank-sum statistic (ties get the average rank)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores[np.diff(np.sort(scores), prepend=np.nan) == 0]):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    positives = labels == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    """Confusion counts and the usual rates. The positive class is pneumonia (label 1)."""
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "balanced_accuracy": 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0)),
        "auc": roc_auc(labels, scores),
    }


def load_model(arch: str, weights: Path, device: str) -> torch.nn.Module:
    """Build the architecture and load trained weights in evaluation mode."""
    model = build_model(arch)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def predict_scores(model: torch.nn.Module, data_dir: Path, device: str, batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return the labels and the predicted probability of pneumonia for every image of a folder."""
    items = read_labels_csv(data_dir / "labels.csv", data_dir / "images")
    labels, scores = [], []
    with torch.inference_mode():
        for xb, yb in DataLoader(XRayDataset(items), batch_size=batch, shuffle=False, num_workers=0):
            scores.append(torch.softmax(model(xb.to(device)), dim=-1)[:, 1].cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def main() -> None:
    """Evaluate and write the metrics as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", required=True, choices=["alexnet", "densenet121"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True, help="folder with images/ and labels.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels, scores = predict_scores(load_model(args.arch, args.weights, device), args.data_dir, device)
    metrics = {"arch": args.arch, "dataset_dir": args.data_dir.as_posix(), **binary_metrics(labels, scores)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
