"""Occlusion sensitivity on image patches."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class OcclusionImageResult:
    """Normalized sensitivity heatmap and its overlay."""
    heat: np.ndarray
    overlay_bgr: np.ndarray

def explain(
    img_float01: np.ndarray,
    predict_fn,
    target_index: int,
    patch: int=32,
    stride: int=16,
    batch: int=16,
) -> OcclusionImageResult:
    """Replace sliding patches by the mean colour and record the drop of the target probability."""
    h, w, _ = img_float01.shape
    base_probs = predict_fn(img_float01[np.newaxis, ...])[0]
    base = float(base_probs[target_index])
    mean_color = np.mean(img_float01, axis=(0, 1), keepdims=True)

    ys = list(range(0, h, stride))
    xs = list(range(0, w, stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)

    images = []
    positions = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            y0 = max(0, y - patch // 2)
            x0 = max(0, x - patch // 2)
            y1 = min(h, y0 + patch)
            x1 = min(w, x0 + patch)
            img = img_float01.copy()
            img[y0:y1, x0:x1, :] = mean_color
            images.append(img)
            positions.append((yi, xi))
            if len(images) >= batch:
                probs = predict_fn(np.stack(images, axis=0))
                for (py, px), p in zip(positions, probs):
                    heat[py, px] = base - float(p[target_index])
                images = []
                positions = []

    if images:
        probs = predict_fn(np.stack(images, axis=0))
        for (py, px), p in zip(positions, probs):
            heat[py, px] = base - float(p[target_index])

    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-9)
    heat_full = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    img_uint8 = (np.clip(img_float01, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    heat_uint8 = (heat_full * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(img_bgr, 0.60, heat_color, 0.40, 0)
    return OcclusionImageResult(heat=heat_full, overlay_bgr=overlay)
