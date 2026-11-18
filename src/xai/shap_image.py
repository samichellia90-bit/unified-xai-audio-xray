"""Partition SHAP on images with a blur masker."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import shap

@dataclass
class ShapImageResult:
    """Normalized mean absolute SHAP heatmap and its overlay."""
    heat: np.ndarray
    overlay_bgr: np.ndarray

def explain(img_float01: np.ndarray, predict_fn, max_evals: int=200) -> ShapImageResult:
    """Compute SHAP values with a blur masker and reduce them to one heatmap."""
    masker = shap.maskers.Image("blur(8,8)", img_float01.shape)
    explainer = shap.Explainer(predict_fn, masker)
    sv = explainer(img_float01[np.newaxis, ...], max_evals=max_evals)
    vals = np.array(sv.values)
    if vals.ndim >= 5:
        heat = np.mean(np.abs(vals[0]), axis=(2,3))
    else:
        heat = np.mean(np.abs(vals[0]), axis=2)
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-9)

    img_uint8 = (np.clip(img_float01,0,1)*255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    heat_uint8 = (heat*255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(img_bgr, 0.60, heat_color, 0.40, 0)
    return ShapImageResult(heat=heat, overlay_bgr=overlay)
