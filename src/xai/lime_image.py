"""LIME on images with SLIC superpixels."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
from lime import lime_image
from skimage.segmentation import slic

@dataclass
class LimeImageResult:
    """Positive-evidence heatmap and its overlay."""
    mask: np.ndarray
    overlay_bgr: np.ndarray

def explain(img_uint8: np.ndarray, predict_fn, num_samples: int=800) -> LimeImageResult:
    """Explain the top predicted class and return the superpixel weights as a heatmap."""
    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = lambda x: slic(x, n_segments=160, compactness=12, sigma=1, start_label=0)
    exp = explainer.explain_instance(
        img_uint8,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn,
    )
    label = exp.top_labels[0]
    segments = getattr(exp, "segments", None)
    if segments is None:
        segments = segmentation_fn(img_uint8)
    weights = dict(exp.local_exp[label])
    heat = np.zeros(segments.shape, dtype=np.float32)
    for seg_id, w in weights.items():
        heat[segments == seg_id] = float(w)
    heat = np.maximum(heat, 0.0)
    if float(heat.max()) > float(heat.min()):
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)

    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(img_bgr, 0.60, heat_color, 0.40, 0)
    return LimeImageResult(mask=heat, overlay_bgr=overlay)
