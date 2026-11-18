"""Kernel SHAP on audio segments."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import shap

@dataclass
class ShapAudioResult:
    """SHAP value of each segment and the expected score."""
    values: np.ndarray
    base_value: float

def explain(predict_target_from_mask: Callable[[np.ndarray], float], n_segments: int, nsamples: int=200, background: int=30) -> ShapAudioResult:
    """Estimate the SHAP value of each segment with a random-mask background."""
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 2, size=(background, n_segments)).astype(np.float32)

    def f(masks_2d: np.ndarray) -> np.ndarray:
        return np.array([[predict_target_from_mask(m)] for m in masks_2d], dtype=np.float32)

    explainer = shap.KernelExplainer(f, bg)
    x0 = np.ones((1, n_segments), dtype=np.float32)
    sv = explainer.shap_values(x0, nsamples=nsamples)
    if isinstance(sv, list):
        sv = sv[0]
    values = np.array(sv)[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[0]
    return ShapAudioResult(values=values.astype(np.float32), base_value=float(base))
