"""Occlusion sensitivity on audio segments."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class OcclusionAudioResult:
    """Drop of the target score when each segment is silenced, and the unmasked score."""
    deltas: np.ndarray
    base: float

def explain(predict_target_from_mask: Callable[[np.ndarray], float], n_segments: int) -> OcclusionAudioResult:
    """Silence one segment at a time and record the drop of the target score."""
    base_mask = np.ones(n_segments, dtype=np.float32)
    base = float(predict_target_from_mask(base_mask))
    deltas = np.zeros(n_segments, dtype=np.float32)
    for i in range(n_segments):
        m = base_mask.copy()
        m[i] = 0.0
        val = float(predict_target_from_mask(m))
        deltas[i] = base - val
    return OcclusionAudioResult(deltas=deltas, base=base)
