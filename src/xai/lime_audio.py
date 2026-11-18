"""LIME on audio: features are time segments that are kept or silenced."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

@dataclass
class LimeAudioResult:
    """LIME weight of each segment and the segments ranked by absolute weight."""
    weights: np.ndarray
    ranked: list[tuple[int, float]]

def explain(predict_proba_from_mask: Callable[[np.ndarray], np.ndarray], n_segments: int, num_samples: int=800) -> LimeAudioResult:
    """Fit LIME on random segment masks and return the weight of each segment."""
    rng = np.random.default_rng(42)
    X = rng.integers(0, 2, size=(num_samples, n_segments)).astype(np.float32)
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=[f"seg_{i}" for i in range(n_segments)],
        mode="classification",
        discretize_continuous=False,
    )
    x0 = np.ones((n_segments,), dtype=np.float32)

    def clf_fn(masks_2d: np.ndarray) -> np.ndarray:
        return np.vstack([predict_proba_from_mask(m) for m in masks_2d])

    exp = explainer.explain_instance(x0, clf_fn, num_features=min(12, n_segments), top_labels=1)
    label_to_explain = 1 if 1 in exp.local_exp else list(exp.local_exp.keys())[0]
    local = exp.local_exp[label_to_explain]

    w = np.zeros(n_segments, dtype=np.float32)
    for idx, val in local:
        w[idx] = float(val)
    ranked = sorted([(i, float(w[i])) for i in range(n_segments)], key=lambda t: abs(t[1]), reverse=True)
    return LimeAudioResult(weights=w, ranked=ranked)
