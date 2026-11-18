"""Explanation methods available for each modality."""

from __future__ import annotations
from typing import Dict, List
from ..core.types import Modality

DISPLAY = {"gradcam":"Grad-CAM", "lime":"LIME", "shap":"SHAP", "occlusion":"Occlusion"}
ALLOWED: Dict[Modality, List[str]] = {
    Modality.AUDIO: ["gradcam","lime","shap","occlusion"],
    Modality.IMAGE: ["gradcam","lime","shap","occlusion"],
}
