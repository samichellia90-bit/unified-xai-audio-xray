"""Shared data types."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import numpy as np

class Modality(str, Enum):
    """Input modality handled by the interface."""
    AUDIO = "audio"
    IMAGE = "image"

@dataclass
class InputPayload:
    """Raw uploaded file with its modality and MIME type."""
    modality: Modality
    bytes_data: bytes
    filename: str
    mime: str

@dataclass
class Prediction:
    """Predicted label, confidence and class probabilities."""
    label: str
    confidence: float
    proba_by_class: np.ndarray
    class_names: List[str]
    extra: Optional[Dict[str, Any]] = None
