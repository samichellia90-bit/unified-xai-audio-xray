"""Common interface of the classifiers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch

class BaseModel(ABC):
    """Classifier wrapper exposing class probabilities and, when available, its torch module."""
    name: str
    modality: str
    class_names: List[str]
    available: bool = True
    unavailable_reason: Optional[str] = None

    @abstractmethod
    def predict_proba(self, payload_bytes: bytes, mime: str) -> tuple[np.ndarray, Dict[str, Any]]:
        """Return class probabilities and extra information for a raw file."""
        raise NotImplementedError

    def torch_model(self) -> Optional[torch.nn.Module]:
        """Return the torch module used by gradient-based explanations, if any."""
        return None
