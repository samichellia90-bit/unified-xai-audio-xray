"""Pretrained TorchXRayVision DenseNet121 reduced to a binary score."""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchxrayvision as xrv

from .base import BaseModel

class XRayDenseNetXRV(BaseModel):
    """Multi-label chest X-ray model; the binary score is the maximum probability over a set of lung findings."""
    def __init__(self, weights: str, display: str, device: Optional[str] = None):
        self.name = display
        self.modality = "image"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.model = xrv.models.DenseNet(weights=weights).to(self.device).eval()
            self.pathologies = list(self.model.pathologies)
        except Exception as e:
            self.available = False
            self.unavailable_reason = f"Could not load TorchXRayVision weights: {e}"
            self.model = None
            self.pathologies = []
            self.class_names = ["benin", "malin"]
            return

        self.class_names = ["benin", "malin"]
        self.proxy_keys = [k for k in ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration", "Lung Lesion", "Mass", "Nodule"] if k in self.pathologies] or ([self.pathologies[0]] if self.pathologies else ["Mass"])

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def torch_model(self):
        """Return the underlying torch module."""
        return self.model

    def predict_proba(self, payload_bytes: bytes, mime: str):
        """Return the binary probabilities and the five most likely findings."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        pil = Image.open(__import__("io").BytesIO(payload_bytes)).convert("RGB")
        x = self.tf(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        idxs = [self.pathologies.index(k) for k in self.proxy_keys if k in self.pathologies]
        mal_score = float(np.max(probs[idxs])) if idxs else float(np.max(probs))
        bin_probs = np.array([1.0 - mal_score, mal_score], dtype=np.float32)

        top_idx = np.argsort(-probs)[:5]
        top5 = [(self.pathologies[i], float(probs[i])) for i in top_idx] if self.pathologies else []

        extra = {
            "proxy_keys": self.proxy_keys,
            "malignant_score_rule": "max(prob[key]) over proxy_keys",
            "top5_pathologies": top5,
        }
        return bin_probs, extra

    def build_tensor(self, payload_bytes: bytes):
        """Return the PIL image and its normalized single-channel tensor."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        pil = Image.open(__import__("io").BytesIO(payload_bytes)).convert("RGB")
        x = self.tf(pil).unsqueeze(0).to(self.device)
        return pil, x
