"""Torchvision classifiers trained in this project (X-ray images and audio spectrograms)."""

from __future__ import annotations
from typing import Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

from .base import BaseModel
from ..config import AUDIO_TARGET_SR
from ..utils.audio_io import read_wav_bytes
from ..utils.audio_features import resample_if_needed, mel_spectrogram_image

class TorchVisionImageBinary(BaseModel):
    """Binary chest X-ray classifier (AlexNet or DenseNet121) with locally trained weights."""
    def __init__(self, arch: str, weights_path: str, display: str, device: Optional[str]=None):
        self.name = display
        self.modality = "image"
        self.class_names = ["benin", "malin"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.weights_path = Path(weights_path)

        self.available = self.weights_path.exists()
        if not self.available:
            self.unavailable_reason = f"Missing weights: {self.weights_path}"
            self.model = None
            return

        if arch == "alexnet":
            m = tvm.alexnet(weights=None)
            m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
            self.model = m
        elif arch == "densenet121":
            m = tvm.densenet121(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            self.model = m
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        sd = torch.load(self.weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device).eval()

        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def torch_model(self):
        """Return the underlying torch module."""
        return self.model

    def predict_proba(self, payload_bytes: bytes, mime: str):
        """Return the two class probabilities for an image file."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        pil = Image.open(__import__("io").BytesIO(payload_bytes)).convert("RGB")
        x = self.tf(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x), dim=-1)[0].detach().cpu().numpy()
        return probs.astype(np.float32), {}

    def build_tensor(self, payload_bytes: bytes):
        """Return the PIL image and its normalized input tensor."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        pil = Image.open(__import__("io").BytesIO(payload_bytes)).convert("RGB")
        x = self.tf(pil).unsqueeze(0).to(self.device)
        return pil, x

class TorchVisionAudioSpectroBinary(BaseModel):
    """Binary audio classifier on mel spectrograms (VGG16, ResNet18 or MobileNetV3) with locally trained weights."""
    def __init__(self, arch: str, weights_path: str, display: str, device: Optional[str]=None):
        self.name = display
        self.modality = "audio"
        self.class_names = ["reel", "faux"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.weights_path = Path(weights_path)

        self.available = self.weights_path.exists()
        if not self.available:
            self.unavailable_reason = f"Missing weights: {self.weights_path}"
            self.model = None
            return

        if arch == "vgg16":
            m = tvm.vgg16(weights=None)
            m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
            self.model = m
        elif arch == "resnet18":
            m = tvm.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            self.model = m
        elif arch == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=None)
            m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
            self.model = m
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        sd = torch.load(self.weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device).eval()

        self.tf = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def torch_model(self):
        """Return the underlying torch module."""
        return self.model

    def predict_proba(self, payload_bytes: bytes, mime: str):
        """Return the two class probabilities for a WAV file."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        x, sr = read_wav_bytes(payload_bytes)
        x, sr = resample_if_needed(x, sr, AUDIO_TARGET_SR)
        rgb = mel_spectrogram_image(x, sr)
        pil = Image.fromarray(rgb).convert("RGB")
        t = self.tf(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(t), dim=-1)[0].detach().cpu().numpy()
        return probs.astype(np.float32), {"sr": sr, "duration_s": float(x.shape[0]/sr)}
