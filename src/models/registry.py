"""Catalogue of the available models."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from ..core.types import Modality
from ..config import AUDIO_HF_MODELS, XRAY_XRV_WEIGHTS
from .base import BaseModel
from .audio_wav2vec2 import AudioWav2Vec2HF
from .xray_torchxrv import XRayDenseNetXRV
from .custom_torchvision import TorchVisionImageBinary, TorchVisionAudioSpectroBinary

@dataclass(frozen=True)
class ModelSpec:
    """Identifier, display name and weights location of a model."""
    key: str
    display: str
    modality: Modality
    kind: str
    weights_hint: Optional[str] = None

def list_models(modality: Modality, include_custom: bool=True) -> List[ModelSpec]:
    """List the models available for a modality."""
    specs: List[ModelSpec] = []
    if modality == Modality.AUDIO:
        for k, cfg in AUDIO_HF_MODELS.items():
            specs.append(ModelSpec(k, cfg["display"], Modality.AUDIO, "pretrained"))
        if include_custom:
            specs += [
                ModelSpec("aud_vgg16", "VGG16 (CNN spectrogramme) - poids custom", Modality.AUDIO, "custom", "weights/audio/vgg16_spec.pth"),
                ModelSpec("aud_resnet18", "ResNet18 (CNN spectrogramme) - poids custom", Modality.AUDIO, "custom", "weights/audio/resnet18_spec.pth"),
                ModelSpec("aud_mnetv3", "MobileNetV3Small (CNN spectrogramme) - poids custom", Modality.AUDIO, "custom", "weights/audio/mobilenetv3_spec.pth"),
            ]
        return specs

    for k, cfg in XRAY_XRV_WEIGHTS.items():
        specs.append(ModelSpec(f"xrv_{k}", cfg["display"], Modality.IMAGE, "pretrained"))
    if include_custom:
        specs += [
            ModelSpec("img_alexnet", "AlexNet (X-ray thorax) - poids custom", Modality.IMAGE, "custom", "weights/image/alexnet_xray.pth"),
            ModelSpec("img_densenet121", "DenseNet121 (X-ray thorax) - poids custom", Modality.IMAGE, "custom", "weights/image/densenet121_xray.pth"),
        ]
    return specs

def load_model(spec: ModelSpec) -> BaseModel:
    """Instantiate the model described by a specification."""
    if spec.modality == Modality.AUDIO and spec.kind == "pretrained":
        cfg = AUDIO_HF_MODELS[spec.key]
        return AudioWav2Vec2HF(cfg["model_id"], cfg["display"])
    if spec.modality == Modality.IMAGE and spec.kind == "pretrained":
        key = spec.key.replace("xrv_", "")
        cfg = XRAY_XRV_WEIGHTS[key]
        return XRayDenseNetXRV(cfg["weights"], cfg["display"])

    if spec.key == "img_alexnet":
        return TorchVisionImageBinary("alexnet", spec.weights_hint, spec.display)
    if spec.key == "img_densenet121":
        return TorchVisionImageBinary("densenet121", spec.weights_hint, spec.display)
    if spec.key == "aud_vgg16":
        return TorchVisionAudioSpectroBinary("vgg16", spec.weights_hint, spec.display)
    if spec.key == "aud_resnet18":
        return TorchVisionAudioSpectroBinary("resnet18", spec.weights_hint, spec.display)
    if spec.key == "aud_mnetv3":
        return TorchVisionAudioSpectroBinary("mobilenet_v3_small", spec.weights_hint, spec.display)

    raise ValueError(f"Unknown spec: {spec}")
