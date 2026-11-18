"""Wav2Vec2 audio classifiers loaded from the Hugging Face hub."""

from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AutoProcessor

from .base import BaseModel
from ..config import AUDIO_TARGET_SR
from ..utils.audio_io import read_wav_bytes
from ..utils.audio_features import resample_if_needed

class AudioWav2Vec2HF(BaseModel):
    """Wav2Vec2 bona fide vs spoof classifier."""
    def __init__(self, model_id: str, display: str, device: Optional[str] = None):
        self.name = display
        self.modality = "audio"
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not model_id:
            self.available = False
            self.unavailable_reason = "Identifiant du modele HF manquant. Verifiez src/config.py."
            self.processor = None
            self.model = None
            self.class_names = ["reel", "faux"]
            return

        try:
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            except Exception:
                self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device).eval()
        except Exception as e:
            self.available = False
            msg = f"Could not load Hugging Face model: {e}"
            if "torch.load" in str(e) and "upgrade torch" in str(e):
                msg += " | Essayez: py -m pip install --upgrade torch torchvision torchaudio"
            self.unavailable_reason = msg
            self.processor = None
            self.model = None
            self.class_names = ["reel", "faux"]
            return

        id2label = getattr(self.model.config, "id2label", {0: "bonafide", 1: "spoof"})
        raw = [id2label[i] if i in id2label else str(i) for i in range(len(id2label))]
        self.class_names = ["reel" if r.lower()=="bonafide" else ("faux" if r.lower()=="spoof" else r) for r in raw]
        self._raw_labels = raw

    def torch_model(self):
        """Return the underlying torch module."""
        return self.model

    def predict_proba(self, payload_bytes: bytes, mime: str):
        """Return class probabilities and audio metadata for a WAV file."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        x, sr = read_wav_bytes(payload_bytes)
        x, sr = resample_if_needed(x, sr, AUDIO_TARGET_SR)
        inputs = self.processor([x], sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        extra = {"raw_labels": self._raw_labels, "sr": sr, "duration_s": float(x.shape[0]/sr), "n_samples": int(x.shape[0])}
        return probs.astype(np.float32), extra

    def build_inputs(self, payload_bytes: bytes):
        """Return the waveform, its sampling rate and the model inputs."""
        if not self.available:
            raise RuntimeError(self.unavailable_reason or "Model unavailable.")
        x, sr = read_wav_bytes(payload_bytes)
        x, sr = resample_if_needed(x, sr, AUDIO_TARGET_SR)
        inputs = self.processor([x], sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return x, sr, inputs
