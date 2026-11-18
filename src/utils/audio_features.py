"""Audio resampling and mel spectrograms."""

from __future__ import annotations
import numpy as np
import torch
import torchaudio

def resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    """Resample a waveform to the target rate when the rates differ."""
    if sr == target_sr:
        return x, sr
    t = torch.from_numpy(x).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, target_sr)
    return t.squeeze(0).numpy(), target_sr

def mel_spectrogram_image(
    x: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 256,
    top_db: float = 80.0,
) -> np.ndarray:
    """Return a log-mel spectrogram as a grayscale RGB image."""
    waveform = torch.from_numpy(x).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(mel)
    m = mel_db.squeeze(0).numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    img = (m * 255.0).astype(np.uint8)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb
