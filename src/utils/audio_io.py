"""WAV decoding."""

from __future__ import annotations
import io
import numpy as np
import soundfile as sf

def read_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a mono float32 waveform and its sampling rate."""
    bio = io.BytesIO(wav_bytes)
    x, sr = sf.read(bio, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        x = np.zeros(16000, dtype=np.float32)
        sr = 16000
    return x.astype(np.float32), int(sr)
