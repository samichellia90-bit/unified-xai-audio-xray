"""Demo file download and model warm-up."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import urllib.request

DEMO_FILES: Dict[str, str] = {
    "chest_demo.png": "https://upload.wikimedia.org/wikipedia/commons/b/b3/Chest.png",
    "chest_demo_2.jpg": "https://upload.wikimedia.org/wikipedia/commons/7/72/Boy_chest_x-ray_pic.jpg",
    "chest_demo_3.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/9e/Chest_x-ray_-_posteroanterior_view.jpg",
    "chest_demo_4.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/ca/Chest_radiograph_of_a_lung_with_Kerley_B_lines.jpg",
    "voice_demo.wav": "https://upload.wikimedia.org/wikipedia/commons/3/34/Human_voice_with_clicking_noise_removed.wav",
    "voice_demo_freud.wav": "https://upload.wikimedia.org/wikipedia/commons/3/37/Sigmund_Freud%27s_Voice_%281938%29.wav",
    "voice_demo_wilf.wav": "https://upload.wikimedia.org/wikipedia/commons/c/c6/Wilf%27_Lunn_voice_sample.wav",
    "voice_demo_nav_female.wav": "https://upload.wikimedia.org/wikipedia/commons/e/ea/Navigation_Directions_Albanian_Female_Voice_1.wav",
}

def _download(url: str, out: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out.write_bytes(resp.read())

def download_demo_files(dest: Path=Path("data/demo"), force: bool=False) -> List[Path]:
    """Download the public demo files that are missing."""
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for name, url in DEMO_FILES.items():
        out = dest / name
        if out.exists() and not force:
            continue
        _download(url, out)
        downloaded.append(out)
    return downloaded

def warmup_models() -> List[str]:
    """Load the pretrained models once so that they are cached."""
    from ..core.types import Modality
    from ..models.registry import list_models, load_model

    logs: List[str] = []
    audio_path = Path("data/demo/voice_demo.wav")
    image_path = Path("data/demo/chest_demo.png")
    if not audio_path.exists():
        audio_path = Path("data/demo/placeholder_tone.wav")
    if not image_path.exists():
        image_path = Path("data/demo/placeholder_image.png")

    audio_bytes = audio_path.read_bytes() if audio_path.exists() else None
    image_bytes = image_path.read_bytes() if image_path.exists() else None

    for spec in list_models(Modality.AUDIO, include_custom=False):
        m = load_model(spec)
        if hasattr(m, "available") and not getattr(m, "available", True):
            logs.append(f"{spec.display}: unavailable")
            continue
        if audio_bytes:
            _ = m.predict_proba(audio_bytes, "audio/wav")
        logs.append(f"{spec.display}: ok")

    for spec in list_models(Modality.IMAGE, include_custom=False):
        m = load_model(spec)
        if hasattr(m, "available") and not getattr(m, "available", True):
            logs.append(f"{spec.display}: unavailable")
            continue
        if image_bytes:
            _ = m.predict_proba(image_bytes, "image/png")
        logs.append(f"{spec.display}: ok")

    return logs
