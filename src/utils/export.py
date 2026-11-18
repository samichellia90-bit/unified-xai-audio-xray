"""PNG and JSON export helpers."""

from __future__ import annotations
import io, json
import numpy as np
from PIL import Image

def fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def rgb_array_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes."""
    im = Image.fromarray(rgb.astype("uint8"))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def json_bytes(obj) -> bytes:
    """Serialize an object to UTF-8 JSON bytes."""
    return json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")
