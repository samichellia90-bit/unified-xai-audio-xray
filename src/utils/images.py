"""Image conversion, heatmap rendering and viewer transforms."""

from __future__ import annotations
import io
import numpy as np
import cv2
from PIL import Image

def pil_from_bytes(b: bytes) -> Image.Image:
    """Decode image bytes to an RGB PIL image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def to_uint8_rgb(pil: Image.Image, size: tuple[int,int] | None = None) -> np.ndarray:
    """Convert a PIL image to a uint8 RGB array, optionally resized."""
    if size is not None:
        pil = pil.resize(size)
    return np.array(pil).astype(np.uint8)

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR array to RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def blend_overlay_rgb(base_rgb: np.ndarray, overlay_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend an overlay on a base image."""
    base = base_rgb.astype(np.float32)
    ov = overlay_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + alpha * ov
    return np.clip(out, 0, 255).astype(np.uint8)

def smooth_heatmap(heat: np.ndarray, k: int=7) -> np.ndarray:
    """Gaussian-blur a heatmap with an odd kernel size."""
    if k is None or k < 2:
        return heat
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    return cv2.GaussianBlur(heat.astype(np.float32), (kk, kk), 0)

def normalize_heatmap(heat: np.ndarray, clip_percentile: float=99.0) -> np.ndarray:
    """Scale a heatmap to [0, 1] after clipping at a percentile."""
    h = np.nan_to_num(heat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    h = h - float(np.min(h))
    if clip_percentile is not None and clip_percentile < 100.0:
        hi = float(np.percentile(h, clip_percentile))
        if hi > 0:
            h = np.clip(h, 0.0, hi)
    mx = float(np.max(h))
    if mx > 0:
        h = h / mx
    return h

def heatmap_overlay(base_rgb: np.ndarray, heat01: np.ndarray, alpha: float=0.45, cmap: str="inferno") -> np.ndarray:
    """Colorize a heatmap and blend it on the base image."""
    cmap_map = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    heat_uint8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(heat_uint8, cmap_map.get(cmap, cv2.COLORMAP_INFERNO))
    color_rgb = bgr_to_rgb(color_bgr)
    return blend_overlay_rgb(base_rgb, color_rgb, alpha)

def heatmap_color(heat01: np.ndarray, cmap: str="inferno") -> np.ndarray:
    """Colorize a heatmap in [0, 1]."""
    cmap_map = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    heat_uint8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(heat_uint8, cmap_map.get(cmap, cv2.COLORMAP_INFERNO))
    return bgr_to_rgb(color_bgr)

def apply_window_level(rgb_uint8: np.ndarray, ww: float, wl: float, invert: bool=False, gamma: float=1.0) -> np.ndarray:
    """Apply window width, window level, inversion and gamma to a grayscale view."""
    x = rgb_uint8.astype(np.float32)/255.0
    g = (0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2])
    lo = wl - ww/2.0
    hi = wl + ww/2.0
    y = (g - lo) / (hi - lo + 1e-9)
    y = np.clip(y, 0, 1)
    if invert:
        y = 1.0 - y
    if gamma and abs(gamma-1.0) > 1e-6:
        y = np.power(y, 1.0/max(gamma, 1e-6))
    out = (np.stack([y,y,y], axis=-1)*255.0).astype(np.uint8)
    return out

def zoom_pan(rgb_uint8: np.ndarray, zoom: float, pan_x: float, pan_y: float) -> np.ndarray:
    """Crop around a panned centre and resize back to the original size."""
    h, w = rgb_uint8.shape[:2]
    z = max(1.0, float(zoom))
    if z <= 1.0 and (abs(pan_x) > 1e-3 or abs(pan_y) > 1e-3):
        z = 1.1
    crop_h = int(round(h / z))
    crop_w = int(round(w / z))
    cx = int(round(w/2 + pan_x * (w - crop_w)/2))
    cy = int(round(h/2 + pan_y * (h - crop_h)/2))
    x0 = max(0, min(w - crop_w, cx - crop_w//2))
    y0 = max(0, min(h - crop_h, cy - crop_h//2))
    crop = rgb_uint8[y0:y0+crop_h, x0:x0+crop_w]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    return out
