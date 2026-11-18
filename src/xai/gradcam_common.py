"""Grad-CAM for 1D (waveform) and 2D (image) convolutional models."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

@dataclass
class GradCam1D:
    """Normalized activation map along time."""
    cam: np.ndarray

@dataclass
class GradCam2D:
    """Normalized heatmap at input resolution and its overlay."""
    heat: np.ndarray
    overlay_bgr: np.ndarray

def find_last_conv(model: nn.Module, dim: int):
    """Return the Grad-CAM target layer: the last Conv1d or Conv2d of a model.

    For DenseNet the last dense block is returned instead: its output is the full
    feature map, whereas its last convolution only adds 32 channels to it.
    """
    dense_block = getattr(getattr(model, "features", None), "denseblock4", None)
    if dim == 2 and dense_block is not None:
        return dense_block
    last = None
    for m in model.modules():
        if dim == 1 and isinstance(m, nn.Conv1d):
            last = m
        if dim == 2 and isinstance(m, nn.Conv2d):
            last = m
    return last

def _disable_inplace_relu(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)) and getattr(m, "inplace", False):
            m.inplace = False

def gradcam_1d(model: nn.Module, inputs: dict, target_index: int, conv: Optional[nn.Module]=None) -> GradCam1D:
    """Grad-CAM on the last Conv1d layer for the target class."""
    conv = conv or find_last_conv(model, 1)
    if conv is None:
        raise ValueError("No Conv1d found.")
    _disable_inplace_relu(model)
    model.zero_grad(set_to_none=True)
    acts, grads = [], []
    def fwd(_m,_i,o): acts.append(o)
    def bwd(_m,_gi,go): grads.append(go[0])
    h1 = conv.register_forward_hook(fwd)
    try:
        h2 = conv.register_full_backward_hook(bwd)
    except Exception:
        h2 = conv.register_backward_hook(lambda _m,_gi,go: grads.append(go[0]))

    with torch.enable_grad():
        out = model(**inputs).logits
        score = out[:, target_index].sum()
        score.backward()

    h1.remove(); h2.remove()
    A = acts[-1].detach()
    G = grads[-1].detach()
    w = G.mean(dim=2, keepdim=True)
    cam = torch.relu((w*A).sum(dim=1))[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)
    return GradCam1D(cam=cam.detach().cpu().numpy())

def gradcam_2d(model: nn.Module, x: torch.Tensor, target_index: int, conv: Optional[nn.Module]=None) -> GradCam2D:
    """Grad-CAM on the last Conv2d layer for the target class, resized to the input."""
    import cv2
    conv = conv or find_last_conv(model, 2)
    if conv is None:
        raise ValueError("No Conv2d found.")
    _disable_inplace_relu(model)
    model.zero_grad(set_to_none=True)
    acts, grads = [], []
    def fwd(_m,_i,o): acts.append(o)
    def bwd(_m,_gi,go): grads.append(go[0])
    h1 = conv.register_forward_hook(fwd)
    try:
        h2 = conv.register_full_backward_hook(bwd)
    except Exception:
        h2 = conv.register_backward_hook(lambda _m,_gi,go: grads.append(go[0]))

    with torch.enable_grad():
        out = model(x)
        score = out[:, target_index].sum()
        score.backward()

    h1.remove(); h2.remove()
    A = acts[-1].detach()
    G = grads[-1].detach()
    w = G.mean(dim=(2,3), keepdim=True)
    cam = torch.relu((w*A).sum(dim=1))[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-9)
    cam_np = cam.detach().cpu().numpy()

    _, _, H, W = x.shape
    heat = cv2.resize(cam_np, (W,H))

    img = x[0].detach().cpu().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.transpose(img, (1,2,0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img_uint8 = (img*255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    heat_uint8 = (heat*255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.60, heat_color, 0.40, 0)
    return GradCam2D(heat=heat, overlay_bgr=overlay)
