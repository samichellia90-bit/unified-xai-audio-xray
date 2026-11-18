"""CPU-only unit tests. No model or dataset is downloaded."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.train.common import Item, read_labels_csv, split_train_val
from src.utils.images import apply_window_level, normalize_heatmap, zoom_pan
from src.xai import occlusion_audio, occlusion_image
from src.xai.gradcam_common import find_last_conv, gradcam_2d


class TinyCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv2(torch.relu(self.conv1(x))))
        return self.head(x.mean(dim=(2, 3)))


def test_find_last_conv_returns_the_deepest_layer() -> None:
    model = TinyCnn()
    assert find_last_conv(model, 2) is model.conv2
    assert find_last_conv(model, 1) is None


def test_gradcam_2d_returns_a_normalized_map_at_input_size() -> None:
    torch.manual_seed(0)
    result = gradcam_2d(TinyCnn().eval(), torch.rand(1, 3, 32, 48), target_index=1)
    assert result.heat.shape == (32, 48)
    assert result.overlay_bgr.shape == (32, 48, 3)
    assert 0.0 <= float(result.heat.min()) and float(result.heat.max()) <= 1.0 + 1e-6


def test_occlusion_image_peaks_on_the_region_the_classifier_reads() -> None:
    image = np.zeros((64, 64, 3), dtype=np.float32)
    image[40:56, 8:24] = 1.0  # bright square, bottom left

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        score = batch[:, 40:56, 8:24].mean(axis=(1, 2, 3))
        return np.stack([1.0 - score, score], axis=1)

    result = occlusion_image.explain(image, predict_fn, target_index=1, patch=16, stride=8)
    row, col = np.unravel_index(np.argmax(result.heat), result.heat.shape)
    assert result.heat.shape == (64, 64)
    assert 40 <= row < 56 and 8 <= col < 24


def test_occlusion_audio_reports_the_exact_score_drop() -> None:
    weights = np.array([0.1, 0.5, 0.0, 0.4], dtype=np.float32)
    result = occlusion_audio.explain(lambda mask: float(mask @ weights), n_segments=4)
    assert result.base == pytest.approx(1.0)
    assert result.deltas.tolist() == pytest.approx(weights.tolist())


def test_split_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("filename,label\n" + "".join(f"f{i}.png,{i % 2}\n" for i in range(20)), encoding="utf-8")
    items = read_labels_csv(csv_path, tmp_path)
    assert items[3] == Item(path=tmp_path / "f3.png", label=1)

    train, val = split_train_val(items, val_ratio=0.15, seed=0)
    assert (len(train), len(val)) == (17, 3)
    assert not {i.path for i in train} & {i.path for i in val}
    assert split_train_val(items, val_ratio=0.15, seed=0) == (train, val)


def test_image_helpers() -> None:
    heat = normalize_heatmap(np.array([[0.0, 2.0], [4.0, np.nan]]), clip_percentile=100.0)
    assert heat.tolist() == [[0.0, 0.5], [1.0, 0.0]]

    gray = np.full((4, 4, 3), 128, dtype=np.uint8)
    assert apply_window_level(gray, ww=1.0, wl=0.5).shape == (4, 4, 3)
    assert int(apply_window_level(gray, ww=1.0, wl=0.5, invert=True)[0, 0, 0]) == 126

    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    assert zoom_pan(image, zoom=2.0, pan_x=0.0, pan_y=0.0).shape == image.shape
