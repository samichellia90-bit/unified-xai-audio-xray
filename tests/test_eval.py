"""CPU-only tests of the evaluation code. No model or dataset is downloaded."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torchvision.models as tvm

from src.eval.evaluate_xray import binary_metrics, roc_auc
from src.eval.faithfulness import border_share, curve_auc, make_predict_fn, perturbation_curve, sample_items
from src.train.common import Item
from src.xai.gradcam_common import find_last_conv, gradcam_2d


def test_roc_auc_and_binary_metrics_on_a_hand_computed_case() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.8, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(5 / 6)
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)

    metrics = binary_metrics(labels, scores)
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (2, 1, 1, 1)
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx((2 / 3 + 0.5) / 2)


def brightness_predict_fn(images: np.ndarray) -> np.ndarray:
    """Probability of class 1 = mean brightness of the top-left quadrant."""
    score = images[:, :8, :8].mean(axis=(1, 2, 3))
    return np.stack([1.0 - score, score], axis=1)


def test_a_faithful_heatmap_beats_an_inverted_one_on_both_curves() -> None:
    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[:8, :8] = 1.0
    faithful = np.zeros((16, 16), dtype=np.float32)
    faithful[:8, :8] = 1.0
    inverted = 1.0 - faithful

    def auc(heat: np.ndarray, mode: str) -> float:
        return curve_auc(perturbation_curve(image, heat, brightness_predict_fn, 1, mode, steps=8))

    assert auc(faithful, "deletion") < auc(inverted, "deletion")
    assert auc(faithful, "insertion") > auc(inverted, "insertion")
    assert curve_auc(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    assert curve_auc(np.array([0.0, 1.0])) == pytest.approx(0.5)


def test_border_share() -> None:
    centre = np.zeros((20, 20), dtype=np.float32)
    centre[8:12, 8:12] = 1.0
    edge = np.zeros((20, 20), dtype=np.float32)
    edge[0, :] = 1.0
    assert border_share(centre, top_fraction=0.04) == 0.0
    assert border_share(edge, top_fraction=0.05) == 1.0


def test_sample_items_is_balanced_and_reproducible(tmp_path) -> None:
    items = [Item(path=tmp_path / f"{i}.png", label=i % 2) for i in range(20)]
    sample = sample_items(items, per_class=3, seed=0)
    assert [item.label for item in sample] == [0, 0, 0, 1, 1, 1]
    assert sample == sample_items(items, per_class=3, seed=0)


def test_gradcam_targets_the_last_dense_block_of_densenet() -> None:
    torch.manual_seed(0)
    model = tvm.densenet121(weights=None).eval()
    layer = find_last_conv(model, 2)
    assert layer is model.features.denseblock4

    x = torch.rand(1, 3, 64, 64)
    result = gradcam_2d(model, x, target_index=0, conv=layer)
    assert result.heat.shape == (64, 64)
    assert np.isfinite(result.heat).all()

    probs = make_predict_fn(model, "cpu")(np.random.default_rng(0).random((3, 64, 64, 3), dtype=np.float32))
    assert probs.shape == (3, 1000) and probs.sum(axis=1) == pytest.approx(1.0, abs=1e-4)
