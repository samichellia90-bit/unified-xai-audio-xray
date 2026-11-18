"""Plot the mean deletion and insertion curves of each explanation method from a summary file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# One fixed colour per method; the two random references are drawn in grey with their own dash pattern.
STYLE = {
    "gradcam": ("Grad-CAM", "#2a78d6"),
    "occlusion": ("Occlusion", "#eb6834"),
    "lime": ("LIME", "#1baf7a"),
    "shap": ("SHAP", "#eda100"),
    "random": ("Random pixels", "#898781"),
    "random_blocks": ("Random blocks", "#52514e"),
}
DASHES = {"random": (0, (4, 3)), "random_blocks": (0, (1, 2))}
PANELS = {
    "deletion": "Deletion: most important pixels removed first (lower is better)",
    "insertion": "Insertion: most important pixels revealed first (higher is better)",
}
SURFACE, INK, INK_SECONDARY, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def main() -> None:
    """Draw one panel per perturbation mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="summary.json written by src.eval.faithfulness")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE, sharey=True)
    for ax, (mode, title) in zip(axes, PANELS.items()):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
        for key, (label, colour) in STYLE.items():
            stats = summary["methods"][key]
            curve = np.array(stats[f"{mode}_curve"])
            fraction = np.linspace(0, 100, len(curve))
            ax.plot(fraction, curve, color=colour, linewidth=2, linestyle=DASHES.get(key, "-"),
                    label=f"{label} (AUC {stats[f'{mode}_auc_mean']:.3f})")
        ax.set_xlabel("Pixels changed (%)", fontsize=9, color=INK_SECONDARY)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[0].set_ylabel("Probability of the predicted class", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"{summary['arch']}: mean over {summary['n_images']} test images", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
