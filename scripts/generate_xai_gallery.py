"""Generate Grad-CAM and LIME images for every demo X-ray."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.types import Modality
from src.models.registry import list_models, load_model
from src.utils.images import to_uint8_rgb, heatmap_overlay, smooth_heatmap, normalize_heatmap
from src.xai import lime_image, shap_image, occlusion_image
from src.xai.gradcam_common import gradcam_2d, find_last_conv
from src.config import DEFAULT_LIME_SAMPLES, DEFAULT_SHAP_EVALS


def list_image_models() -> None:
    specs = list_models(Modality.IMAGE, include_custom=True)
    for s in specs:
        hint = f" ({s.weights_hint})" if s.weights_hint else ""
        print(f"{s.key}: {s.display} [{s.kind}]{hint}")


def pick_image_model(model_key: str | None) -> object:
    specs = list_models(Modality.IMAGE, include_custom=True)
    if not specs:
        raise RuntimeError("Aucun modele image disponible.")
    if model_key:
        for s in specs:
            if s.key == model_key:
                return load_model(s), s
        raise RuntimeError(f"Modele inconnu: {model_key}")

    # Priorite: custom avec poids disponibles, sinon premier pre-entraine.
    for s in specs:
        if s.kind == "custom" and s.weights_hint and Path(s.weights_hint).exists():
            return load_model(s), s
    return load_model(specs[0]), specs[0]


def image_predict_fn(model, imgs_float: np.ndarray) -> np.ndarray:
    out = []
    for im in imgs_float:
        buf = io.BytesIO()
        Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG")
        p, _ = model.predict_proba(buf.getvalue(), "image/png")
        out.append(p)
    return np.vstack(out).astype(np.float32)


def colorize_heatmap(heat01: np.ndarray, cmap: str) -> np.ndarray:
    cmap_map = {
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
    }
    heat_uint8 = (np.clip(heat01, 0, 1) * 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(heat_uint8, cmap_map.get(cmap, cv2.COLORMAP_INFERNO))
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def collect_images(path: Path | None) -> list[Path]:
    if path is None:
        path = Path("data/demo")
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(path.glob(ext))
    return sorted(files)


def main() -> int:
    """Explain each image with the selected model and save the heatmaps and overlays."""
    p = argparse.ArgumentParser(description="Genere une galerie XAI (image) avec Grad-CAM/LIME/SHAP/Occlusion.")
    p.add_argument("--input", type=str, default="", help="Fichier image ou dossier (defaut: data/demo)")
    p.add_argument("--out", type=str, default="outputs/xai_gallery", help="Dossier de sortie")
    p.add_argument("--model", type=str, default="", help="Cle du modele (voir --list-models)")
    p.add_argument("--list-models", action="store_true", help="Lister les modeles image disponibles")
    p.add_argument("--methods", type=str, default="gradcam,lime,shap,occlusion", help="Methodes separees par virgule")
    default_lime = min(400, int(DEFAULT_LIME_SAMPLES))
    default_shap = min(200, int(DEFAULT_SHAP_EVALS))
    p.add_argument("--lime-samples", type=int, default=default_lime)
    p.add_argument("--shap-evals", type=int, default=default_shap)
    p.add_argument("--occ-patch", type=int, default=32)
    p.add_argument("--occ-stride", type=int, default=16)
    p.add_argument("--heat-blur", type=int, default=7)
    p.add_argument("--heat-clip", type=float, default=98.5)
    p.add_argument("--heat-cmap", type=str, default="inferno")
    args = p.parse_args()

    if args.list_models:
        list_image_models()
        return 0

    input_path = Path(args.input) if args.input else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_images(input_path)
    if not files:
        print("Aucune image trouvee.")
        return 1

    model, spec = pick_image_model(args.model or None)
    if hasattr(model, "available") and not getattr(model, "available", True):
        reason = getattr(model, "unavailable_reason", "Modele indisponible.")
        raise RuntimeError(reason)

    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    supported = {"gradcam", "lime", "shap", "occlusion"}
    methods = [m for m in methods if m in supported]
    if not methods:
        print("Aucune methode valide.")
        return 1

    print(f"Modele: {spec.display} ({spec.key})")
    for fp in files:
        payload = fp.read_bytes()
        probs, _ = model.predict_proba(payload, "image/png")
        idx = int(np.argmax(probs))
        label = model.class_names[idx] if hasattr(model, "class_names") else str(idx)
        conf = float(probs[idx])

        pil, x_tensor = model.build_tensor(payload)
        base224 = to_uint8_rgb(pil, (224, 224))
        base512 = to_uint8_rgb(pil, (512, 512))
        img_float = base224.astype(np.float32) / 255.0

        file_dir = out_dir / fp.stem
        file_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(base512).save(file_dir / "base.png")

        meta = {
            "file": str(fp),
            "model_key": spec.key,
            "model_name": spec.display,
            "label": label,
            "confidence": conf,
        }
        (file_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        def predict_fn(imgs_float: np.ndarray) -> np.ndarray:
            return image_predict_fn(model, imgs_float)

        for method in methods:
            heat224 = None
            if method == "lime":
                res = lime_image.explain(base224, predict_fn, args.lime_samples)
                heat224 = res.mask.astype(np.float32)
            elif method == "shap":
                res = shap_image.explain(img_float, predict_fn, args.shap_evals)
                heat224 = res.heat.astype(np.float32)
            elif method == "occlusion":
                res = occlusion_image.explain(
                    img_float,
                    predict_fn,
                    target_index=idx,
                    patch=int(args.occ_patch),
                    stride=int(args.occ_stride),
                )
                heat224 = res.heat.astype(np.float32)
            elif method == "gradcam":
                target_name = getattr(model, "proxy_keys", ["Mass"])[0]
                if hasattr(model, "pathologies") and target_name in getattr(model, "pathologies", []):
                    target_index = model.pathologies.index(target_name)
                else:
                    target_index = 0
                conv = find_last_conv(model.model, 2)
                if conv is None:
                    print(f"[WARN] Grad-CAM indisponible pour {fp.name} (pas de Conv2d).")
                    continue
                res = gradcam_2d(model.model, x_tensor, target_index=target_index, conv=conv)
                heat224 = res.heat.astype(np.float32)

            if heat224 is None:
                continue
            heat224 = np.maximum(heat224, 0.0)
            heat224 = smooth_heatmap(heat224, int(args.heat_blur))
            heat224 = normalize_heatmap(heat224, float(args.heat_clip))

            overlay224 = heatmap_overlay(base224, heat224, alpha=0.55, cmap=args.heat_cmap)
            heat_color = colorize_heatmap(heat224, args.heat_cmap)

            overlay512 = np.array(Image.fromarray(overlay224).resize((512, 512)))
            heat512 = np.array(Image.fromarray(heat_color).resize((512, 512)))

            Image.fromarray(overlay512).save(file_dir / f"{method}_overlay.png")
            Image.fromarray(heat512).save(file_dir / f"{method}_heat.png")

        print(f"OK: {fp.name} -> {file_dir}")

    print(f"Galerie generee dans: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
