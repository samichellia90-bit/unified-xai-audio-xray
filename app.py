"""Streamlit interface: classification and explanations for audio clips and chest X-rays."""

from __future__ import annotations
import io, csv, json, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

from src.core.types import Modality, InputPayload
from src.config import (
    DEFAULT_AUDIO_SEGMENTS, DEFAULT_LIME_SAMPLES, DEFAULT_SHAP_EVALS, DEFAULT_OVERLAY_ALPHA,
    DEFAULT_WW, DEFAULT_WL, DEFAULT_GAMMA, AUDIO_TARGET_SR
)
from src.models.registry import list_models, load_model, ModelSpec
from src.ui.style import APP_CSS

from src.utils.audio_io import read_wav_bytes
from src.utils.audio_features import resample_if_needed, mel_spectrogram_image
from src.utils.viz import (
    prob_bar_chart, prob_bar_chart_plotly,
    audio_importance_plot, audio_importance_plotly,
    spectrogram_plotly,
)
from src.utils.demo_setup import download_demo_files
from src.utils.export import fig_to_png_bytes, rgb_array_to_png_bytes
from src.utils.images import (
    pil_from_bytes, to_uint8_rgb, blend_overlay_rgb, apply_window_level, zoom_pan,
    normalize_heatmap, smooth_heatmap, heatmap_overlay, heatmap_color,
)
from src.report.pdf_report import build_pdf_report

from src.xai.compat import DISPLAY, ALLOWED
from src.xai.gradcam_common import gradcam_1d, gradcam_2d, find_last_conv
from src.xai import lime_audio, shap_audio, lime_image, shap_image, occlusion_audio, occlusion_image

st.set_page_config(page_title="Interface XAI unifiee", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    """<div class="card hero">
        <div class="badge">Interface XAI unifiee</div>
        <h1 style="margin: 10px 0 6px 0;">Explicabilite pour audio et radiographies thoraciques</h1>
        <div class="small-muted">Classification multi-modale avec <b>LIME</b>, <b>SHAP</b> et <b>Grad-CAM</b>. Outils visualiseur: fenetre/niveau, inversion, zoom/pan et controle surcouche.</div>
    </div>""",
    unsafe_allow_html=True
)
st.write("")

@st.cache_resource(show_spinner=False)
def get_model(spec: ModelSpec):
    return load_model(spec)

def safe_get_model(spec: ModelSpec):
    try:
        m = get_model(spec)
        if not (hasattr(m, "available") and not getattr(m, "available", True)):
            used = st.session_state.get("used_models", {})
            used[spec.key] = {
                "display": spec.display,
                "kind": spec.kind,
                "modality": spec.modality.value,
            }
            st.session_state["used_models"] = used
        return m, None
    except Exception as e:
        return None, str(e)

def make_payload(modality: Modality, uploaded) -> InputPayload | None:
    if uploaded is None:
        return None
    return InputPayload(modality=modality, bytes_data=uploaded.getvalue(), filename=uploaded.name, mime=uploaded.type or "")

def image_predict_fn(model, imgs_float: np.ndarray) -> np.ndarray:
    out = []
    for im in imgs_float:
        buf = io.BytesIO()
        Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)).save(buf, format="PNG")
        p, _ = model.predict_proba(buf.getvalue(), "image/png")
        out.append(p)
    return np.vstack(out).astype(np.float32)

def build_audio_mask_predictors(model, wav_bytes: bytes, n_segments: int):
    if hasattr(model, "processor") and getattr(model, "processor", None) is not None:
        x, sr, inputs = model.build_inputs(wav_bytes)
        T = x.shape[0]
        seg_len = int(np.ceil(T / n_segments))

        def apply_mask(mask_vec: np.ndarray) -> np.ndarray:
            y = x.copy()
            for i in range(n_segments):
                if mask_vec[i] < 0.5:
                    a = i * seg_len
                    b = min(T, (i + 1) * seg_len)
                    y[a:b] = 0.0
            return y

        def predict_proba_from_mask(mask_vec: np.ndarray) -> np.ndarray:
            y = apply_mask(mask_vec)
            inp = model.processor([y], sampling_rate=sr, return_tensors="pt", padding=True)
            inp = {kk: vv.to(model.device) for kk, vv in inp.items()}
            import torch
            with torch.inference_mode():
                lg = model.model(**inp).logits
                pr = torch.softmax(lg, dim=-1).detach().cpu().numpy()[0]
            return pr.astype(np.float32)

        def predict_target(mask_vec: np.ndarray) -> float:
            pr = predict_proba_from_mask(mask_vec)
            return float(pr[1]) if pr.shape[0] > 1 else float(pr[0])

        return x, sr, inputs, predict_proba_from_mask, predict_target, "hf"

    x, sr = read_wav_bytes(wav_bytes)
    x, sr = resample_if_needed(x, sr, AUDIO_TARGET_SR)
    T = x.shape[0]
    seg_len = int(np.ceil(T / n_segments))

    def apply_mask(mask_vec: np.ndarray) -> np.ndarray:
        y = x.copy()
        for i in range(n_segments):
            if mask_vec[i] < 0.5:
                a = i * seg_len
                b = min(T, (i + 1) * seg_len)
                y[a:b] = 0.0
        return y

    def spec_tensor_from_wave(y: np.ndarray):
        rgb = mel_spectrogram_image(y, sr)
        pil = Image.fromarray(rgb).convert("RGB")
        return model.tf(pil).unsqueeze(0).to(model.device)

    spec_inputs = spec_tensor_from_wave(x)

    def predict_proba_from_mask(mask_vec: np.ndarray) -> np.ndarray:
        import torch
        t = spec_tensor_from_wave(apply_mask(mask_vec))
        with torch.inference_mode():
            pr = torch.softmax(model.model(t), dim=-1).detach().cpu().numpy()[0]
        return pr.astype(np.float32)

    def predict_target(mask_vec: np.ndarray) -> float:
        pr = predict_proba_from_mask(mask_vec)
        return float(pr[1]) if pr.shape[0] > 1 else float(pr[0])

    return x, sr, spec_inputs, predict_proba_from_mask, predict_target, "spectro"

def label_counts_from_csv(path: Path):
    if not path.exists():
        return None
    try:
        counts = Counter()
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                counts[row.get("label", "unknown")] += 1
        return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return None

def load_training_runs():
    runs = []
    runs_root = Path("runs")
    if not runs_root.exists():
        return runs
    for m in runs_root.rglob("metrics.json"):
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("task") not in ("audio", "image"):
            continue
        data["run_dir"] = str(m.parent)
        data["run_name"] = m.parent.name
        runs.append(data)
    runs.sort(key=lambda d: d.get("run_name", ""))
    return runs

def dataset_label_from_dir(dataset_dir: str) -> str:
    if not dataset_dir:
        return ""
    norm = dataset_dir.replace("\\", "/").lower()
    if norm.endswith("data/audio") or "/data/audio" in norm:
        return "ASVspoof 2019 LA (local)"
    if norm.endswith("data/xray") or "/data/xray" in norm:
        return "Chest X-ray Pneumonia (local)"
    return dataset_dir

def smooth_1d(vec: np.ndarray, k: int=3) -> np.ndarray:
    if k is None or k < 2:
        return vec
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    pad = kk // 2
    v = np.pad(vec.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(kk, dtype=np.float32) / float(kk)
    return np.convolve(v, kernel, mode="valid")

def process_importance(vec: np.ndarray, smooth_k: int=3, clip_percentile: float=99.0) -> np.ndarray:
    v = np.abs(vec.astype(np.float32))
    v = smooth_1d(v, smooth_k)
    v = v - float(np.min(v))
    if clip_percentile is not None and clip_percentile < 100.0:
        hi = float(np.percentile(v, clip_percentile))
        if hi > 0:
            v = np.clip(v, 0.0, hi)
    mx = float(np.max(v))
    if mx > 0:
        v = v / mx
    return v

def spectrogram_overlay_from_segments(x: np.ndarray, sr: int, seg_importance: np.ndarray, cmap: str, alpha: float=0.55):
    spec = mel_spectrogram_image(x, sr)
    h, w = spec.shape[0], spec.shape[1]
    n_segments = max(1, int(seg_importance.shape[0]))
    seg_len = int(np.ceil(w / n_segments))
    col_imp = np.zeros(w, dtype=np.float32)
    for i in range(n_segments):
        a = i * seg_len
        b = min(w, (i + 1) * seg_len)
        col_imp[a:b] = float(seg_importance[i])
    col_imp = np.clip(col_imp, 0.0, 1.0)
    heat = np.repeat(col_imp[np.newaxis, :], h, axis=0)
    overlay = heatmap_overlay(spec, heat, alpha=alpha, cmap=cmap)
    return spec, overlay

def spectrogram_xai_images(x: np.ndarray, sr: int, seg_importance: np.ndarray, cmap: str, size=(640, 256)):
    base, overlay = spectrogram_overlay_from_segments(x, sr, seg_importance, cmap, alpha=0.55)
    if size:
        base = np.array(Image.fromarray(base).resize(size))
        overlay = np.array(Image.fromarray(overlay).resize(size))
    return base, overlay

def focus_mask_ellipse(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = h * 0.55
    rx, ry = w * 0.22, h * 0.42
    cx_l, cx_r = w * 0.36, w * 0.64
    dist_l = ((xx - cx_l) / rx) ** 2 + ((yy - cy) / ry) ** 2
    dist_r = ((xx - cx_r) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask_l = np.clip(1.0 - dist_l, 0.0, 1.0)
    mask_r = np.clip(1.0 - dist_r, 0.0, 1.0)
    mask = np.maximum(mask_l, mask_r)
    return np.sqrt(mask)

def pick_default_model_index(specs: list[ModelSpec], modality: Modality) -> int:
    runs = load_training_runs()
    run_map = {
        "audio": {"vgg16": "aud_vgg16", "resnet18": "aud_resnet18", "mobilenet_v3_small": "aud_mnetv3"},
        "image": {"densenet121": "img_densenet121", "alexnet": "img_alexnet"},
    }
    best = None
    best_acc = -1.0
    for r in runs:
        if r.get("task") == modality.value:
            acc = float(r.get("best_val_acc", 0.0))
            if acc > best_acc:
                best_acc = acc
                best = r
    if best:
        key = run_map.get(modality.value, {}).get(best.get("arch", ""))
        if key:
            for i, s in enumerate(specs):
                if s.key == key:
                    return i
    if modality == Modality.AUDIO:
        for key in ("w2v2_df_2", "w2v2_df_1"):
            for i, s in enumerate(specs):
                if s.key == key:
                    return i
    if modality == Modality.IMAGE:
        for i, s in enumerate(specs):
            if s.key == "img_densenet121" and s.weights_hint and Path(s.weights_hint).exists():
                return i
        for i, s in enumerate(specs):
            if s.key == "xrv_chex":
                return i
    for i, s in enumerate(specs):
        if s.kind == "custom" and s.weights_hint and Path(s.weights_hint).exists():
            return i
    return 0

DEMO_DEFAULT_FILES = {
    Modality.AUDIO: "audio_reel_01.wav",
    Modality.IMAGE: "xray_pneumonie_04.jpg",
}

PRESET_XAI_CHOICE = {
    Modality.AUDIO: DISPLAY["lime"],
    Modality.IMAGE: DISPLAY["gradcam"],
}

PRESET_PARAMS = {
    Modality.AUDIO: {
        "Graphiques interactifs (Plotly)": True,
        "Segments audio": 24,
        "Echantillons LIME": 800,
        "Evaluations SHAP": 300,
        "Seuil importance audio": 0.6,
        "Lissage heatmap": 5,
        "Clip heatmap (%)": 98.0,
        "Couleur heatmap": "turbo",
        "Lissage audio (segments)": 5,
        "Contraste (WW)": float(DEFAULT_WW),
        "Luminosite (WL)": float(DEFAULT_WL),
        "Gamma (courbe)": float(DEFAULT_GAMMA),
        "Inverser noir/blanc": False,
        "Intensite heatmap (opacite)": float(DEFAULT_OVERLAY_ALPHA),
        "Masque heatmap (seuil)": 0.0,
        "Affichage XAI": "Superposition",
        "Zoom": 1.0,
        "Decalage horizontal": 0.0,
        "Decalage vertical": 0.0,
    },
    Modality.IMAGE: {
        "Graphiques interactifs (Plotly)": True,
        "Echantillons LIME": 1200,
        "Evaluations SHAP": 300,
        "Taille patch occlusion (image)": 32,
        "Pas occlusion (image)": 12,
        "Lissage heatmap": 3,
        "Clip heatmap (%)": 99.0,
        "Couleur heatmap": "magma",
        "Contraste (WW)": 0.9,
        "Luminosite (WL)": 0.5,
        "Gamma (courbe)": 1.0,
        "Inverser noir/blanc": False,
        "Intensite heatmap (opacite)": 0.55,
        "Masque heatmap (seuil)": 0.15,
        "Affichage XAI": "Superposition",
        "Zoom": 1.0,
        "Decalage horizontal": 0.0,
        "Decalage vertical": 0.0,
    },
}

# Sidebar
with st.sidebar:
    st.markdown("### 1) Modalite")
    dataset = st.radio(
        "Type d'entree",
        ["Photo (radiographie)", "Audio (voix, WAV)"],
        index=0,
        help="Change la modalite et filtre les modeles + methodes XAI compatibles.",
    )
    st.caption("Choisissez une photo de thorax ou un audio de voix.")
    modality = Modality.IMAGE if dataset.startswith("Photo") else Modality.AUDIO

    if st.session_state.get("preset_modality") != modality.value:
        st.session_state["preset_modality"] = modality.value
        st.session_state["Source des donnees"] = "Demo"
        st.session_state.pop("Fichier demo", None)
        st.session_state.pop("Selectionner un modele", None)
        st.session_state["Methode XAI"] = PRESET_XAI_CHOICE[modality]
        for key, val in PRESET_PARAMS[modality].items():
            st.session_state[key] = val

    st.markdown("### 2) Source")
    source = st.radio(
        "Source des donnees",
        ["Televersement", "Demo"],
        index=1,
        help="Televersement = vos fichiers. Demo = fichiers exemples locaux.",
    )
    uploaded = None
    demo_path = None

    if source == "Televersement":
        if modality == Modality.AUDIO:
            st.caption("Glissez-deposez un fichier WAV ou cliquez pour choisir.")
            uploaded = st.file_uploader("Fichier audio (WAV)", type=["wav"])
        else:
            st.caption("Glissez-deposez une image PNG/JPG ou cliquez pour choisir.")
            uploaded = st.file_uploader("Image thorax (PNG/JPG)", type=["png","jpg","jpeg"])
    else:
        demo_dir = Path("data/demo")
        if modality == Modality.AUDIO:
            has_demo = any(demo_dir.glob("*.wav"))
        else:
            has_demo = any(demo_dir.glob("*.png")) or any(demo_dir.glob("*.jpg")) or any(demo_dir.glob("*.jpeg"))
        if not has_demo:
            st.info("Aucun fichier demo trouve. Utilisez le bouton pour telecharger.")
            if st.button("Telecharger la demo"):
                with st.spinner("Telechargement en cours..."):
                    try:
                        download_demo_files(force=True)
                        st.success("Fichiers demo telecharges.")
                    except Exception as e:
                        st.warning(f"Echec telechargement demo: {e}")
        if modality == Modality.AUDIO:
            demo_candidates = sorted(demo_dir.glob("*.wav"))
        else:
            demo_candidates = sorted(list(demo_dir.glob("*.png")) + list(demo_dir.glob("*.jpg")) + list(demo_dir.glob("*.jpeg")))
        if demo_candidates:
            demo_names = [p.name for p in demo_candidates]
            default_name = DEMO_DEFAULT_FILES.get(modality)
            if default_name in demo_names:
                default_idx = demo_names.index(default_name)
            else:
                default_idx = 0
            chosen_demo = st.selectbox("Fichier demo", demo_names, index=default_idx)
            demo_path = demo_dir / chosen_demo

    st.markdown("### 3) Modele")
    has_custom_audio = any(Path("weights/audio").glob("*.pth"))
    has_custom_image = any(Path("weights/image").glob("*.pth"))
    default_show_custom = has_custom_audio if modality == Modality.AUDIO else has_custom_image
    show_custom = st.checkbox(
        "Inclure les modeles entraines",
        value=default_show_custom,
    )
    specs = list_models(modality, include_custom=show_custom)
    default_idx = pick_default_model_index(specs, modality)
    if default_idx > 0:
        best_spec = specs[default_idx]
        specs = [best_spec] + [s for i, s in enumerate(specs) if i != default_idx]
    chosen_display = st.selectbox(
        "Selectionner un modele",
        [s.display for s in specs],
        index=0,
        help="Choisissez l'architecture qui fera la prediction.",
    )
    chosen_spec = next(s for s in specs if s.display == chosen_display)

    st.markdown("### 4) XAI")
    allowed = ALLOWED[modality]
    st.caption("Filtre par modalite")
    st.write(" ".join([f'<span class="pill">{DISPLAY[m]}</span>' for m in allowed]), unsafe_allow_html=True)

    use_plotly = st.checkbox("Graphiques interactifs (Plotly)", value=True, help="Plus lisible, un peu plus lourd.")
    n_segments = st.slider(
        "Segments audio",
        8,
        40,
        DEFAULT_AUDIO_SEGMENTS,
        step=2,
        help="Plus de segments = localisation plus fine mais plus lente.",
    ) if modality == Modality.AUDIO else None
    lime_samples = st.slider(
        "Echantillons LIME",
        200,
        2000,
        DEFAULT_LIME_SAMPLES,
        step=100,
        help="Plus d'echantillons = plus precis mais plus lent.",
    )
    shap_evals = st.slider(
        "Evaluations SHAP",
        50,
        600,
        DEFAULT_SHAP_EVALS,
        step=50,
        help="Plus d'evaluations = plus precis mais plus lent.",
    )
    audio_thr = st.slider(
        "Seuil importance audio",
        0.0,
        1.0,
        0.7,
        step=0.05,
        help="Masque les zones faibles dans le graphe audio.",
    ) if modality == Modality.AUDIO else None
    occ_patch = st.slider(
        "Taille patch occlusion (image)",
        8,
        96,
        48,
        step=8,
        help="Patch plus petit = plus precis mais plus lent.",
    ) if modality == Modality.IMAGE else None
    occ_stride = st.slider(
        "Pas occlusion (image)",
        8,
        64,
        16,
        step=8,
        help="Pas plus grand = plus rapide mais moins fin.",
    ) if modality == Modality.IMAGE else None

    with st.expander("Reglages XAI (avance)"):
        heat_blur = st.slider("Lissage heatmap", 0, 15, 7, step=2, help="Adoucit la carte (moins de bruit).")
        heat_clip = st.slider("Clip heatmap (%)", 90.0, 99.9, 98.5, step=0.5, help="Cache les valeurs faibles.")
        heat_cmap = st.selectbox("Couleur heatmap", ["turbo", "inferno", "magma"], index=0, help="Palette visuelle.")
        audio_smooth = st.slider(
            "Lissage audio (segments)",
            1,
            9,
            5,
            step=2,
            help="Lisse les scores d'importance audio.",
        ) if modality == Modality.AUDIO else 3

    st.markdown("### 5) Visualisation image (contraste, luminosite, heatmap)")
    ww = st.slider("Contraste (WW)", 0.1, 1.0, float(DEFAULT_WW), step=0.05, help="WW = largeur de fenetre, contraste global.")
    wl = st.slider("Luminosite (WL)", 0.0, 1.0, float(DEFAULT_WL), step=0.05, help="WL = niveau de fenetre, luminosite globale.")
    gamma = st.slider("Gamma (courbe)", 0.5, 2.5, float(DEFAULT_GAMMA), step=0.1, help="Ajuste la courbe de contraste.")
    invert = st.checkbox("Inverser noir/blanc", value=False, help="Inverse les niveaux de gris.")
    overlay_alpha = st.slider("Intensite heatmap (opacite)", 0.0, 1.0, float(DEFAULT_OVERLAY_ALPHA), step=0.05, help="Force visuelle de la heatmap.")
    overlay_thr = st.slider("Masque heatmap (seuil)", 0.0, 1.0, 0.0, step=0.05, help="Masque les zones faibles.")
    view_opts = ["Image seule", "Heatmap seule", "Surcouche XAI", "Superposition"]
    view_default = 2 if modality == Modality.IMAGE else 3
    view_mode = st.radio("Affichage XAI", view_opts, index=view_default, help="Choix d'affichage des cartes.")
    zoom = st.slider("Zoom", 1.0, 5.0, 1.0, step=0.25, help="Zoom de l'image.")
    pan_x = st.slider("Decalage horizontal", -1.0, 1.0, 0.0, step=0.05, help="Cadrage gauche/droite.")
    pan_y = st.slider("Decalage vertical", -1.0, 1.0, 0.0, step=0.05, help="Cadrage haut/bas.")

# Payload
payload = None
if source == "Televersement":
    payload = make_payload(modality, uploaded)
else:
    if demo_path and demo_path.exists():
        payload = InputPayload(modality, demo_path.read_bytes(), demo_path.name, "")

tabs = st.tabs(["Analyse", "Comparaison XAI", "Presentation", "Rapports", "A propos"])

if "last_report" not in st.session_state:
    st.session_state["last_report"] = None

with tabs[0]:
    st.markdown("## Analyser une entree")
    if payload is None:
        st.info("Choisissez Televersement ou Demo dans la barre laterale.")
    else:
        model, err = safe_get_model(chosen_spec)
        if model is None:
            st.error(f"Impossible de charger le modele: {err}")
        elif hasattr(model, "available") and not getattr(model, "available", True):
            st.warning(getattr(model, "unavailable_reason", "Modele indisponible"))
            if chosen_spec.weights_hint:
                st.code(f"Poids attendus: {chosen_spec.weights_hint}")
        else:
            left, right = st.columns([1.05, 1.0])

            # preview
            with left:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Apercu entree")
                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
                if modality == Modality.AUDIO:
                    st.audio(payload.bytes_data, format="audio/wav")
                    x, sr0 = read_wav_bytes(payload.bytes_data)
                    x, sr = resample_if_needed(x, sr0, AUDIO_TARGET_SR)
                    st.caption(f"Taux d'echantillonnage: {sr0}->{sr} Hz | Duree: {x.shape[0]/sr:.2f}s | Echantillons: {x.shape[0]}")
                    spec = mel_spectrogram_image(x, sr)
                    if use_plotly:
                        st.plotly_chart(spectrogram_plotly(spec, title="Spectrogramme log-mel"), use_container_width=True)
                    else:
                        st.image(spec, caption="Spectrogramme log-mel (visuel)", use_container_width=True)
                else:
                    pil = pil_from_bytes(payload.bytes_data)
                    base_rgb = to_uint8_rgb(pil, (512,512))
                    view = zoom_pan(apply_window_level(base_rgb, ww, wl, invert, gamma), zoom, pan_x, pan_y)
                    st.image(view, caption=f"{payload.filename} (transfos visualiseur appliquees)", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # analysis
            with right:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Prediction et explicabilite")
                st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

                xai_choice = st.selectbox("Methode XAI", [DISPLAY[m] for m in allowed], index=0)
                xai_key = {v:k for k,v in DISPLAY.items()}[xai_choice]

                b1, b2 = st.columns(2)
                run_full = b1.button("Analyser (rapport complet)", type="primary")
                run_single = b2.button("Analyser (methode choisie)")
                st.caption("Le rapport complet peut prendre du temps (surtout SHAP).")

                if run_single or run_full:
                    report_all = bool(run_full)
                    methods_to_run = allowed if report_all else [xai_key]
                    with st.spinner("Prediction..."):
                        probs, extra = model.predict_proba(payload.bytes_data, payload.mime)

                    idx = int(np.argmax(probs))
                    label = model.class_names[idx]
                    conf = float(probs[idx])

                    c1, c2 = st.columns(2)
                    c1.markdown(f'<div class="kpi"><div class="small-muted">Prediction</div><h2 style="margin:6px 0 0 0;">{label}</h2></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="kpi"><div class="small-muted">Confiance</div><h2 style="margin:6px 0 0 0;">{conf:.3f}</h2></div>', unsafe_allow_html=True)

                    fig_probs = prob_bar_chart(model.class_names, probs, "Probabilites par classe")
                    if use_plotly:
                        st.plotly_chart(prob_bar_chart_plotly(model.class_names, probs, "Probabilites par classe"), use_container_width=True)
                    else:
                        st.pyplot(fig_probs, use_container_width=True)

                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    params = dict(
                        lime_samples=int(lime_samples),
                        shap_evals=int(shap_evals),
                        audio_segments=int(n_segments) if n_segments is not None else None,
                        audio_threshold=float(audio_thr) if audio_thr is not None else None,
                        occlusion_patch=int(occ_patch) if occ_patch is not None else None,
                        occlusion_stride=int(occ_stride) if occ_stride is not None else None,
                        ww=float(ww), wl=float(wl), gamma=float(gamma), invert=bool(invert),
                        overlay_alpha=float(overlay_alpha), overlay_threshold=float(overlay_thr),
                        view_mode=str(view_mode),
                        zoom=float(zoom), pan_x=float(pan_x), pan_y=float(pan_y),
                        interactive_plots=bool(use_plotly),
                    )

                    report_obj = dict(
                        timestamp=now,
                        filename=payload.filename,
                        modality=payload.modality.value,
                        model=getattr(model, "name", chosen_spec.display),
                        label=label,
                        confidence=conf,
                        probs={model.class_names[i]: float(probs[i]) for i in range(len(probs))},
                        xai=methods_to_run if report_all else xai_key,
                        params=params,
                        extra=extra,
                    )
                    probs_png = fig_to_png_bytes(fig_probs)
                    sections = []

                    if modality == Modality.AUDIO:
                        x, sr, inputs, predict_proba_from_mask, predict_target, audio_kind = build_audio_mask_predictors(model, payload.bytes_data, n_segments)
                        thr = float(audio_thr) if audio_thr is not None else 0.65
                        spec_base = mel_spectrogram_image(x, sr)
                        spec_base_png = rgb_array_to_png_bytes(spec_base)
                        xai_results = []

                        def run_audio_method(k: str, show_ui: bool):
                            title = {
                                "lime": "LIME - importance par segment",
                                "shap": "SHAP - importance par segment",
                                "occlusion": "Occlusion - impact par segment",
                                "gradcam": "Grad-CAM - importance",
                            }.get(k, "XAI")

                            if k == "lime":
                                with st.spinner("Calcul LIME..."):
                                    res = lime_audio.explain(predict_proba_from_mask, n_segments, int(lime_samples))
                                imp = process_importance(res.weights, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            elif k == "shap":
                                with st.spinner("Calcul SHAP..."):
                                    res = shap_audio.explain(predict_target, n_segments, int(shap_evals))
                                imp = process_importance(res.values, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            elif k == "occlusion":
                                with st.spinner("Calcul occlusion..."):
                                    res = occlusion_audio.explain(predict_target, n_segments)
                                imp = process_importance(res.deltas, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            else:
                                if audio_kind == "hf":
                                    conv = find_last_conv(model.model, 1)
                                    if conv is None:
                                        st.warning("Grad-CAM indisponible (pas de Conv1d).")
                                        return None
                                    with st.spinner("Calcul Grad-CAM 1D..."):
                                        res = gradcam_1d(model.model, inputs, target_index=idx, conv=conv)
                                    bins = np.array_split(res.cam, n_segments)
                                    imp_raw = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
                                    imp = process_importance(imp_raw, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                                else:
                                    conv = find_last_conv(model.model, 2)
                                    if conv is None:
                                        st.warning("Grad-CAM indisponible (pas de Conv2d).")
                                        return None
                                    with st.spinner("Calcul Grad-CAM 2D (spectrogramme)..."):
                                        res = gradcam_2d(model.model, inputs, target_index=idx, conv=conv)
                                    heat = res.heat.astype(np.float32)
                                    imp_raw = np.mean(heat, axis=0)
                                    bins = np.array_split(imp_raw, n_segments)
                                    imp_raw = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
                                    imp = process_importance(imp_raw, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))

                            fig = audio_importance_plot(x, sr, imp, title, threshold=thr)
                            if show_ui:
                                st.markdown(f"#### {DISPLAY[k]}")
                                spec_base_local, spec_overlay = spectrogram_xai_images(x, sr, imp, heat_cmap)
                                st.image(spec_overlay, caption="Spectrogramme + XAI", use_container_width=True)
                                if use_plotly:
                                    fig_p = audio_importance_plotly(x, sr, imp, title, threshold=thr)
                                    st.plotly_chart(fig_p, use_container_width=True)
                                else:
                                    st.pyplot(fig, use_container_width=True)
                            spec_base_local, spec_overlay = spectrogram_xai_images(x, sr, imp, heat_cmap)
                            return {
                                "method": k,
                                "spec_overlay_png": rgb_array_to_png_bytes(spec_overlay),
                                "wave_png": fig_to_png_bytes(fig),
                            }

                        for k in methods_to_run:
                            show_ui = report_all or k == xai_key
                            res = run_audio_method(k, show_ui)
                            if res:
                                xai_results.append(res)

                        sections = [
                            {"heading":"Entree", "text":f"Fichier: {payload.filename} | audio", "images":[spec_base_png]},
                            {"heading":"Prediction", "text":f"Modele: {report_obj['model']}<br/>Label: <b>{label}</b> | Confiance: {conf:.3f}", "images":[probs_png]},
                        ]
                        for r in xai_results:
                            sections.append({
                                "heading": f"Explicabilite - {DISPLAY[r['method']]}",
                                "text": "Spectrogramme colore + importance temporelle.",
                                "images": [r["spec_overlay_png"], r["wave_png"]],
                            })
                        sections.append({"heading":"Parametres", "text":"<br/>".join([f"{k}: {v}" for k,v in params.items()]), "images":[]})
                        zip_items = [
                            ("audio_spectrogramme_base.png", spec_base_png),
                            ("prediction_probs.png", probs_png),
                        ]
                        for r in xai_results:
                            zip_items.append((f"{r['method']}_spectrogramme.png", r["spec_overlay_png"]))
                            zip_items.append((f"{r['method']}_importance.png", r["wave_png"]))

                    else:
                        pil, x_tensor = model.build_tensor(payload.bytes_data)
                        base_rgb = to_uint8_rgb(pil, (512,512))
                        base_view = zoom_pan(apply_window_level(base_rgb, ww, wl, invert, gamma), zoom, pan_x, pan_y)

                        base224 = to_uint8_rgb(pil, (224,224))
                        img_float = base224.astype(np.float32)/255.0
                        base_png = rgb_array_to_png_bytes(base_view)

                        def predict_fn(imgs_float: np.ndarray) -> np.ndarray:
                            return image_predict_fn(model, imgs_float)

                        xai_results = []

                        def run_image_method(k: str, show_ui: bool):
                            heat224 = None
                            if k == "lime":
                                with st.spinner("Calcul LIME..."):
                                    res = lime_image.explain(base224, predict_fn, int(lime_samples))
                                heat224 = res.mask.astype(np.float32)
                            elif k == "shap":
                                with st.spinner("Calcul SHAP..."):
                                    res = shap_image.explain(img_float, predict_fn, int(shap_evals))
                                heat224 = res.heat.astype(np.float32)
                            elif k == "occlusion":
                                with st.spinner("Calcul occlusion..."):
                                    res = occlusion_image.explain(
                                        img_float,
                                        predict_fn,
                                        target_index=idx,
                                        patch=int(occ_patch or 32),
                                        stride=int(occ_stride or 16),
                                    )
                                heat224 = res.heat.astype(np.float32)
                            else:
                                if hasattr(model, "pathologies") and getattr(model, "pathologies", None):
                                    target_name = getattr(model, "proxy_keys", ["Mass"])[0]
                                    if target_name in getattr(model, "pathologies", []):
                                        target_index = model.pathologies.index(target_name)
                                    else:
                                        target_index = 0
                                    target_label = target_name
                                else:
                                    target_index = idx
                                    target_label = model.class_names[idx] if hasattr(model, "class_names") else f"class_{idx}"
                                conv = find_last_conv(model.model, 2)
                                if conv is None:
                                    st.warning("Grad-CAM indisponible (pas de Conv2d).")
                                    return None
                                with st.spinner("Calcul Grad-CAM..."):
                                    res = gradcam_2d(model.model, x_tensor, target_index=target_index, conv=conv)
                                heat224 = res.heat.astype(np.float32)

                            heat224 = np.maximum(heat224, 0.0)
                            heat224 = smooth_heatmap(heat224, int(heat_blur))
                            heat224 = normalize_heatmap(heat224, float(heat_clip))
                            if source == "Demo":
                                h, w = heat224.shape[:2]
                                mask = focus_mask_ellipse(h, w)
                                mask = smooth_heatmap(mask, 9)
                                mask = mask / (float(mask.max()) + 1e-9)
                                heat224 = heat224 * (0.4 + 0.6 * mask)

                            heat512 = np.array(Image.fromarray((np.clip(heat224, 0, 1) * 255).astype(np.uint8)).resize((512,512))) / 255.0
                            heat_color = heatmap_color(heat512, heat_cmap)
                            heat_view = zoom_pan(heat_color, zoom, pan_x, pan_y)
                            overlay_view = blend_overlay_rgb(base_view, heat_view, overlay_alpha)
                            if overlay_thr > 0:
                                mask = heat512 < overlay_thr
                                overlay_view = overlay_view.copy()
                                overlay_view[mask] = base_view[mask]

                            blended = overlay_view

                            if show_ui:
                                st.markdown(f"#### {DISPLAY[k]}")
                                if view_mode == "Image seule":
                                    st.image(base_view, caption="Image de base", use_container_width=True)
                                elif view_mode == "Heatmap seule":
                                    st.image(heat_view, caption="Heatmap XAI", use_container_width=True)
                                elif view_mode == "Surcouche XAI":
                                    st.image(overlay_view, caption="Surcouche XAI", use_container_width=True)
                                else:
                                    colA, colB = st.columns(2)
                                    colA.image(base_view, caption="Image de base", use_container_width=True)
                                    colB.image(blended, caption=f"Superposition (alpha={overlay_alpha:.2f})", use_container_width=True)

                            return {
                                "method": k,
                                "heat_png": rgb_array_to_png_bytes(heat_view),
                                "overlay_png": rgb_array_to_png_bytes(overlay_view),
                                "blended_png": rgb_array_to_png_bytes(blended),
                            }

                        for k in methods_to_run:
                            show_ui = report_all or k == xai_key
                            res = run_image_method(k, show_ui)
                            if res:
                                xai_results.append(res)

                        sections = [
                            {"heading":"Entree", "text":f"Fichier: {payload.filename} | image", "images":[base_png]},
                            {"heading":"Prediction", "text":f"Modele: {report_obj['model']}<br/>Label: <b>{label}</b> | Confiance: {conf:.3f}", "images":[probs_png]},
                        ]
                        for r in xai_results:
                            sections.append({
                                "heading": f"Explicabilite - {DISPLAY[r['method']]}",
                                "text": "Zones mises en evidence par la methode.",
                                "images": [r["heat_png"], r["blended_png"]],
                            })
                        sections.append({"heading":"Parametres", "text":"<br/>".join([f"{k}: {v}" for k,v in params.items()]), "images":[]})
                        zip_items = [
                            ("image_base.png", base_png),
                            ("prediction_probs.png", probs_png),
                        ]
                        for r in xai_results:
                            zip_items.append((f"{r['method']}_heatmap.png", r["heat_png"]))
                            zip_items.append((f"{r['method']}_overlay.png", r["overlay_png"]))
                            zip_items.append((f"{r['method']}_superposition.png", r["blended_png"]))

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for name, content in zip_items:
                            if content is None:
                                continue
                            zf.writestr(name, content)
                    zip_buf.seek(0)
                    zip_name = f"xai_{Path(payload.filename).stem}.zip"
                    st.download_button("Telecharger images XAI (ZIP)", zip_buf.getvalue(), zip_name, "application/zip")

                    xai_meta = "toutes" if report_all else DISPLAY.get(xai_key, xai_key)
                    pdf_bytes = build_pdf_report(
                        title=f"Rapport XAI - {payload.filename}",
                        meta={"Horodatage": now, "Modalite": payload.modality.value, "Modele": report_obj["model"], "XAI": xai_meta},
                        sections=sections,
                    )
                    st.download_button("Telecharger rapport complet (PDF)", pdf_bytes, f"report_{payload.filename}.pdf", "application/pdf")
                    st.session_state["last_report"] = {"obj": report_obj, "pdf": pdf_bytes}

                st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown("## Comparer les methodes XAI")
    if payload is None:
        st.info("Choisissez Televersement ou Demo dans la barre laterale.")
    else:
        model, err = safe_get_model(chosen_spec)
        if model is None:
            st.error(f"Impossible de charger le modele: {err}")
        elif hasattr(model, "available") and not getattr(model, "available", True):
            st.warning(getattr(model, "unavailable_reason", "Modele indisponible"))
        else:
            methods = st.multiselect("Methodes", [DISPLAY[m] for m in ALLOWED[modality]],
                                     default=[DISPLAY[m] for m in ALLOWED[modality]])
            keys = [{v:k for k,v in DISPLAY.items()}[m] for m in methods]
            if st.button("Lancer comparaison", type="primary"):
                probs, _ = model.predict_proba(payload.bytes_data, payload.mime)
                idx = int(np.argmax(probs))
                label = model.class_names[idx]
                conf = float(probs[idx])
                st.success(f"Prediction: **{label}** (confiance {conf:.3f})")

                if modality == Modality.AUDIO:
                    x, sr, inputs, predict_proba_from_mask, predict_target, audio_kind = build_audio_mask_predictors(model, payload.bytes_data, n_segments)
                    thr = float(audio_thr) if audio_thr is not None else 0.65
                else:
                    pil, x_tensor = model.build_tensor(payload.bytes_data)
                    base_rgb = to_uint8_rgb(pil, (512,512))
                    base_view = zoom_pan(apply_window_level(base_rgb, ww, wl, invert, gamma), zoom, pan_x, pan_y)
                    base224 = to_uint8_rgb(pil, (224,224))
                    img_float = base224.astype(np.float32)/255.0

                    def predict_fn(imgs_float: np.ndarray) -> np.ndarray:
                        return image_predict_fn(model, imgs_float)

                for k in keys:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"### {DISPLAY[k]}")
                    if modality == Modality.AUDIO:
                        if k == "lime":
                            res = lime_audio.explain(predict_proba_from_mask, n_segments, int(lime_samples))
                            imp = process_importance(res.weights, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            title = "LIME"
                        elif k == "shap":
                            res = shap_audio.explain(predict_target, n_segments, int(shap_evals))
                            imp = process_importance(res.values, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            title = "SHAP"
                        elif k == "occlusion":
                            res = occlusion_audio.explain(predict_target, n_segments)
                            imp = process_importance(res.deltas, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            title = "Occlusion"
                        else:
                            if audio_kind == "hf":
                                conv = find_last_conv(model.model, 1)
                                if conv is None:
                                    st.warning("Pas de Conv1d.")
                                    imp = None
                                else:
                                    res = gradcam_1d(model.model, inputs, target_index=idx, conv=conv)
                                    bins = np.array_split(res.cam, n_segments)
                                    imp_raw = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
                                    imp = process_importance(imp_raw, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            else:
                                conv = find_last_conv(model.model, 2)
                                if conv is None:
                                    st.warning("Pas de Conv2d.")
                                    imp = None
                                else:
                                    res = gradcam_2d(model.model, inputs, target_index=idx, conv=conv)
                                    heat = res.heat.astype(np.float32)
                                    imp_raw = np.mean(heat, axis=0)
                                    bins = np.array_split(imp_raw, n_segments)
                                    imp_raw = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
                                    imp = process_importance(imp_raw, smooth_k=int(audio_smooth), clip_percentile=float(heat_clip))
                            title = "Grad-CAM"

                        if imp is not None:
                            c1, c2 = st.columns([1.05, 1.0])
                            spec_base, spec_overlay = spectrogram_xai_images(x, sr, imp, heat_cmap)
                            c1.image(spec_overlay, caption="Spectrogramme + XAI", use_container_width=True)
                            fig = audio_importance_plot(x, sr, imp, title, threshold=thr)
                            if use_plotly:
                                fig_p = audio_importance_plotly(x, sr, imp, title, threshold=thr)
                                c2.plotly_chart(fig_p, use_container_width=True)
                            else:
                                c2.pyplot(fig, use_container_width=True)
                    else:
                        heat224 = None
                        if k == "lime":
                            res = lime_image.explain(base224, predict_fn, int(lime_samples))
                            heat224 = res.mask.astype(np.float32)
                        elif k == "shap":
                            res = shap_image.explain(img_float, predict_fn, int(shap_evals))
                            heat224 = res.heat.astype(np.float32)
                        elif k == "occlusion":
                            res = occlusion_image.explain(
                                img_float,
                                predict_fn,
                                target_index=idx,
                                patch=int(occ_patch or 32),
                                stride=int(occ_stride or 16),
                            )
                            heat224 = res.heat.astype(np.float32)
                        else:
                            target_name = getattr(model, "proxy_keys", ["Mass"])[0]
                            target_index = model.pathologies.index(target_name) if hasattr(model, "pathologies") and target_name in model.pathologies else 0
                            conv = find_last_conv(model.model, 2)
                            if conv is None:
                                heat224 = None
                            else:
                                res = gradcam_2d(model.model, x_tensor, target_index=target_index, conv=conv)
                                heat224 = res.heat.astype(np.float32)

                        heat_view = None
                        overlay_view = base_view
                        if heat224 is not None:
                            heat224 = np.maximum(heat224, 0.0)
                            heat224 = smooth_heatmap(heat224, int(heat_blur))
                            heat224 = normalize_heatmap(heat224, float(heat_clip))

                            heat512 = np.array(Image.fromarray((np.clip(heat224, 0, 1) * 255).astype(np.uint8)).resize((512,512))) / 255.0
                            heat_color = heatmap_color(heat512, heat_cmap)
                            heat_view = zoom_pan(heat_color, zoom, pan_x, pan_y)
                            overlay_view = blend_overlay_rgb(base_view, heat_view, overlay_alpha)
                            if overlay_thr > 0:
                                mask = heat512 < overlay_thr
                                overlay_view = overlay_view.copy()
                                overlay_view[mask] = base_view[mask]
                        blended = overlay_view

                        c1, c2, c3 = st.columns(3)
                        c1.image(base_view, caption="Base", use_container_width=True)
                        if heat_view is None:
                            c2.info("Heatmap indisponible.")
                        else:
                            c2.image(heat_view, caption="Heatmap", use_container_width=True)
                        c3.image(blended, caption="Superposition", use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("## Presentation")
    st.caption("Resume simple des modeles utilises et de l'entrainement local.")

    st.markdown("### Modeles utilises dans cette session")
    used = st.session_state.get("used_models", {})
    if used:
        rows = []
        for key, info in used.items():
            rows.append([info.get("display"), info.get("modality"), info.get("kind")])
        st.table([["modele","modalite","type"]] + rows)
    else:
        st.info("Aucun modele encore charge dans cette session.")

    st.markdown("### Modeles entraines localement (poids trouves)")
    audio_custom = [s for s in list_models(Modality.AUDIO, include_custom=True) if s.kind == "custom"]
    image_custom = [s for s in list_models(Modality.IMAGE, include_custom=True) if s.kind == "custom"]
    rows = []
    for s in audio_custom + image_custom:
        p = Path(s.weights_hint or "")
        if p.exists():
            rows.append([s.display, s.modality.value, str(p)])
    if rows:
        st.table([["modele","modalite","poids"]] + rows)
    else:
        st.info("Aucun poids custom detecte.")

    st.markdown("### Datasets utilises (sources + local)")
    rows = []
    demo_manifest = Path("data/demo/demo_pro_manifest.json")
    if demo_manifest.exists():
        try:
            demo = json.loads(demo_manifest.read_text(encoding="utf-8"))
        except Exception:
            demo = {}
        audio_info = demo.get("audio", {})
        xray_info = demo.get("xray", {})
        if audio_info:
            counts = audio_info.get("counts", {})
            rows.append([
                "demo",
                "audio",
                audio_info.get("dataset", ""),
                "data/demo",
                f"reel={counts.get('reel', 0)}, faux={counts.get('faux', 0)}",
            ])
        if xray_info:
            counts = xray_info.get("counts", {})
            rows.append([
                "demo",
                "image",
                xray_info.get("dataset", ""),
                "data/demo",
                f"normal={counts.get('normal', 0)}, pneumonie={counts.get('pneumonie', 0)}",
            ])

    runs = load_training_runs()
    for r in runs:
        train_sz = int(r.get("train_size", 0) or 0)
        val_sz = int(r.get("val_size", 0) or 0)
        total = train_sz + val_sz
        rows.append([
            "entrainement",
            r.get("task", ""),
            dataset_label_from_dir(r.get("dataset_dir", "")),
            r.get("dataset_dir", ""),
            f"total={total}",
        ])

    if rows:
        st.table([["usage","modalite","dataset/source","local","samples"]] + rows)
    else:
        st.info("Aucun dataset detecte.")

    st.markdown("### Entrainement (resume)")
    if not runs:
        st.info("Aucun entrainement detecte.")
    else:
        tbl = []
        for r in runs:
            tbl.append([
                r.get("task"),
                r.get("arch"),
                str(r.get("epochs")),
                f"{float(r.get('best_val_acc', 0.0)):.4f}",
                dataset_label_from_dir(r.get("dataset_dir", "")),
            ])
        st.table([["tache","modele","epochs","best_val_acc","dataset"]] + tbl)

with tabs[3]:
    st.markdown("## Rapports")
    st.caption("Re-telecharger le dernier rapport PDF genere dans Analyse.")
    last = st.session_state.get("last_report")
    if not last:
        st.info("Lancez une analyse d'abord.")
    else:
        st.json(last["obj"])
        ro = last["obj"]
        st.download_button("Telecharger le dernier rapport PDF", last["pdf"], f"report_{ro['filename']}.pdf", "application/pdf")

with tabs[4]:
    st.markdown("## A propos")
    st.markdown(
        "Cette interface unifie l'analyse XAI audio et image dans un meme workflow. "
        "Vous importez un fichier (voix WAV ou radiographie), vous choisissez un modele, "
        "puis vous appliquez des methodes d'explicabilite (Grad-CAM, LIME, SHAP, Occlusion). "
        "Les resultats sont affiches avec les visualisations cote a cote, "
        "avec un rapport PDF telechargeable pour documenter la prediction et les heatmaps."
    )
