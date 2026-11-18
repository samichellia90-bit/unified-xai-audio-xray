"""Model identifiers and default explanation settings."""

from __future__ import annotations

AUDIO_TARGET_SR = 16000

AUDIO_HF_MODELS = {
    "w2v2_df_1": {"display":"Wav2Vec2 - reel vs faux (HF #1)", "model_id":"HyperMoon/wav2vec2-base-960h-finetuned-deepfake"},
    "w2v2_df_2": {"display":"Wav2Vec2 - reel vs faux (HF #2)", "model_id":"HyperMoon/wav2vec2-base-finetuned-deepfake-0919"},
}

XRAY_XRV_WEIGHTS = {
    "chex": {"display":"DenseNet121 (CheXpert) - pre-entraine", "weights":"densenet121-res224-chex"},
    "nih":  {"display":"DenseNet121 (NIH) - pre-entraine", "weights":"densenet121-res224-nih"},
}

DEFAULT_AUDIO_SEGMENTS = 32
DEFAULT_LIME_SAMPLES = 1400
DEFAULT_SHAP_EVALS = 400
DEFAULT_OVERLAY_ALPHA = 0.6

DEFAULT_WW = 0.9
DEFAULT_WL = 0.5
DEFAULT_GAMMA = 1.0
