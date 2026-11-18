"""Probability, importance and spectrogram charts."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def prob_bar_chart(class_names, probs, title="Probabilites"):
    """Matplotlib bar chart of class probabilities."""
    fig = plt.figure()
    plt.bar(class_names, probs)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("p")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig

def prob_bar_chart_plotly(class_names, probs, title="Probabilites"):
    """Plotly bar chart of class probabilities."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(x=class_names, y=probs, marker_color="#0f766e"))
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1], title="p"),
        xaxis=dict(tickangle=15),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig

def audio_importance_plot(x: np.ndarray, sr: int, seg_importance: np.ndarray, title: str, threshold: float=0.65):
    """Matplotlib chart of the importance of each audio segment."""
    T = x.shape[0]
    n_segments = seg_importance.shape[0]
    seg_len = int(np.ceil(T / n_segments))
    imp = np.zeros(T, dtype=np.float32)
    for i in range(n_segments):
        a = i * seg_len
        b = min(T, (i + 1) * seg_len)
        imp[a:b] = seg_importance[i]
    imp = imp - imp.min()
    imp = imp / (imp.max() + 1e-9)

    t = np.arange(T) / float(sr)
    fig = plt.figure()
    plt.plot(t, x, linewidth=0.8)
    plt.fill_between(t, x.min(), x.max(), where=(imp > threshold), alpha=0.25)
    plt.title(title)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    return fig

def audio_importance_plotly(x: np.ndarray, sr: int, seg_importance: np.ndarray, title: str, threshold: float=0.65):
    """Plotly chart of the importance of each audio segment."""
    import plotly.graph_objects as go
    T = x.shape[0]
    n_segments = seg_importance.shape[0]
    seg_len = int(np.ceil(T / n_segments))
    imp = np.zeros(T, dtype=np.float32)
    for i in range(n_segments):
        a = i * seg_len
        b = min(T, (i + 1) * seg_len)
        imp[a:b] = seg_importance[i]
    imp = imp - imp.min()
    imp = imp / (imp.max() + 1e-9)

    t = np.arange(T) / float(sr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(width=1, color="#111827")))

    y0 = float(np.min(x))
    y1 = float(np.max(x))
    for i in range(n_segments):
        a = i * seg_len
        b = min(T, (i + 1) * seg_len)
        score = float(np.mean(imp[a:b])) if b > a else 0.0
        if score >= threshold:
            fig.add_shape(
                type="rect",
                x0=float(t[a]),
                x1=float(t[b - 1]) if b > a else float(t[a]),
                y0=y0,
                y1=y1,
                fillcolor="rgba(15,118,110,0.20)",
                line_width=0,
                layer="below",
            )
    fig.update_layout(
        title=title,
        xaxis_title="Temps (s)",
        yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    return fig

def spectrogram_plotly(rgb_uint8: np.ndarray, title: str="Spectrogramme"):
    """Plotly heatmap of a spectrogram."""
    import plotly.express as px
    fig = px.imshow(rgb_uint8, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
