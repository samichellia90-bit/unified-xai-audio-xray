"""CSS of the Streamlit interface."""

from __future__ import annotations
APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-1: #0a0f14;
  --bg-2: #101621;
  --panel: #141b27;
  --panel-alt: #111827;
  --ink: #f8fafc;
  --muted: #94a3b8;
  --accent: #22d3ee;
  --accent-2: #38bdf8;
  --accent-warm: #f59e0b;
  --border: #243244;
  --shadow: 0 16px 34px rgba(0, 0, 0, 0.45);
  --shadow-soft: 0 8px 18px rgba(0, 0, 0, 0.35);
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 800px at 15% -20%, rgba(34, 211, 238, 0.10), transparent 60%),
    radial-gradient(900px 700px at 90% 10%, rgba(245, 158, 11, 0.08), transparent 60%),
    linear-gradient(135deg, var(--bg-1), var(--bg-2));
  color: var(--ink) !important;
  font-family: "Manrope", sans-serif;
}

section.main { background: transparent; }

body, p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
  color: var(--ink) !important;
}

h1, h2, h3 {
  font-family: "Sora", sans-serif;
  letter-spacing: -0.01em;
}

.small-muted { opacity: 0.95; font-size: 0.95rem; color: var(--muted) !important; }
a { color: var(--accent-2) !important; text-decoration: none; }
a:hover { text-decoration: underline; }

header, #MainMenu, footer { visibility: hidden; height: 0; }
header, #MainMenu, footer { pointer-events: none; }

[data-testid="stSidebar"] {
  background: rgba(16, 22, 33, 0.98);
  border-right: 1px solid var(--border);
  box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.03);
  z-index: 10;
}

.block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; max-width: 1280px; }

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: var(--shadow);
  animation: fadeUp 0.5s ease-out;
}
.card.hero {
  background: linear-gradient(135deg, rgba(34, 211, 238, 0.14), rgba(245, 158, 11, 0.08));
}
.badge {
  display:inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(34, 211, 238, 0.35);
  background: rgba(34, 211, 238, 0.12);
  color: var(--accent) !important;
  font-weight: 700;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}
.hr { height:1px; background: var(--border); margin: 10px 0 14px 0; }
.kpi {
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 12px;
  box-shadow: var(--shadow-soft);
}
.pill {
  display:inline-block;
  padding: 4px 10px;
  border-radius:999px;
  border: 1px solid rgba(34, 211, 238, 0.35);
  background: rgba(34, 211, 238, 0.12);
  color: var(--accent) !important;
  font-size: 0.82rem;
  font-weight: 600;
  margin-right: 6px;
  margin-bottom: 6px;
}

input, textarea, select {
  color: var(--ink) !important;
  background: #0c131d !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: #0c131d !important;
}

div[data-baseweb="select"] > div {
  background: #0c131d !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] span, div[data-baseweb="select"] input {
  color: var(--ink) !important;
}
div[data-baseweb="select"] svg { fill: var(--muted) !important; }

div[data-testid="stFileUploader"] {
  background: #0c131d;
  border: 1px dashed #334155;
  border-radius: 14px;
  padding: 10px;
}
div[data-testid="stFileUploader"] * { color: var(--ink) !important; }

.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  color: #081018 !important;
  border: none;
  border-radius: 999px;
  padding: 0.48rem 1.2rem;
  font-weight: 700;
  box-shadow: var(--shadow-soft);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow);
}

.stTabs [role="tablist"] { gap: 0.4rem; }
.stTabs [role="tab"] {
  border-radius: 999px;
  padding: 0.35rem 0.9rem;
  border: 1px solid var(--border);
  background: rgba(14, 20, 30, 0.7);
  color: var(--muted) !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(34, 211, 238, 0.18);
  border-color: rgba(34, 211, 238, 0.45);
  color: var(--ink) !important;
}

code, pre, .stCode {
  font-family: "JetBrains Mono", "Consolas", monospace;
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""
