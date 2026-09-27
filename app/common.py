"""Shared UI layer: design tokens, layout helpers, chart template, data access."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import FIGURES, MODELS, RESULTS  # noqa: E402

# --------------------------------------------------------------------------
# Design tokens (one set per theme; the active theme comes from the browser)
# --------------------------------------------------------------------------
THEMES = {
    "dark": dict(accent="#818cf8", text="#e6e8ef", body="#c9cdd8", muted="#8b93a7",
                 border="#262a36", surface="#171a23", bg="#0f1117", neutral="#4b5263", grid="rgba(139,147,167,.12)",
                 unknown="#e6e8ef", note_bg="rgba(250,204,21,.08)", note_bd="rgba(250,204,21,.28)", note_fg="#fde68a"),
    "light": dict(accent="#4f46e5", text="#1e2330", body="#394052", muted="#6b7385",
                  border="#e3e6ee", surface="#ffffff", bg="#f6f7fb", neutral="#b6bccb", grid="rgba(30,35,48,.08)",
                  unknown="#334155", note_bg="#fffbeb", note_bd="#fde68a", note_fg="#92400e"),
}
_CLASS_COLORS = {  # one fixed colour per class, used everywhere
    "Benign": "#10b981", "DDoS": "#ef4444", "DoS": "#f97316", "Recon": "#eab308",
    "Web-based": "#8b5cf6", "BruteForce": "#ec4899", "Spoofing": "#06b6d4", "Mirai": "#b45309",
}
SEV_COLORS = {"None": "#10b981", "Low": "#eab308", "Medium": "#f97316", "High": "#ef4444", "Critical": "#dc2626"}


def theme() -> str:
    """'light' or 'dark': whatever the viewer's browser is currently showing."""
    try:
        t = (st.context.theme or {}).get("type")
    except Exception:
        t = None
    return t if t in THEMES else "dark"


def T() -> dict:
    return THEMES[theme()]


def colors() -> dict:
    return {**_CLASS_COLORS, "Unknown attack": T()["unknown"]}



CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3, h4 {letter-spacing: -.01em;}
[data-testid="stSidebarNav"] {padding-top: .5rem;}
.page-head {margin-bottom: 1.4rem; padding-bottom: 1rem; border-bottom: 1px solid var(--x-border);}
.page-head h1 {font-size: 1.75rem; font-weight: 700; margin: 0 0 .25rem 0; padding: 0; color: var(--x-text);}
.page-head p {color: var(--x-muted); margin: 0; font-size: .98rem;}
.section {font-size: 1.02rem; font-weight: 650; margin: 1.6rem 0 .6rem 0; color: var(--x-text);}
.section span {color: var(--x-muted); font-weight: 400; font-size: .88rem; margin-left: .5rem;}
.kpis {display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 12px; margin: .2rem 0 .4rem 0;}
@media (max-width: 700px) { .kpis {grid-template-columns: repeat(2, minmax(0, 1fr));} }
.kpi {background: var(--x-surface); border: 1px solid var(--x-border); border-top: 3px solid var(--x-accent);
      border-radius: 12px; padding: 14px 16px; box-shadow: var(--x-shadow);}
.kpi .l {color: var(--x-muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.kpi .v {color: var(--x-text); font-size: clamp(1.05rem, 1.9vw, 1.55rem); font-weight: 700; margin-top: 4px;
         line-height: 1.2; white-space: nowrap;}
.kpi .s {color: var(--x-muted); font-size: .8rem; margin-top: 2px;}
.card {background: var(--x-surface); border: 1px solid var(--x-border); border-radius: 12px; padding: 18px 20px;
       height: 100%; box-shadow: var(--x-shadow);}
.card h4 {margin: 0 0 .5rem 0; font-size: 1rem; color: var(--x-text);}
.card p, .card li {color: var(--x-body); font-size: .93rem; line-height: 1.55;}
.muted {color: var(--x-muted); font-size: .88rem;}
.tag {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
      border: 1px solid currentColor; margin-right: 6px;}
.verdict {background: var(--x-surface); border: 1px solid var(--x-border); border-left: 4px solid var(--c);
          border-radius: 12px; padding: 16px 20px; margin-bottom: 1rem; box-shadow: var(--x-shadow);}
.verdict .t {font-size: 1.35rem; font-weight: 700; color: var(--c);}
.verdict .body {color: var(--x-text); line-height: 1.6;}
.verdict b {color: var(--x-text);}
.note {background: var(--x-note-bg); border: 1px solid var(--x-note-bd); color: var(--x-note-fg);
       border-radius: 10px; padding: 10px 14px; font-size: .88rem; margin: .8rem 0;}
[data-testid="stSidebar"] .status {font-size: .8rem; color: var(--x-muted); line-height: 1.5; padding: 0 .5rem;}
[data-testid="stSidebar"] .status b {color: var(--x-text);}
.dot {display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; background: var(--c);}
</style>
"""


def _css_vars() -> str:
    t = T()
    shadow = "0 1px 2px rgba(16,24,40,.05), 0 1px 3px rgba(16,24,40,.07)" if theme() == "light" else "none"
    v = {"accent": t["accent"], "text": t["text"], "body": t["body"], "muted": t["muted"], "border": t["border"],
         "surface": t["surface"], "note-bg": t["note_bg"], "note-bd": t["note_bd"], "note-fg": t["note_fg"],
         "shadow": shadow}
    return "<style>:root{" + "".join(f"--x-{k}:{val};" for k, val in v.items()) + "}</style>"


# --------------------------------------------------------------------------
# Plotly templates (one per theme)
# --------------------------------------------------------------------------
for _name, _t in THEMES.items():
    _ax = dict(automargin=True, gridcolor=_t["grid"], zerolinecolor=_t["grid"], linecolor=_t["grid"])
    pio.templates[f"explainids_{_name}"] = go.layout.Template(layout=dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_t["body"], size=12), title=dict(font=dict(size=14, color=_t["text"]), x=0, xanchor="left"),
        colorway=list(_CLASS_COLORS.values()), xaxis=_ax, yaxis=_ax,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=40, b=10, l=10, r=10),
        hoverlabel=dict(bgcolor=_t["surface"], bordercolor=_t["border"], font_color=_t["text"]),
    ))


def chart(fig, height=340, **layout):
    fig.update_layout(template=f"explainids_{theme()}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=height, **layout)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def theme_toggle():
    """Sidebar Light/Dark switch. Streamlit keeps the chosen theme in localStorage
    ("stActiveTheme-<path>-v2"), so we store it for every page and reload."""
    current = theme()
    with st.sidebar:
        choice = st.segmented_control("Theme", ["Light", "Dark"], default=current.capitalize(),
                                      key="_theme_choice", label_visibility="collapsed", width="stretch",
                                      format_func=lambda o: (":material/light_mode: " if o == "Light" else ":material/dark_mode: ") + o)
    if choice and choice.lower() != current:
        paths = ["/", "/home", "/detect", "/explain", "/live", "/models", "/zero_day", "/robustness", "/report"]
        js = "".join(f"localStorage.setItem('stActiveTheme-{p}-v2', JSON.stringify('{choice}'));" for p in paths)
        st.html(f"<script>{js} window.parent.location.reload();</script>", unsafe_allow_javascript=True)


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------
def setup():
    """Called once by the entry point."""
    st.set_page_config(page_title="ExplainIDS", page_icon=":material/shield:", layout="wide")
    st.markdown(CSS + _css_vars(), unsafe_allow_html=True)


def header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, hint: str = ""):
    st.markdown(f'<div class="section">{title}{f"<span>{hint}</span>" if hint else ""}</div>', unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, sub) tuples."""
    html = "".join(f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div>'
                   f'{f"<div class=s>{s}</div>" if s else ""}</div>' for l, v, s in items)
    st.markdown(f'<div class="kpis" style="--n:{len(items)}">{html}</div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def tag(text: str, color: str) -> str:
    return f'<span class="tag" style="color:{color}">{text}</span>'


def sidebar_status():
    d = deploy_info()
    with st.sidebar:
        st.divider()
        if d:
            src = "Synthetic sample" if "Synthetic" in d.get("source", "") else "CIC IoT-DIAD 2024"
            st.markdown(f'<div class="status"><span class="dot" style="--c:#34d399"></span><b>Model ready</b><br>'
                        f'{d["model_name"]} · {d["trained_at"]}<br>Data: {src}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status"><span class="dot" style="--c:#ef4444"></span><b>No model</b><br>'
                        'Run <code>python src/train.py</code></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
def deploy_info():
    p = MODELS / "deploy.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def is_synthetic():
    m = load_json("metrics.json")
    return bool(m and "Synthetic" in m["dataset"]["source"])


@st.cache_resource(show_spinner="Loading models…")
def get_detector():
    from detector import Detector
    return Detector()


def require_models():
    if not deploy_info():
        st.error("Models are not trained yet. Run `python src/train.py` first (see README).")
        st.stop()
    return get_detector()


def load_demo_if_requested():
    """`?demo=1` in the URL auto-analyses the demo capture (handy for presentations)."""
    if "demo" in st.query_params and "analysis" not in st.session_state:
        from flow_extractor import extract_flows
        det = require_models()
        with st.spinner("Loading demo capture…"):
            df = extract_flows(ROOT / "data" / "sample" / "demo_mixed_traffic.pcap")
            res, ex = det.analyse(df, extras=True)
        st.session_state["analysis"] = {"name": "demo_mixed_traffic.pcap", "result": res, **ex}


def require_analysis():
    load_demo_if_requested()
    A = st.session_state.get("analysis")
    if not A:
        st.info("No traffic analysed yet. Open **Detect** and upload a capture or use a demo file.")
        st.page_link("views/detect.py", label="Go to Detect", icon=":material/arrow_forward:")
        st.stop()
    return A


def figure(name, caption=None):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.caption(f"Figure `{name}` not generated yet.")


def deployed_metrics(m):
    return m["models"]["Fusion CNN-XGBoost" if m["deployed"] == "fusion" else "Hybrid CNN-XGBoost"]
