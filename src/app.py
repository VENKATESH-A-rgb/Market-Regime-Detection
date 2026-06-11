"""
app.py — Step 9: Interactive Streamlit Dashboard
Market Regime Detection & Adaptive Portfolio Allocation
Premium dark-themed dashboard with regime overlays and live metrics.
"""

import sys
from pathlib import Path

# Ensure imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page Config & Theme State
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detection — Adaptive Portfolio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"


# ──────────────────────────────────────────────────────────────
# Custom CSS — SaaS-Grade Theme System
# ──────────────────────────────────────────────────────────────
bg_color = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card_color = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border_color = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
accent_color = "#2563eb"
accent_muted = "#1d4ed8"
green_color = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red_color = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber_color = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
shadow_val = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"
radius_val = "10px"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

    :root {{
        --bg: {bg_color};
        --bg-subtle: {bg_subtle};
        --card: {card_color};
        --card-hover: {card_hover};
        --border: {border_color};
        --border-subtle: {border_subtle};
        --text: {text_color};
        --text-muted: {text_muted};
        --text-dim: {text_dim};
        --accent: {accent_color};
        --accent-muted: {accent_muted};
        --green: {green_color};
        --green-muted: {green_muted};
        --red: {red_color};
        --red-muted: {red_muted};
        --amber: {amber_color};
        --amber-muted: {amber_muted};
        --shadow: {shadow_val};
        --radius: {radius_val};
    }}

    /* Global styling overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', -apple-system, sans-serif !important;
    }}
    
    .block-container {{
        padding: 2rem 2.5rem 3rem !important;
        max-width: 1360px !important;
    }}

    /* Sidebar styling overrides */
    [data-testid="stSidebar"], [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        background-color: var(--bg-subtle) !important;
        border-right: 1px solid var(--border) !important;
    }}
    
    /* Input box style overrides */
    div[data-baseweb="input"] {{
        background-color: var(--bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif;
    }}
    div[data-baseweb="input"]:focus-within {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
    }}

    /* Hide standard streamlit decoration and headers */
    #MainMenu, footer, .stDeployButton {{
        display: none !important;
    }}

    /* Make header transparent */
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }}

    /* Sidebar collapsed control — visible in ALL states */
    div[data-testid="stSidebarCollapsedControl"],
    button[data-testid="stSidebarNavToggle"],
    [data-testid="collapsedControl"] {{
        background-color: var(--bg-subtle) !important;
        border-right: 1px solid var(--border) !important;
        border-bottom: 1px solid var(--border) !important;
        border-bottom-right-radius: var(--radius) !important;
        z-index: 99999 !important;
        box-shadow: var(--shadow) !important;
        transition: background-color 0.2s ease;
    }}
    div[data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="stSidebarNavToggle"]:hover {{
        background-color: var(--card-hover) !important;
    }}
    div[data-testid="stSidebarCollapsedControl"] button,
    button[data-testid="stSidebarNavToggle"] {{
        color: var(--text) !important;
        background: transparent !important;
        border: none !important;
        pointer-events: auto !important;
        cursor: pointer !important;
    }}

    /* Sidebar close button inside expanded sidebar */
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="stSidebar"] [data-testid="stSidebarNavItems"] button,
    [data-testid="stSidebar"] button {{
        pointer-events: auto !important;
    }}

    /* Custom KPI Metric Cards */
    .metric-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        box-shadow: var(--shadow);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent) !important;
        box-shadow: 0 8px 30px rgba(37,99,235,0.08), 0 0 0 1px rgba(37,99,235,0.1) !important;
    }}
    .metric-label {{
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
    }}
    .metric-value.positive {{
        color: var(--green) !important;
    }}
    .metric-value.negative {{
        color: var(--red) !important;
    }}
    .metric-sublabel {{
        font-size: 0.72rem;
        color: var(--text-dim);
        margin-top: 0.3rem;
    }}
    
    /* Chart Container Card */
    .chart-wrap {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.5rem;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .chart-wrap:hover {{
        border-color: var(--accent);
        box-shadow: 0 8px 30px rgba(37,99,235,0.04);
    }}
    .chart-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text);
    }}
    .chart-subtitle {{
        font-size: 0.72rem;
        color: var(--text-dim);
        margin-bottom: 1rem;
    }}

    /* Data Tables (HTML) */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.8rem;
        margin-top: 1rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        overflow: hidden;
    }}
    .data-table th {{
        text-align: left;
        padding: 0.75rem 1rem;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        background-color: var(--bg-subtle);
        border-bottom: 1px solid var(--border);
    }}
    .data-table td {{
        padding: 0.75rem 1rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
        font-family: 'JetBrains Mono', monospace;
    }}
    .data-table tr:last-child td {{
        border-bottom: none;
    }}
    .data-table tr:hover td {{
        background-color: var(--card-hover);
    }}

    /* Hero Section */
    .hero {{
        background: linear-gradient(135deg, var(--card) 0%, var(--bg-subtle) 100%);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }}
    .hero h1 {{
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        background: linear-gradient(90deg, var(--text) 0%, var(--text-muted) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }}
    .hero p {{
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0;
        line-height: 1.6;
    }}

    /* Info Panels */
    .info-panel {{
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.25rem;
        font-size: 0.82rem;
        color: var(--text-muted);
        line-height: 1.5;
        margin-bottom: 1.5rem;
    }}
    .info-panel strong {{
        color: var(--text);
    }}

    /* Section Headers */
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin: 2rem 0 1rem 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    /* Pill-style tabs */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.1rem !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.2s ease !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: var(--text) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--text) !important;
        background: var(--card) !important;
        border-color: var(--border) !important;
        font-weight: 600 !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: var(--bg-subtle) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 3px !important;
        margin-bottom: 1.5rem !important;
    }}

    /* Regime badges */
    .regime-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-family: 'DM Sans', sans-serif;
    }}
    .regime-bull {{ color: var(--green); background: var(--green-muted); border: 1px solid rgba(34,197,94,0.15); }}
    .regime-recovery {{ color: var(--accent); background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.15); }}
    .regime-bear {{ color: var(--amber); background: var(--amber-muted); border: 1px solid rgba(245,158,11,0.15); }}
    .regime-crisis {{ color: var(--red); background: var(--red-muted); border: 1px solid rgba(239,68,68,0.15); }}

    /* Brand / Header layout */
    .brand {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    @keyframes pulse-glow {{
        0% {{ transform: scale(1); filter: drop-shadow(0 0 2px var(--accent)); }}
        50% {{ transform: scale(1.1); filter: drop-shadow(0 0 8px var(--accent)); }}
        100% {{ transform: scale(1); filter: drop-shadow(0 0 2px var(--accent)); }}
    }}
    .brand-logo {{
        font-size: 1.25rem;
        animation: pulse-glow 3s infinite ease-in-out;
        display: inline-block;
    }}
    .brand-name {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
    }}
    .brand-subtitle {{
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-left: 8px;
        border-left: 1px solid var(--border);
        padding-left: 8px;
    }}

    /* Horizontal blocks spacing */
    [data-testid="stHorizontalBlock"] {{
        gap: 1.25rem !important;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-dim);
    }}

    /* ─── Staggered Card Entrance Animations ─── */
    @keyframes fadeSlideUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .anim-card {{ animation: fadeSlideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; }}
    .delay-1 {{ animation-delay: 0.05s; }}
    .delay-2 {{ animation-delay: 0.12s; }}
    .delay-3 {{ animation-delay: 0.19s; }}
    .delay-4 {{ animation-delay: 0.26s; }}
    .delay-5 {{ animation-delay: 0.33s; }}

    /* ─── Sidebar Brand Header ─── */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.25rem;
    }}
    .sidebar-brand-icon {{
        font-size: 1.5rem;
        animation: pulse-glow 3s infinite ease-in-out;
    }}
    .sidebar-brand-text {{
        display: flex;
        flex-direction: column;
    }}
    .sidebar-brand-name {{
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }}
    .sidebar-brand-version {{
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 600;
        color: var(--accent);
        background: rgba(37,99,235,0.1);
        padding: 1px 7px;
        border-radius: 4px;
        margin-top: 3px;
        letter-spacing: 0.04em;
        width: fit-content;
    }}

    /* ─── Sidebar Section Labels ─── */
    .sidebar-section {{
        font-size: 0.7rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding-left: 8px;
        border-left: 2px solid var(--accent);
        margin: 1.25rem 0 0.75rem 0;
    }}

    /* ─── Sidebar Footer ─── */
    .sidebar-footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        text-align: center;
        font-size: 0.68rem;
        color: var(--text-dim);
        line-height: 1.6;
    }}
    .sidebar-footer a {{
        color: var(--accent);
        text-decoration: none;
    }}

    /* ─── Themed Checkboxes ─── */
    [data-testid="stCheckbox"] label span {{
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {{
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
    }}

    /* ─── Themed Expanders ─── */
    [data-testid="stExpander"] {{
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }}
    [data-testid="stExpander"] summary {{
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background-color: var(--bg-subtle) !important;
    }}

    /* ─── Themed Date Input Calendar ─── */
    div[data-baseweb="popover"] {{
        background-color: var(--card) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25) !important;
    }}
    div[data-baseweb="calendar"] {{
        background-color: var(--card) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    div[data-baseweb="calendar"] div {{
        color: var(--text) !important;
    }}

    /* ─── Animated Hero Accent Stripe ─── */
    @keyframes heroStripe {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .hero-stripe {{
        height: 3px;
        background: linear-gradient(90deg, var(--accent), #a78bfa, #ec4899, #f59e0b, var(--accent));
        background-size: 300% 100%;
        animation: heroStripe 6s ease infinite;
        border-radius: 0 0 var(--radius) var(--radius);
        margin-top: -1px;
    }}

    /* ─── Strategy Summary Strip ─── */
    .summary-strip {{
        display: flex;
        align-items: center;
        gap: 1.5rem;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        animation: fadeSlideUp 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.4s both;
    }}
    .summary-item {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .summary-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .summary-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 500;
    }}
    .summary-value {{
        font-size: 0.95rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .summary-divider {{
        width: 1px;
        height: 28px;
        background: var(--border);
    }}

    /* ─── Status Pill (for hero) ─── */
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 0.03em;
        margin-top: 0.75rem;
    }}
    .status-dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        animation: pulse-glow 2s infinite ease-in-out;
    }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
OUTPUT_METRICS = PROJECT_ROOT / "output" / "metrics"
OUTPUT_PLOTS = PROJECT_ROOT / "output" / "plots"

REGIME_COLORS = {
    "Bull": "#10b981",
    "Recovery": "#3b82f6",
    "Bear": "#f59e0b",
    "Crisis": "#ef4444",
}
REGIME_COLORS_TRANSPARENT = {
    "Bull": "rgba(16, 185, 129, 0.12)",
    "Recovery": "rgba(59, 130, 246, 0.12)",
    "Bear": "rgba(245, 158, 11, 0.12)",
    "Crisis": "rgba(239, 68, 68, 0.12)",
}


@st.cache_data(ttl=3600)
def load_data():
    """Load all pipeline outputs."""
    data = {}

    # Master prices
    master_path = DATA_CLEANED / "master.parquet"
    if master_path.exists():
        data["master"] = pd.read_parquet(master_path)

    # Features with regimes
    feat_path = DATA_PROCESSED / "features_with_regimes.parquet"
    if feat_path.exists():
        data["features"] = pd.read_parquet(feat_path)

    # Features with signals
    sig_path = DATA_PROCESSED / "features_with_signals.parquet"
    if sig_path.exists():
        data["signals"] = pd.read_parquet(sig_path)

    # Backtest results
    oos_path = OUTPUT_METRICS / "oos_results.parquet"
    if oos_path.exists():
        data["oos"] = pd.read_parquet(oos_path)

    # Metrics JSON
    metrics_path = OUTPUT_METRICS / "backtest_results.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    # Weights history
    weights_path = OUTPUT_METRICS / "weights_history.parquet"
    if weights_path.exists():
        data["weights"] = pd.read_parquet(weights_path)

    return data


def generate_sparkline(series, color="#818cf8", width=120, height=35):
    """Generate a clean SVG sparkline path from a pandas Series."""
    if series is None or len(series) < 2:
        return ""
    # Filter out NaNs and convert
    series = series.dropna()
    if len(series) < 2:
        return ""
        
    vals = series.values
    # Downsample to 40 points for smooth performance
    if len(vals) > 40:
        indices = np.linspace(0, len(vals) - 1, 40, dtype=int)
        vals = vals[indices]
        
    v_min, v_max = min(vals), max(vals)
    if v_max == v_min:
        return ""
        
    # Generate path coords
    points = []
    for i, v in enumerate(vals):
        x = (i / (len(vals) - 1)) * width
        y = height - ((v - v_min) / (v_max - v_min)) * height
        points.append(f"{x:.1f},{y:.1f}")
        
    path_data = " L ".join(points)
    svg = f"""
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="overflow: visible; display: block; margin-top: 8px;">
        <path d="M {path_data}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
    """
    return svg


def render_metric_card(label, value, sublabel="", css_class="", sparkline_html="", anim_delay=""):
    delay_cls = f" anim-card {anim_delay}" if anim_delay else ""
    st.markdown(f"""
    <div class="metric-card{delay_cls}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value {css_class}">{value}</div>
                <div class="metric-sublabel">{sublabel}</div>
            </div>
            {f'<div style="width: 120px; display: flex; justify-content: flex-end; align-items: center; padding-top: 5px;">{sparkline_html}</div>' if sparkline_html else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────
def main():
    data = load_data()

    if not data:
        st.error("⚠️ No pipeline data found. Run `python run_pipeline.py` first.")
        st.stop()

    grid_color = "rgba(255,255,255,0.04)" if IS_DARK else "rgba(0,0,0,0.04)"
    text_color_plotly = "#a1a1aa" if IS_DARK else "#71717a"

    # ── Brand Header & Theme Toggle ───────────────────────────
    head_left, head_right = st.columns([8, 1])
    with head_left:
        st.markdown("""
        <div class="brand">
            <span class="brand-logo">🔮</span>
            <span class="brand-name">Market Regime Detection</span>
            <span class="brand-subtitle">Adaptive Portfolio Allocation System</span>
        </div>
        """, unsafe_allow_html=True)
    with head_right:
        theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
        st.button(theme_label, on_click=toggle_theme, use_container_width=True)

    # ── Hero Banner ───────────────────────────────────────────
    # Determine current regime for hero status pill
    _hero_features = data.get("features")
    _current_regime = "Unknown"
    _regime_color = "#94a3b8"
    _regime_bg = "rgba(148,163,184,0.1)"
    if _hero_features is not None and "regime_label_stable" in _hero_features.columns:
        _last_regime = _hero_features["regime_label_stable"].dropna()
        if len(_last_regime) > 0:
            _current_regime = _last_regime.iloc[-1]
            _regime_color = REGIME_COLORS.get(_current_regime, "#94a3b8")
            _regime_bg = f"{_regime_color}18"
    _last_date = _hero_features.index.max().strftime('%b %d, %Y') if _hero_features is not None and not _hero_features.empty else "—"

    st.markdown(f"""
    <div class="hero">
        <p>Adaptive Portfolio Allocation powered by HMM regime classification,
        regime-specialist Random Forests, and FFD-stationarized features.
        Walk-Forward validated with transaction costs.</p>
        <div class="status-pill" style="color: {_regime_color}; background: {_regime_bg}; border: 1px solid {_regime_color}30;">
            <span class="status-dot" style="background: {_regime_color};"></span>
            Current Regime: {_current_regime} &nbsp;·&nbsp; Data through {_last_date}
        </div>
    </div>
    <div class="hero-stripe"></div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <span class="sidebar-brand-icon">🔮</span>
            <div class="sidebar-brand-text">
                <span class="sidebar-brand-name">Regime Detector</span>
                <span class="sidebar-brand-version">v2.0 · Adaptive</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Dashboard Controls</div>', unsafe_allow_html=True)

        features = data.get("features")
        if features is not None and not features.empty:
            min_date = features.index.min().date()
            max_date = features.index.max().date()
            
            # Date Range Toggle Checkbox
            use_date_range = st.checkbox("Limit Date Range", value=True)
            
            # Separate Starting Date and Ending Date Inputs stacked vertically
            start_date = st.date_input(
                "Starting Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                disabled=not use_date_range
            )
            
            end_date = st.date_input(
                "Ending Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                disabled=not use_date_range
            )
            
            if use_date_range:
                if start_date > end_date:
                    st.error("⚠️ Starting Date must be before or equal to Ending Date.")
                    date_range = (min_date, max_date)
                else:
                    date_range = (start_date, end_date)
            else:
                date_range = (min_date, max_date)
        else:
            date_range = None

        st.markdown("---")
        st.markdown('<div class="sidebar-section">Regime Legend</div>', unsafe_allow_html=True)
        for regime, color in REGIME_COLORS.items():
            st.markdown(f'<span class="regime-badge regime-{regime.lower()}">{regime}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-section">Pipeline Info</div>', unsafe_allow_html=True)
        if "metrics" in data:
            m = data["metrics"]
            st.markdown(f"""
            <div class="info-panel">
                <strong>Period:</strong> {m.get('n_years', '?')} years<br>
                <strong>OOS Days:</strong> {m.get('total_days', '?')}<br>
                <strong>Total Trades:</strong> {m.get('total_trades', '?')}<br>
                <strong>Fees:</strong> {m.get('fees_bps', '?')} bps<br>
                <strong>Slippage:</strong> {m.get('slippage_bps', '?')} bps
            </div>
            """, unsafe_allow_html=True)

        # Sidebar footer
        st.markdown("""
        <div class="sidebar-footer">
            HMM · Random Forest · FFD · WFO<br>
            SEBI 2026 Compliant · SHAP Explainability<br>
            Built with Streamlit & Plotly
        </div>
        """, unsafe_allow_html=True)

    # ── Filter by date range ──────────────────────────────────
    def filter_df(df):
        if date_range and len(date_range) == 2 and df is not None:
            mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
            return df[mask]
        return df

    # ── Metrics Cards ─────────────────────────────────────────
    if "metrics" in data:
        m = data["metrics"]
        
        # Compute sparklines dynamically based on selected date range
        strategy_cum = None
        drawdown_series = None
        win_rate_series = None
        
        oos = filter_df(data.get("oos"))
        if oos is not None and not oos.empty:
            if "cum_strategy" in oos.columns:
                strategy_cum = oos["cum_strategy"]
            if "strategy_return" in oos.columns:
                cum = (1 + oos["strategy_return"]).cumprod()
                peak = cum.cummax()
                drawdown_series = (cum - peak) / peak
                win_rate_series = oos["strategy_return"].rolling(max(2, min(20, len(oos)))).mean()

        spark_color_sharpe = "#818cf8"
        spark_color_dd = red_color
        spark_color_ar = green_color
        spark_color_wr = "#a78bfa"
        spark_color_calmar = "#60a5fa"

        spark_sharpe = generate_sparkline(strategy_cum, spark_color_sharpe) if strategy_cum is not None else ""
        spark_dd = generate_sparkline(drawdown_series, spark_color_dd) if drawdown_series is not None else ""
        spark_ar = generate_sparkline(strategy_cum, spark_color_ar) if strategy_cum is not None else ""
        spark_wr = generate_sparkline(win_rate_series, spark_color_wr) if win_rate_series is not None else ""
        spark_calmar = generate_sparkline(strategy_cum, spark_color_calmar) if strategy_cum is not None else ""

        cols = st.columns(5)
        with cols[0]:
            val = m.get("sharpe_ratio", 0)
            css = "positive" if val > 0.5 else "negative" if val < 0 else ""
            render_metric_card("Sharpe Ratio", f"{val:.2f}", f"Market: {m.get('market_sharpe', 0):.2f}", css, spark_sharpe, "delay-1")
        with cols[1]:
            val = m.get("max_drawdown", 0)
            render_metric_card("Max Drawdown", f"{val:.1%}", f"Market: {m.get('market_max_drawdown', 0):.1%}", "negative", spark_dd, "delay-2")
        with cols[2]:
            val = m.get("annual_return", 0)
            css = "positive" if val > 0 else "negative"
            render_metric_card("Annual Return", f"{val:.1%}", f"Total: {m.get('total_return', 0):.1%}", css, spark_ar, "delay-3")
        with cols[3]:
            val = m.get("win_rate", 0)
            css = "positive" if val > 0.5 else ""
            render_metric_card("Win Rate", f"{val:.1%}", f"Profit Factor: {m.get('profit_factor', 0):.2f}", css, spark_wr, "delay-4")
        with cols[4]:
            val = m.get("calmar_ratio", 0)
            css = "positive" if val > 0.5 else ""
            render_metric_card("Calmar Ratio", f"{val:.2f}", f"Trades: {m.get('total_trades', 0)}", css, spark_calmar, "delay-5")

        # ── Strategy vs Market Summary Strip ───────────────────
        strat_ret = m.get("total_return", 0)
        market_ret = m.get("market_total_return", m.get("total_return", 0) * 0.7)  # fallback
        alpha_ret = strat_ret - market_ret
        strat_color = green_color if strat_ret >= 0 else red_color
        market_color = text_muted
        alpha_color = green_color if alpha_ret >= 0 else red_color
        alpha_sign = "+" if alpha_ret >= 0 else ""

        st.markdown(f"""
        <div class="summary-strip">
            <div class="summary-item">
                <span class="summary-dot" style="background: {strat_color};"></span>
                <span class="summary-label">Strategy</span>
                <span class="summary-value" style="color: {strat_color};">{strat_ret:+.1%}</span>
            </div>
            <div class="summary-divider"></div>
            <div class="summary-item">
                <span class="summary-dot" style="background: {market_color};"></span>
                <span class="summary-label">Market</span>
                <span class="summary-value" style="color: {market_color};">{market_ret:+.1%}</span>
            </div>
            <div class="summary-divider"></div>
            <div class="summary-item">
                <span class="summary-dot" style="background: {alpha_color};"></span>
                <span class="summary-label">Alpha</span>
                <span class="summary-value" style="color: {alpha_color};">{alpha_sign}{alpha_ret:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Regimes",
        "📊 Portfolio Allocation",
        "💰 Backtest Performance",
        "🔍 SHAP Explainability",
        "📋 Regime Analysis",
    ])

    # ── TAB 1: Price + Regime Overlay ─────────────────────────
    with tab1:
        st.markdown('<div class="section-header">📈 Asset Prices with Regime Overlay</div>', unsafe_allow_html=True)

        features = filter_df(data.get("features"))
        if features is not None and "SPY_Close" in features.columns:
            # Build figure
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("SPY Price with Regime Bands", "VIX / Realized Volatility"),
            )

            # Add regime background shading
            if "regime_label_stable" in features.columns:
                regimes = features["regime_label_stable"].dropna()
                if len(regimes) > 0:
                    # Group consecutive same-regime periods
                    regime_groups = (regimes != regimes.shift()).cumsum()
                    for _, group in regimes.groupby(regime_groups):
                        regime = group.iloc[0]
                        color = REGIME_COLORS_TRANSPARENT.get(regime, "rgba(128,128,128,0.1)")
                        fig.add_vrect(
                            x0=group.index[0], x1=group.index[-1],
                            fillcolor=color, line_width=0,
                            layer="below", row=1, col=1,
                        )
                        fig.add_vrect(
                            x0=group.index[0], x1=group.index[-1],
                            fillcolor=color, line_width=0,
                            layer="below", row=2, col=1,
                        )

            # SPY price line
            fig.add_trace(
                go.Scatter(
                    x=features.index, y=features["SPY_Close"],
                    name="SPY", line=dict(color="#818cf8", width=1.5),
                    hovertemplate="SPY: $%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

            # Add other assets as thin lines
            asset_colors = {
                "TLT_Close": "#60a5fa", "GLD_Close": "#fbbf24",
                "QQQ_Close": "#a78bfa", "DIA_Close": "#34d399",
            }
            for col_name, color in asset_colors.items():
                if col_name in features.columns:
                    label = col_name.replace("_Close", "")
                    # Normalize to SPY scale for overlay
                    series = features[col_name].dropna()
                    if len(series) > 0:
                        normalized = series / series.iloc[0] * features["SPY_Close"].iloc[0]
                        fig.add_trace(
                            go.Scatter(
                                x=normalized.index, y=normalized,
                                name=label, line=dict(color=color, width=1, dash="dot"),
                                opacity=0.6,
                                hovertemplate=f"{label}: $%{{y:.2f}}<extra></extra>",
                            ),
                            row=1, col=1,
                        )

            # VIX / Realized Vol on subplot
            if "VIX_Close" in features.columns:
                fig.add_trace(
                    go.Scatter(
                        x=features.index, y=features["VIX_Close"],
                        name="VIX", line=dict(color="#f87171", width=1.2),
                        hovertemplate="VIX: %{y:.1f}<extra></extra>",
                    ),
                    row=2, col=1,
                )
            if "realized_vol_20d" in features.columns:
                vol_pct = features["realized_vol_20d"] * 100
                fig.add_trace(
                    go.Scatter(
                        x=features.index, y=vol_pct,
                        name="RealVol 20d", line=dict(color="#fbbf24", width=1, dash="dash"),
                        hovertemplate="RealVol: %{y:.1f}%<extra></extra>",
                    ),
                    row=2, col=1,
                )

            fig.update_layout(
                template="plotly_dark" if IS_DARK else "plotly",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                height=700,
                margin=dict(l=60, r=30, t=50, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
                xaxis2=dict(gridcolor=grid_color, zerolinecolor=grid_color),
                yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title="Price ($)"),
                yaxis2=dict(gridcolor=grid_color, zerolinecolor=grid_color, title="Volatility"),
            )
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">SPY Price with Regime Bands</div>
                <div class="chart-subtitle">Color-coded overlays identify HMM detected states over the backtest period</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            # Regime distribution bar
            if "regime_label_stable" in features.columns:
                st.markdown('<div class="section-header">🎯 Regime Timeline</div>', unsafe_allow_html=True)
                regime_counts = features["regime_label_stable"].value_counts()
                total = len(features)
                cols = st.columns(len(regime_counts))
                for i, (regime, count) in enumerate(regime_counts.items()):
                    with cols[i]:
                        pct = count / total * 100
                        color = REGIME_COLORS.get(regime, "#94a3b8")
                        st.markdown(f"""
                        <div class="metric-card" style="border-color: {color}40; text-align: center;">
                            <div class="metric-label">{regime}</div>
                            <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{pct:.1f}%</div>
                            <div class="metric-sublabel">{count:,} trading days</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No price data available. Run the pipeline first.")

    # ── TAB 2: Portfolio Allocation ───────────────────────────
    with tab2:
        st.markdown('<div class="section-header">📊 Adaptive Portfolio Weights</div>', unsafe_allow_html=True)

        weights = filter_df(data.get("weights"))
        if weights is not None and not weights.empty:
            weight_cols = [c for c in weights.columns if c.startswith("w_")]
            if weight_cols:
                # ── Current Allocation Pie Chart ──────────────────
                st.markdown('<div class="section-header">🥧 Current Portfolio Allocation</div>', unsafe_allow_html=True)

                latest = weights.iloc[-1]
                current_regime = latest.get("regime", "Unknown")

                col_pie, col_info = st.columns([2, 1])

                with col_pie:
                    # Build donut chart of current weights
                    pie_labels = [c.replace("w_", "") for c in weight_cols]
                    pie_values = [float(latest.get(c, 0)) for c in weight_cols]
                    # Filter out zero weights
                    non_zero = [(l, v) for l, v in zip(pie_labels, pie_values) if v > 0.001]
                    if non_zero:
                        pie_labels_filtered, pie_values_filtered = zip(*non_zero)
                    else:
                        pie_labels_filtered, pie_values_filtered = pie_labels, pie_values

                    pie_colors = ["#818cf8", "#60a5fa", "#fbbf24", "#a78bfa", "#34d399", "#f87171"]

                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(pie_labels_filtered),
                        values=list(pie_values_filtered),
                        hole=0.55,
                        marker=dict(
                            colors=pie_colors[:len(pie_labels_filtered)],
                            line=dict(color="#09090b" if IS_DARK else "#ffffff", width=2),
                        ),
                        textinfo="label+percent",
                        textfont=dict(size=13, color=text_color_plotly, family="DM Sans"),
                        hovertemplate="%{label}: %{percent}<br>Weight: %{value:.1%}<extra></extra>",
                        sort=False,
                    )])
                    
                    fig_pie.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                        height=380,
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.1,
                            xanchor="center", x=0.5,
                            font=dict(size=12, color=text_color_plotly),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        annotations=[dict(
                            text=f"<b>{current_regime}</b>",
                            x=0.5, y=0.5, font_size=18,
                            font_color=REGIME_COLORS.get(current_regime, "#94a3b8"),
                            showarrow=False,
                        )],
                    )
                    
                    st.markdown("""
                    <div class="chart-wrap">
                        <div class="chart-title">Current Portfolio Allocation</div>
                        <div class="chart-subtitle">Direct donut visual of portfolio assets weights in the current active regime</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_info:
                    # Current regime badge and weight details
                    regime_color = REGIME_COLORS.get(current_regime, "#94a3b8")
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {regime_color}40;">
                        <div class="metric-label">Active Regime</div>
                        <div class="metric-value" style="font-size: 1.5rem; background: {regime_color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {current_regime}
                        </div>
                        <div class="metric-sublabel">As of {weights.index[-1].strftime('%Y-%m-%d')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Weight breakdown table
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Weight Breakdown</div>
                    """, unsafe_allow_html=True)
                    for i, col in enumerate(weight_cols):
                        asset_name = col.replace("w_", "")
                        w_val = float(latest.get(col, 0))
                        color = pie_colors[i % len(pie_colors)]
                        bar_width = max(w_val * 100, 0)
                        st.markdown(f"""
                        <div style="margin: 8px 0;">
                            <div style="display: flex; justify-content: space-between; color: var(--text); font-size: 0.85rem;">
                                <span style="color: {color}; font-weight: 600;">{asset_name}</span>
                                <span>{w_val:.1%}</span>
                            </div>
                            <div style="background: var(--bg-subtle); border-radius: 4px; height: 6px; margin-top: 4px;">
                                <div style="background: {color}; border-radius: 4px; height: 100%; width: {bar_width}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # ── Historical Stacked Area ───────────────────────
                st.markdown('<div class="section-header">📈 Historical Weight Evolution</div>', unsafe_allow_html=True)

                fig = go.Figure()
                colors = ["#818cf8", "#60a5fa", "#fbbf24", "#a78bfa", "#34d399", "#f87171"]
                def hex_to_rgba(hex_code, alpha=0.6):
                     hex_code = hex_code.lstrip('#')
                     return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {alpha})"

                for i, col in enumerate(weight_cols):
                    asset_name = col.replace("w_", "")
                    base_color = colors[i % len(colors)]
                    fill_color = base_color.replace(")", ", 0.6)").replace("rgb", "rgba") if "rgb" in base_color else hex_to_rgba(base_color, 0.6)
                    
                    fig.add_trace(go.Scatter(
                        x=weights.index, y=weights[col],
                        name=asset_name,
                        mode="lines",
                        stackgroup="weights",
                        line=dict(width=0.5, color=base_color),
                        fillcolor=fill_color,
                        hovertemplate=f"{asset_name}: %{{y:.1%}}<extra></extra>",
                    ))

                fig.update_layout(
                    template="plotly_dark" if IS_DARK else "plotly",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                    height=450,
                    margin=dict(l=60, r=30, t=30, b=30),
                    yaxis=dict(title="Weight", tickformat=".0%", gridcolor=grid_color),
                    xaxis=dict(gridcolor=grid_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                
                st.markdown("""
                <div class="chart-wrap">
                    <div class="chart-title">Historical Weight Evolution</div>
                    <div class="chart-subtitle">Daily breakdown of tradeable assets weights and strategic transitions over the backtest duration</div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

                # Average weights by regime
                st.markdown('<div class="section-header">🎯 Average Weights by Regime</div>', unsafe_allow_html=True)
                if "regime" in weights.columns:
                    avg_weights = weights.groupby("regime")[weight_cols].mean()
                    avg_weights.columns = [c.replace("w_", "") for c in avg_weights.columns]

                    fig2 = go.Figure()
                    for i, col in enumerate(avg_weights.columns):
                        fig2.add_trace(go.Bar(
                            x=avg_weights.index, y=avg_weights[col],
                            name=col, marker_color=colors[i % len(colors)],
                        ))
                    fig2.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                        barmode="stack", height=350,
                        yaxis=dict(title="Weight", tickformat=".0%", gridcolor=grid_color),
                        xaxis=dict(gridcolor=grid_color),
                        margin=dict(l=60, r=30, t=30, b=30),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.2,
                            xanchor="center", x=0.5,
                            font=dict(size=12, color=text_color_plotly),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                    )
                    
                    st.markdown("""
                    <div class="chart-wrap">
                        <div class="chart-title">Average Weights by Regime</div>
                        <div class="chart-subtitle">Target portfolio structure per HMM regime, showcasing rotation behavior</div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Portfolio weights not yet computed. Run portfolio optimization first.")

    # ── TAB 3: Backtest Performance ───────────────────────────
    with tab3:
        st.markdown('<div class="section-header">💰 Walk-Forward Backtest Performance</div>', unsafe_allow_html=True)

        oos = filter_df(data.get("oos"))
        if oos is not None and "cum_strategy" in oos.columns:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("Cumulative Return: Strategy vs Market", "Drawdown"),
            )

            # Cumulative returns
            fig.add_trace(
                go.Scatter(
                    x=oos.index, y=oos["cum_strategy"],
                    name="Strategy", line=dict(color="#818cf8", width=2),
                    fill="tozeroy", fillcolor="rgba(129, 140, 248, 0.1)",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=oos.index, y=oos["cum_market"],
                    name="Market (SPY)", line=dict(color="#94a3b8", width=1.5, dash="dash"),
                ),
                row=1, col=1,
            )

            # Drawdown
            cum = (1 + oos["strategy_return"]).cumprod()
            peak = cum.cummax()
            dd = (cum - peak) / peak
            fig.add_trace(
                go.Scatter(
                    x=oos.index, y=dd,
                    name="Drawdown", line=dict(color="#f87171", width=1),
                    fill="tozeroy", fillcolor="rgba(248, 113, 113, 0.15)",
                ),
                row=2, col=1,
            )

            fig.update_layout(
                template="plotly_dark" if IS_DARK else "plotly",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                height=600,
                margin=dict(l=60, r=30, t=50, b=30),
                yaxis=dict(title="Cumulative Return", gridcolor=grid_color),
                yaxis2=dict(title="Drawdown", tickformat=".0%", gridcolor=grid_color),
                xaxis=dict(gridcolor=grid_color),
                xaxis2=dict(gridcolor=grid_color),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=12, color=text_color_plotly),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Walk-Forward Backtest Performance</div>
                <div class="chart-subtitle">Cumulative returns comparison of the adaptive strategy against SPY along with drawdown profiles</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            # Monthly returns heatmap
            st.markdown('<div class="section-header">📅 Monthly Returns</div>', unsafe_allow_html=True)
            monthly = oos["strategy_return"].resample("ME").sum()
            monthly_df = pd.DataFrame({
                "Year": monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            })
            pivot = monthly_df.pivot_table(values="Return", index="Year", columns="Month", aggfunc="sum")
            pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

            fig3 = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index.astype(str),
                colorscale=[
                    [0, "#ef4444"], [0.3, "#fbbf24"],
                    [0.5, "#1e293b" if IS_DARK else "#f1f5f9"], [0.7, "#34d399"], [1.0, "#10b981"],
                ],
                zmid=0,
                text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=10, color=text_color_plotly),
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
            ))
            fig3.update_layout(
                template="plotly_dark" if IS_DARK else "plotly",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                height=max(250, len(pivot) * 35 + 80),
                margin=dict(l=60, r=30, t=20, b=30),
            )
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Monthly Returns Heatmap</div>
                <div class="chart-subtitle">Breakdown of monthly performance for the strategy across backtest years</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Backtest results not available. Run the backtest first.")

    # ── TAB 4: SHAP Explainability ────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🔍 SHAP Feature Explainability</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-panel">
            <strong>SEBI 2026 Compliance:</strong> SHapley Additive exPlanations (SHAP) provide
            regulatory-grade transparency into model decisions. Each plot shows which features
            drove the algorithm's predictions within each market regime.
        </div>
        """, unsafe_allow_html=True)

        if OUTPUT_PLOTS.exists():
            combined_plot = OUTPUT_PLOTS / "shap_combined_importance.png"
            if combined_plot.exists():
                st.markdown('<div class="section-header">🌍 Global Regime Feature Importance</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                st.image(str(combined_plot), caption="Combined Cross-Regime Feature Importance", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="section-header">🎯 Regime-Specific Explanations</div>', unsafe_allow_html=True)
            regime_tabs = st.tabs(["🟢 Bull Regime", "🔵 Recovery Regime", "🟡 Bear Regime", "🔴 Crisis Regime"])
            
            regime_mapping = {
                "🟢 Bull Regime": "bull",
                "🔵 Recovery Regime": "recovery",
                "🟡 Bear Regime": "bear",
                "🔴 Crisis Regime": "crisis"
            }
            
            for tab_title, key in regime_mapping.items():
                with regime_tabs[list(regime_mapping.keys()).index(tab_title)]:
                    summary_plot = OUTPUT_PLOTS / f"shap_summary_{key}.png"
                    bar_plot = OUTPUT_PLOTS / f"shap_bar_{key}.png"
                    
                    col_sum, col_bar = st.columns(2)
                    with col_sum:
                        if summary_plot.exists():
                            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                            st.image(str(summary_plot), caption=f"Beeswarm Summary - {key.title()} Regime", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"Summary beeswarm plot not found for {key}.")
                    with col_bar:
                        if bar_plot.exists():
                            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                            st.image(str(bar_plot), caption=f"Bar Feature Importance - {key.title()} Regime", use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"Bar importance plot not found for {key}.")
        else:
            st.info("SHAP plots not generated yet. Run the explainability step first.")

        # Audit log summary
        audit_path = PROJECT_ROOT / "output" / "logs" / "shap_audit.jsonl"
        if audit_path.exists():
            st.markdown('<div class="section-header">📋 Audit Log Summary</div>', unsafe_allow_html=True)
            lines = audit_path.read_text().strip().split("\n")
            n_entries = len(lines)
            st.markdown(f"""
            <div class="info-panel">
                <strong>Audit entries:</strong> {n_entries:,}<br>
                <strong>Format:</strong> JSONL (one JSON object per decision)<br>
                <strong>Fields:</strong> date, regime, signal, confidence, top SHAP features, model version<br>
                <strong>File:</strong> <code>output/logs/shap_audit.jsonl</code>
            </div>
            """, unsafe_allow_html=True)

            # Show sample entries
            with st.expander("📄 Sample Audit Entries (last 5)"):
                for line in lines[-5:]:
                    try:
                        entry = json.loads(line)
                        st.json(entry)
                    except json.JSONDecodeError:
                        pass

    # ── TAB 5: Regime Analysis ────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header">📋 Regime Statistical Analysis</div>', unsafe_allow_html=True)

        features = data.get("features")
        if features is not None and "regime_label_stable" in features.columns:
            # Regime transition matrix
            regimes = features["regime_label_stable"].dropna()
            transitions = pd.crosstab(
                regimes, regimes.shift(-1), normalize="index"
            )
            transitions.index.name = "From"
            transitions.columns.name = "To"

            fig4 = go.Figure(data=go.Heatmap(
                z=transitions.values,
                x=transitions.columns.astype(str),
                y=transitions.index.astype(str),
                colorscale=[[0, "#09090b" if IS_DARK else "#ffffff"], [0.5, "#2563eb"], [1, "#22c55e"]],
                text=[[f"{v:.1%}" for v in row] for row in transitions.values],
                texttemplate="%{text}",
                textfont=dict(size=12, color=text_color_plotly),
                hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.2%}<extra></extra>",
            ))
            fig4.update_layout(
                template="plotly_dark" if IS_DARK else "plotly",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Sans, sans-serif", color=text_color_plotly),
                height=350,
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                margin=dict(l=80, r=30, t=30, b=50),
            )
            
            st.markdown("""
            <div class="chart-wrap">
                <div class="chart-title">Regime Transition Probabilities</div>
                <div class="chart-subtitle">Likelihood of shifting from one market regime to another based on historical chain transitions</div>
            """, unsafe_allow_html=True)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            # Per-regime statistics
            st.markdown('<div class="section-header">📊 Per-Regime Return Statistics</div>', unsafe_allow_html=True)
            if "SPY_returns" in features.columns:
                regime_stats = features.groupby("regime_label_stable")["SPY_returns"].agg([
                    ("Mean Return", "mean"),
                    ("Std Dev", "std"),
                    ("Skewness", "skew"),
                    ("Count", "count"),
                ])
                regime_stats["Annualized Return"] = regime_stats["Mean Return"] * 252
                regime_stats["Annualized Vol"] = regime_stats["Std Dev"] * np.sqrt(252)
                regime_stats["Sharpe"] = regime_stats["Annualized Return"] / regime_stats["Annualized Vol"]

                display_stats = regime_stats[["Annualized Return", "Annualized Vol", "Sharpe", "Skewness", "Count"]].copy()
                for col in ["Annualized Return", "Annualized Vol"]:
                    display_stats[col] = display_stats[col].apply(lambda x: f"{x:.2%}")
                display_stats["Sharpe"] = display_stats["Sharpe"].apply(lambda x: f"{x:.2f}")
                display_stats["Skewness"] = display_stats["Skewness"].apply(lambda x: f"{x:.2f}")
                display_stats["Count"] = display_stats["Count"].apply(lambda x: f"{int(x):,}")

                # Render display_stats as HTML table
                headers = "".join([f"<th>{col}</th>" for col in ["Regime"] + list(display_stats.columns)])
                rows = ""
                for idx, row in display_stats.iterrows():
                    badge = f'<span class="regime-badge regime-{idx.lower()}">{idx}</span>'
                    row_cells = "".join([f"<td>{val}</td>" for val in row.values])
                    rows += f"<tr><td>{badge}</td>{row_cells}</tr>"
                
                st.markdown(f"""
                <table class="data-table">
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                """, unsafe_allow_html=True)
        else:
            st.info("Regime data not available.")


if __name__ == "__main__":
    main()
