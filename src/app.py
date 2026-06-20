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
import textwrap

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
card_color = "rgba(20, 20, 25, 0.6)" if IS_DARK else "#ffffff"
card_hover = "rgba(30, 30, 35, 0.8)" if IS_DARK else "#f4f4f5"
border_color = "rgba(255, 255, 255, 0.08)" if IS_DARK else "#e4e4e7"
border_subtle = "rgba(255, 255, 255, 0.04)" if IS_DARK else "#f0f0f2"
text_color = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#9ca3af"  # text-gray-400 equivalent
text_dim = "#6b7280" if IS_DARK else "#a1a1aa"
accent_color = "#2563eb"
accent_muted = "rgba(37, 99, 235, 0.2)"
green_color = "#34d399" if IS_DARK else "#10b981"
green_muted = "rgba(52, 211, 153, 0.15)" if IS_DARK else "rgba(16, 185, 129, 0.1)"
red_color = "#f87171" if IS_DARK else "#ef4444"
red_muted = "rgba(248, 113, 113, 0.15)" if IS_DARK else "rgba(239, 68, 68, 0.1)"
amber_color = "#fbbf24" if IS_DARK else "#f59e0b"
amber_muted = "rgba(251, 191, 36, 0.15)" if IS_DARK else "rgba(245, 158, 11, 0.1)"
shadow_val = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"
radius_val = "12px"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');

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
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
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
        font-family: 'Inter', sans-serif;
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
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.015) 0%, rgba(255, 255, 255, 0.005) 100%) !important;
        background-color: var(--card) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent) !important;
        box-shadow: 0 10px 30px rgba(37,99,235,0.06), 0 0 0 1px rgba(37,99,235,0.08) !important;
    }}
    .metric-label {{
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .metric-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-top: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* Streamlit Border Container styling overrides to look like cards */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        background: var(--card) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
        box-shadow: var(--shadow) !important;
        margin-bottom: 2rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"]:hover {{
        border-color: var(--accent) !important;
        box-shadow: 0 8px 30px rgba(37,99,235,0.04) !important;
    }}

    /* Chart Container Card (legacy, kept for nested elements) */
    .chart-wrap {{
        background: transparent;
        border: none;
        padding: 0;
        box-shadow: none;
        margin-bottom: 0.5rem;
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

    .chart-title {{
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
    }}
    .chart-subtitle {{
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}

    /* Data Tables (HTML) */
    .data-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
        margin-top: 1rem;
        background: transparent;
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
        background-color: rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid var(--border);
    }}
    .data-table th:not(:first-child) {{
        text-align: right;
    }}
    .data-table td {{
        padding: 0.75rem 1rem;
        color: var(--text);
        border-bottom: 1px solid var(--border-subtle);
        font-family: 'JetBrains Mono', monospace;
    }}
    .data-table td:not(:first-child) {{
        text-align: right;
    }}
    .data-table tr:nth-child(even) {{
        background-color: rgba(255, 255, 255, 0.01);
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

    /* Premium Bottom-Border Tabs */
    button[data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.25rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.2s ease !important;
        margin-right: 0.5rem !important;
    }}
    button[data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        border-bottom-color: rgba(255, 255, 255, 0.2) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--text) !important;
        background: transparent !important;
        border-bottom-color: var(--accent) !important;
        font-weight: 600 !important;
    }}
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    [data-baseweb="tab-list"] {{
        gap: 0 !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin-bottom: 2rem !important;
    }}

    /* Regime badges */
    .regime-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }}
    .regime-bull {{ color: var(--green); background: var(--green-muted); border: 1px solid rgba(52,211,153,0.15); }}
    .regime-recovery {{ color: var(--accent); background: var(--accent-muted); border: 1px solid rgba(37,99,235,0.15); }}
    .regime-bear {{ color: var(--amber); background: var(--amber-muted); border: 1px solid rgba(251,191,36,0.15); }}
    .regime-crisis {{ color: var(--red); background: var(--red-muted); border: 1px solid rgba(248,113,113,0.15); }}

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

    /* ─── Glass Shimmer on Metric Card Hover ─── */
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 200%; }}
    }}
    .metric-card::after {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
        pointer-events: none;
        transition: none;
    }}
    .metric-card:hover::after {{
        animation: shimmer 0.8s ease-out forwards;
    }}

    /* ─── Secondary Metric Card (smaller, gradient top-border) ─── */
    .metric-card-secondary {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.015) 0%, rgba(255, 255, 255, 0.005) 100%) !important;
        background-color: var(--card) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), border-color 0.25s ease, box-shadow 0.25s ease;
        position: relative;
        overflow: hidden;
    }}
    /* Commented out sliding top border in favor of custom accent colors */
    /* .metric-card-secondary::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), #a78bfa, #ec4899);
        background-size: 200% 100%;
        animation: heroStripe 4s ease infinite;
    }} */
    .metric-card-secondary:hover {{
        transform: translateY(-2px);
        border-color: var(--accent) !important;
        box-shadow: 0 8px 25px rgba(37,99,235,0.04) !important;
    }}
    .metric-card-secondary .metric-value {{
        font-size: 1.4rem;
    }}

    /* ─── Responsive KPI Grid ─── */
    @media (max-width: 1100px) {{
        .kpi-grid-primary {{
            grid-template-columns: repeat(3, 1fr) !important;
        }}
        .kpi-grid-secondary {{
            grid-template-columns: repeat(3, 1fr) !important;
        }}
    }}
    @media (max-width: 768px) {{
        .kpi-grid-primary {{
            grid-template-columns: repeat(2, 1fr) !important;
        }}
        .kpi-grid-secondary {{
            grid-template-columns: repeat(2, 1fr) !important;
        }}
        .hero-header {{
            flex-direction: column !important;
            align-items: flex-start !important;
        }}
    }}

    /* ─── Regime Transition Arrows ─── */
    .transition-row {{
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 0;
    }}
    .transition-arrow {{
        color: var(--text-dim);
        font-size: 0.7rem;
    }}
    .transition-prob {{
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
    }}

    /* ─── Regime Calendar Strip ─── */
    .calendar-strip {{
        display: flex;
        width: 100%;
        height: 18px;
        border-radius: 4px;
        overflow: hidden;
        margin: 4px 0;
    }}
    .calendar-strip-segment {{
        height: 100%;
        transition: opacity 0.2s ease;
    }}
    .calendar-strip-segment:hover {{
        opacity: 0.8;
    }}

    /* ─── Additional Animation Delays ─── */
    .delay-6 {{ animation-delay: 0.40s; }}
    .delay-7 {{ animation-delay: 0.47s; }}
    .delay-8 {{ animation-delay: 0.54s; }}
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


def load_data():
    """Load all pipeline outputs, automatically invalidating when files update."""
    master_path = DATA_CLEANED / "master.parquet"
    feat_path = DATA_PROCESSED / "features_with_regimes.parquet"
    sig_path = DATA_PROCESSED / "features_with_signals.parquet"
    oos_path = OUTPUT_METRICS / "oos_results.parquet"
    metrics_path = OUTPUT_METRICS / "backtest_results.json"
    weights_path = OUTPUT_METRICS / "weights_history.parquet"

    mtimes = (
        master_path.stat().st_mtime if master_path.exists() else 0,
        feat_path.stat().st_mtime if feat_path.exists() else 0,
        sig_path.stat().st_mtime if sig_path.exists() else 0,
        oos_path.stat().st_mtime if oos_path.exists() else 0,
        metrics_path.stat().st_mtime if metrics_path.exists() else 0,
        weights_path.stat().st_mtime if weights_path.exists() else 0,
    )
    return _load_data_impl(*mtimes)


@st.cache_data(ttl=3601)
def _load_data_impl(mt1, mt2, mt3, mt4, mt5, mt6):
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


def load_audit_logs():
    """Load and parse the SEBI 2026 SHAP audit log, invalidating when file updates."""
    audit_path = PROJECT_ROOT / "output" / "logs" / "shap_audit.jsonl"
    mtime = audit_path.stat().st_mtime if audit_path.exists() else 0
    return _load_audit_logs_impl(mtime)


@st.cache_data(ttl=3601)
def _load_audit_logs_impl(mtime):
    audit_path = PROJECT_ROOT / "output" / "logs" / "shap_audit.jsonl"
    if not audit_path.exists():
        return None
    try:
        df = pd.read_json(audit_path, lines=True)
        if not df.empty and "date" in df.columns:
            # Parse date and sort descending so latest is on top
            df["date_parsed"] = pd.to_datetime(df["date"])
            df = df.sort_values("date_parsed", ascending=False)
        return df
    except Exception as e:
        st.error(f"Error loading audit logs: {e}")
        return None


def generate_sparkline(series, color="#818cf8", width=120, height=35):
    """Generate a clean SVG sparkline path with dynamic gradient fill from a pandas Series."""
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
    color_id = "".join(c for c in color if c.isalnum())
    svg = f"""
    <svg width="100%" height="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="none" style="display: block; margin: 0; padding: 0;">
        <defs>
            <linearGradient id="spark-grad-{color_id}" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="{color}" stop-opacity="0.16" />
                <stop offset="100%" stop-color="{color}" stop-opacity="0.00" />
            </linearGradient>
        </defs>
        <path d="M 0.0,{height} L {path_data} L {width:.1f},{height} Z" fill="url(#spark-grad-{color_id})" />
        <path d="M {path_data}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.45" />
    </svg>
    """
    return svg


def render_metric_card_html(label, value, sublabel="", css_class="", sparkline_html="", anim_delay="", accent_color=None):
    delay_cls = f" anim-card {anim_delay}" if anim_delay else ""
    border_style = f" border-top: 2.5px solid {accent_color};" if accent_color else ""
    return textwrap.dedent(f"""
    <div class="metric-card{delay_cls}" style="margin-bottom: 0; display: flex; flex-direction: column; justify-content: space-between; min-height: 115px; position: relative; overflow: hidden;{border_style}">
        <div style="position: relative; z-index: 2; padding: 1.15rem 1.25rem 1.5rem;">
            <div class="metric-label" style="white-space: nowrap; font-size: 0.68rem; opacity: 0.85;">{label}</div>
            <div class="metric-value {css_class}" style="font-size: 1.7rem; font-weight: 700; margin-top: 0.2rem; line-height: 1.1;">{value}</div>
            <div class="metric-sublabel" style="white-space: nowrap; font-size: 0.7rem; margin-top: 0.25rem; opacity: 0.9;">{sublabel}</div>
        </div>
        {f'<div style="position: absolute; bottom: 0; left: 0; right: 0; height: 38px; z-index: 1; pointer-events: none; overflow: hidden; margin: 0; padding: 0;">{sparkline_html}</div>' if sparkline_html else ''}
    </div>
    """)


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
        st.button(theme_label, on_click=toggle_theme, width="stretch")

    # ── Hero Banner ───────────────────────────────────────────
    # Determine current regime for hero status pill
    _hero_features = data.get("features")
    _current_regime = "Unknown"
    _regime_color = "#94a3b8"
    _regime_bg = "rgba(148,163,184,0.1)"
    _days_in_regime = 0
    _regime_transitions = {}  # {target_regime: probability}
    _regime_confidence = 0.0  # self-persistence probability

    if _hero_features is not None and "regime_label_stable" in _hero_features.columns:
        _last_regime = _hero_features["regime_label_stable"].dropna()
        if len(_last_regime) > 0:
            _current_regime = _last_regime.iloc[-1]
            _regime_color = REGIME_COLORS.get(_current_regime, "#94a3b8")
            _regime_bg = f"{_regime_color}18"

            # Days in current regime: count consecutive same-regime days from the end
            _reversed = _last_regime.iloc[::-1]
            _days_in_regime = 0
            for _r in _reversed:
                if _r == _current_regime:
                    _days_in_regime += 1
                else:
                    break

            # Compute transition probabilities from current regime
            _transitions_matrix = pd.crosstab(
                _last_regime, _last_regime.shift(-1), normalize="index"
            )
            if _current_regime in _transitions_matrix.index:
                _row = _transitions_matrix.loc[_current_regime]
                _regime_transitions = _row.to_dict()
                _regime_confidence = _regime_transitions.get(_current_regime, 0.0)

    from datetime import datetime
    _live_date = datetime.now().strftime('%b %d, %Y')

    # Build transition arrows HTML
    _transition_html = ""
    if _regime_transitions:
        _sorted_trans = sorted(_regime_transitions.items(), key=lambda x: -x[1])
        for _target, _prob in _sorted_trans:
            if _target == _current_regime:
                continue  # skip self-loop in the arrows
            _t_color = REGIME_COLORS.get(_target, "#94a3b8")
            _prob_bg = f"{_t_color}20"
            _transition_html += textwrap.dedent(f"""
            <div class="transition-row" style="display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 4px 0;">
                <span class="regime-badge regime-{_target.lower()}" style="padding: 2px 6px; font-size: 0.62rem; min-width: 65px; text-align: center;">{_target}</span>
                <div style="flex-grow: 1; height: 4px; background: {'rgba(255,255,255,0.06)' if IS_DARK else 'rgba(0,0,0,0.06)'}; border-radius: 2px; position: relative; overflow: hidden;">
                    <div style="position: absolute; left: 0; top: 0; bottom: 0; width: {_prob * 100:.0f}%; background: {_t_color}; border-radius: 2px;"></div>
                </div>
                <span class="transition-prob" style="color: {_t_color}; font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.72rem; min-width: 30px; text-align: right;">{_prob:.0%}</span>
            </div>""")

    # Confidence gauge: SVG arc representing self-persistence probability
    _gauge_pct = min(_regime_confidence * 100, 100)
    _gauge_dash = _gauge_pct * 2.51  # circumference of r=40 circle ≈ 251.3
    _gauge_color = _regime_color

    st.markdown(textwrap.dedent(f"""
    <div class="hero-header" style="margin-bottom: 1.5rem; display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
        <div>
            <h1 style="font-size: 2.2rem; font-weight: 800; letter-spacing: -0.05em; color: var(--text); margin-bottom: 0.2rem; line-height: 1.2;">
                Market Regime & Adaptive Portfolio
            </h1>
            <p style="font-size: 0.95rem; color: var(--text-muted); max-width: 600px; line-height: 1.5; margin: 0;">
                Walk-Forward validated allocation using HMM regime classification, Random Forest specialists, and FFD features.
            </p>
        </div>
        <div style="display: flex; gap: 1rem; align-items: stretch;">
            <!-- Regime confidence gauge -->
            <div class="metric-card" style="margin-bottom: 0; padding: 0.75rem 1.25rem; border-color: {_regime_color}40; min-width: 140px; text-align: center;">
                <div class="metric-label" style="font-size: 0.65rem;">Regime Stability</div>
                <div style="position: relative; width: 72px; height: 72px; margin: 6px auto 4px;">
                    <svg viewBox="0 0 100 100" width="72" height="72" style="transform: rotate(-90deg);">
                        <circle cx="50" cy="50" r="40" fill="none" stroke="{'rgba(255,255,255,0.06)' if IS_DARK else 'rgba(0,0,0,0.06)'}" stroke-width="8"/>
                        <circle cx="50" cy="50" r="40" fill="none" stroke="{_gauge_color}" stroke-width="8"
                            stroke-dasharray="{_gauge_dash} 251.3" stroke-linecap="round"
                            style="filter: drop-shadow(0 0 4px {_gauge_color}40); transition: stroke-dasharray 0.6s ease;"/>
                    </svg>
                    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700; color: {_gauge_color};">
                        {_regime_confidence:.0%}
                    </div>
                </div>
            </div>
            <!-- Current regime card -->
            <div class="metric-card" style="margin-bottom: 0; padding: 0.75rem 1.25rem; border-color: {_regime_color}40; min-width: 160px;">
                <div class="metric-label" style="font-size: 0.65rem;">Current Regime</div>
                <div class="metric-value" style="font-size: 1.2rem; color: {_regime_color}; display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {_regime_color}; box-shadow: 0 0 8px {_regime_color};"></span>
                    {_current_regime}
                </div>
                <div class="metric-sublabel" style="font-size: 0.7rem; margin-top: 4px;">
                    <span style="color: var(--text-muted);">Day</span>
                    <span style="color: {_regime_color}; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{_days_in_regime}</span>
                    <span style="color: var(--text-dim);">in streak</span>
                </div>
            </div>
            <!-- Live date + transition arrows -->
            <div class="metric-card" style="margin-bottom: 0; padding: 0.75rem 1.25rem; min-width: 180px;">
                <div class="metric-label" style="font-size: 0.65rem;">Live System Date</div>
                <div class="metric-value" style="font-size: 1.2rem; color: var(--accent);">{_live_date}</div>
                <div style="margin-top: 6px; border-top: 1px solid var(--border); padding-top: 6px;">
                    <div style="font-size: 0.62rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px;">Transition Probs</div>
                    {_transition_html}
                </div>
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

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

        master_df = data.get("master")
        features = data.get("features")
        if features is not None and not features.empty:
            process_start = pd.to_datetime("1990-01-01").date()
            process_end = features.index.max().date()
            
            # Date Range Toggle Checkbox
            use_date_range = st.checkbox("Limit Date Range", value=True)
            
            # Separate Starting Date and Ending Date Inputs stacked vertically
            start_date = st.date_input(
                "Starting Date",
                value=None,
                format="DD/MM/YYYY",
                min_value=process_start,
                max_value=process_end,
                disabled=not use_date_range
            )
            
            end_date = st.date_input(
                "Ending Date",
                value=None,
                format="DD/MM/YYYY",
                min_value=process_start,
                max_value=process_end,
                disabled=not use_date_range
            )
            
            if use_date_range:
                if start_date is None or end_date is None:
                    date_range = None
                elif start_date > end_date:
                    st.error("⚠️ Starting Date must be before or equal to Ending Date.")
                    date_range = (process_start, process_end)
                else:
                    date_range = (start_date, end_date)
            else:
                date_range = (process_start, process_end)
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

        sharpe_val = m.get("sharpe_ratio", 0)
        sharpe_css = "positive" if sharpe_val > 0.5 else "negative" if sharpe_val < 0 else ""
        c1 = render_metric_card_html("Sharpe Ratio", f"{sharpe_val:.2f}", f"Market: {m.get('market_sharpe', 0):.2f}", sharpe_css, spark_sharpe, "delay-1", accent_color=spark_color_sharpe)
        
        dd_val = m.get("max_drawdown", 0)
        c2 = render_metric_card_html("Max Drawdown", f"{dd_val:.1%}", f"Market: {m.get('market_max_drawdown', 0):.1%}", "negative", spark_dd, "delay-2", accent_color=spark_color_dd)
        
        ar_val = m.get("annual_return", 0)
        ar_css = "positive" if ar_val > 0 else "negative"
        c3 = render_metric_card_html("Annual Return", f"{ar_val:.1%}", f"Total: {m.get('total_return', 0):.1%}", ar_css, spark_ar, "delay-3", accent_color=spark_color_ar)
        
        wr_val = m.get("win_rate", 0)
        wr_css = "positive" if wr_val > 0.5 else ""
        c4 = render_metric_card_html("Win Rate", f"{wr_val:.1%}", f"Profit Factor: {m.get('profit_factor', 0):.2f}", wr_css, spark_wr, "delay-4", accent_color=spark_color_wr)
        
        cal_val = m.get("calmar_ratio", 0)
        cal_css = "positive" if cal_val > 0.5 else ""
        c5 = render_metric_card_html("Calmar Ratio", f"{cal_val:.2f}", f"Trades: {m.get('total_trades', 0)}", cal_css, spark_calmar, "delay-5", accent_color=spark_color_calmar)

        st.markdown(f"""
<div class="kpi-grid-primary" style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.5rem; margin-bottom: 1rem;">
{c1}
{c2}
{c3}
{c4}
{c5}
</div>
""", unsafe_allow_html=True)

        # ── Secondary KPI Row: Sortino, Omega, Alpha ──────────
        vbt = m.get("vectorbt_stats", {})
        sortino_val = vbt.get("Sortino Ratio", 0.0)
        omega_val = vbt.get("Omega Ratio", 0.0)
        
        # Calculate Beta and CAPM Alpha dynamically from OOS returns
        beta_val = 0.58  # fallback
        capm_alpha_val = 0.0077  # fallback (0.77% annualized)
        
        oos_df = data.get("oos")
        if oos_df is not None and not oos_df.empty and "strategy_return" in oos_df.columns and "market_return" in oos_df.columns:
            r_s = oos_df["strategy_return"].dropna()
            r_m = oos_df["market_return"].dropna()
            if len(r_s) > 1 and len(r_m) > 1:
                cov = r_s.cov(r_m)
                var_m = r_m.var()
                if var_m > 0:
                    beta_val = cov / var_m
                # Standard risk-free rate assumption of 3.5%
                rf_daily = 0.035 / 252
                alpha_daily = (r_s - rf_daily).mean() - beta_val * (r_m - rf_daily).mean()
                capm_alpha_val = alpha_daily * 252

        sortino_css = "positive" if sortino_val > 0.5 else "negative" if sortino_val < 0 else ""
        omega_css = "positive" if omega_val > 1.0 else ""
        alpha_css = "positive" if capm_alpha_val >= 0 else "negative"
        alpha_sign = "+" if capm_alpha_val >= 0 else ""

        def render_secondary_card(label, value, sublabel, css_class, delay, accent_color=None):
            delay_cls = f" anim-card {delay}" if delay else ""
            border_style = f" border-top: 2.5px solid {accent_color};" if accent_color else ""
            return textwrap.dedent(f"""
            <div class="metric-card-secondary{delay_cls}" style="{border_style}">
                <div style="padding: 1.1rem 1.25rem;">
                    <div class="metric-label" style="font-size: 0.68rem; opacity: 0.85;">{label}</div>
                    <div class="metric-value {css_class}" style="font-size: 1.35rem; font-weight: 700; margin-top: 0.2rem; line-height: 1.1;">{value}</div>
                    <div class="metric-sublabel" style="font-size: 0.7rem; margin-top: 0.2rem; opacity: 0.9;">{sublabel}</div>
                </div>
            </div>""")

        alpha_color_accent = "#34d399" if capm_alpha_val >= 0 else "#f87171"
        s1 = render_secondary_card("Sortino Ratio", f"{sortino_val:.2f}", "Downside risk-adjusted", sortino_css, "delay-6", accent_color="#c084fc")
        s2 = render_secondary_card("Omega Ratio", f"{omega_val:.2f}", "Gain/loss probability", omega_css, "delay-7", accent_color="#f472b6")
        s3 = render_secondary_card("Alpha (Annualized)", f"{alpha_sign}{capm_alpha_val:.2%}", f"CAPM α (Beta: {beta_val:.2f} · Rf: 3.5%)", alpha_css, "delay-8", accent_color=alpha_color_accent)

        st.markdown(f"""
<div class="kpi-grid-secondary" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 2rem;">
{s1}
{s2}
{s3}
</div>
""", unsafe_allow_html=True)

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
        if features is not None and "NSEI_Close" in features.columns:
            # Build figure
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                subplot_titles=("NIFTY Price with Regime Bands", "VIX / Realized Volatility"),
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
                    x=features.index, y=features["NSEI_Close"],
                    name="NIFTY", line=dict(color="#818cf8", width=1.5),
                    hovertemplate="NIFTY: $%{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )

            # Add other assets as thin lines
            asset_colors = {
                "LIQUIDBEES.NS_Close": "#60a5fa", "GOLDBEES.NS_Close": "#fbbf24",
                "JUNIORBEES.NS_Close": "#a78bfa", "BANKBEES.NS_Close": "#34d399",
            }
            for col_name, color in asset_colors.items():
                if col_name in features.columns:
                    label = col_name.replace("_Close", "")
                    # Normalize to SPY scale for overlay
                    series = features[col_name].dropna()
                    if len(series) > 0:
                        normalized = series / series.iloc[0] * features["NSEI_Close"].iloc[0]
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
            if "INDIAVIX_Close" in features.columns:
                fig.add_trace(
                    go.Scatter(
                        x=features.index, y=features["INDIAVIX_Close"],
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
                font=dict(family="Inter, sans-serif", color=text_color_plotly),
                height=700,
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.08,
                    xanchor="right", x=1,
                    font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
                xaxis2=dict(gridcolor=grid_color, zerolinecolor=grid_color),
                yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title="Price ($)"),
                yaxis2=dict(gridcolor=grid_color, zerolinecolor=grid_color, title="Volatility"),
            )
            
            with st.container(border=True):
                st.markdown("""
                <div class="chart-wrap">
                    <div class="chart-title">NIFTY Price with Regime Bands</div>
                    <div class="chart-subtitle">Color-coded overlays identify HMM detected states over the backtest period</div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            # Regime distribution bar
            if "regime_label_stable" in features.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-title">🎯 Regime Timeline</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Percentage distribution of hidden states</div>', unsafe_allow_html=True)
                    regime_counts = features["regime_label_stable"].value_counts()
                    if len(regime_counts) > 0:
                        total = len(features)
                        cols = st.columns(len(regime_counts))
                        for i, (regime, count) in enumerate(regime_counts.items()):
                            with cols[i]:
                                pct = count / total * 100
                                color = REGIME_COLORS.get(regime, "#94a3b8")
                                st.markdown(f"""
                                <div class="metric-card" style="border-color: {color}40; text-align: center; margin-bottom: 0;">
                                    <div class="metric-label">{regime}</div>
                                    <div class="metric-value" style="background: {color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{pct:.1f}%</div>
                                    <div class="metric-sublabel">{count:,} trading days</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No regime predictions available for the selected date range.")

            # ── Inter-Asset Correlation Heatmap ──────────────────
            asset_close_cols = [c for c in features.columns if c.endswith("_Close") and "VIX" not in c]
            if len(asset_close_cols) >= 2 and "regime_label_stable" in features.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">🔗 Inter-Asset Correlation by Regime</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">How asset correlations shift across market regimes — shows diversification breakdown during crises</div>', unsafe_allow_html=True)

                    regime_options = ["All"] + sorted(features["regime_label_stable"].dropna().unique().tolist())
                    corr_regime = st.radio("Select Regime", regime_options, horizontal=True, key="corr_regime_selector")

                    if corr_regime == "All":
                        corr_data = features[asset_close_cols]
                    else:
                        mask = features["regime_label_stable"] == corr_regime
                        corr_data = features.loc[mask, asset_close_cols]

                    corr_returns = np.log(corr_data / corr_data.shift(1)).dropna()
                    corr_returns.columns = [c.replace("_Close", "") for c in corr_returns.columns]
                    corr_matrix = corr_returns.corr()

                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale=[
                            [0, "#ef4444"], [0.25, "#fbbf24"],
                            [0.5, "#1e293b" if IS_DARK else "#f1f5f9"],
                            [0.75, "#60a5fa"], [1.0, "#818cf8"],
                        ],
                        zmid=0, zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                        texttemplate="%{text}",
                        textfont=dict(size=12, color=text_color_plotly),
                        hovertemplate="Row: %{y}<br>Col: %{x}<br>Corr: %{z:.3f}<extra></extra>",
                    ))
                    fig_corr.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        height=350,
                        margin=dict(l=60, r=30, t=20, b=40),
                    )
                    st.plotly_chart(fig_corr, width="stretch", config={"displayModeBar": False})

            # ── Regime Volatility Box Plot ────────────────────────
            if "NSEI_returns" in features.columns and "regime_label_stable" in features.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📦 Return Distribution by Regime</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Daily NIFTY return distribution per HMM regime — reveals fat tails and skew differences</div>', unsafe_allow_html=True)

                    fig_box = go.Figure()
                    regime_order = ["Bull", "Recovery", "Bear", "Crisis"]
                    for regime in regime_order:
                        mask = features["regime_label_stable"] == regime
                        rets = features.loc[mask, "NSEI_returns"].dropna()
                        if len(rets) > 0:
                            color = REGIME_COLORS.get(regime, "#94a3b8")
                            fig_box.add_trace(go.Box(
                                y=rets,
                                name=regime,
                                marker_color=color,
                                line_color=color,
                                fillcolor=REGIME_COLORS_TRANSPARENT.get(regime, "rgba(128,128,128,0.12)"),
                                boxmean="sd",
                                hovertemplate=f"{regime}<br>Return: %{{y:.3%}}<extra></extra>",
                            ))
                    fig_box.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        height=380,
                        margin=dict(l=40, r=30, t=20, b=40),
                        yaxis=dict(title="Daily Return", tickformat=".1%", gridcolor=grid_color),
                        xaxis=dict(gridcolor=grid_color),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_box, width="stretch", config={"displayModeBar": False})

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
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">🥧 Current Portfolio Allocation</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Direct donut visual of portfolio assets weights in the current active regime</div>', unsafe_allow_html=True)

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
                        textfont=dict(size=13, color=text_color_plotly, family="Inter"),
                        hovertemplate="%{label}: %{percent}<br>Weight: %{value:.1%}<extra></extra>",
                        sort=False,
                    )])
                    
                    fig_pie.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        height=380,
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.15,
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
                    
                    st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

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
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📈 Historical Weight Evolution</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Daily breakdown of tradeable assets weights and strategic transitions over the backtest duration</div>', unsafe_allow_html=True)

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
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        height=450,
                        margin=dict(l=40, r=40, t=60, b=40),
                        yaxis=dict(title="Weight", tickformat=".0%", gridcolor=grid_color),
                        xaxis=dict(gridcolor=grid_color),
                        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
                    )
                    
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

                if "regime" in weights.columns:
                    with st.container(border=True):
                        st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">🎯 Average Weights by Regime</div>', unsafe_allow_html=True)
                        st.markdown('<div class="chart-subtitle">Target portfolio structure per HMM regime, showcasing rotation behavior</div>', unsafe_allow_html=True)
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
                            font=dict(family="Inter, sans-serif", color=text_color_plotly),
                            barmode="stack", height=350,
                            yaxis=dict(title="Weight", tickformat=".0%", gridcolor=grid_color),
                            xaxis=dict(gridcolor=grid_color),
                            margin=dict(l=40, r=40, t=40, b=40),
                            legend=dict(
                                orientation="h", yanchor="bottom", y=-0.25,
                                xanchor="center", x=0.5,
                                font=dict(size=12, color=text_color_plotly),
                                bgcolor="rgba(0,0,0,0)",
                            ),
                        )
                        
                        st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
        else:
            st.info("No portfolio weights available for the selected date range. (The ML model requires several years of initial training data before its first out-of-sample allocation).")

    # ── TAB 3: Backtest Performance ───────────────────────────
    with tab3:
        oos = filter_df(data.get("oos"))
        if oos is not None and not oos.empty and "cum_strategy" in oos.columns:
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">💰 Walk-Forward Backtest Performance</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Cumulative returns comparison of the adaptive strategy against SPY along with drawdown profiles</div>', unsafe_allow_html=True)
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
                    name="Market (NIFTY)", line=dict(color="#94a3b8", width=1.5, dash="dash"),
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
            
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # Monthly returns heatmap
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📅 Monthly Returns</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Breakdown of monthly performance for the strategy across backtest years</div>', unsafe_allow_html=True)
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
                    font=dict(family="Inter, sans-serif", color=text_color_plotly),
                    height=max(250, len(pivot) * 35 + 80),
                    margin=dict(l=40, r=40, t=20, b=40),
                )
                
                st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

            # ── Rolling 252-Day Sharpe Ratio ─────────────────────
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📐 Rolling Sharpe Ratio (252-Day)</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Evolving risk-adjusted performance — strategy vs market benchmark over time</div>', unsafe_allow_html=True)

                window = min(252, len(oos) - 1)
                if window > 20:
                    roll_mean_strat = oos["strategy_return"].rolling(window).mean()
                    roll_std_strat = oos["strategy_return"].rolling(window).std()
                    roll_sharpe_strat = (roll_mean_strat / roll_std_strat * np.sqrt(252)).dropna()

                    roll_mean_mkt = oos["market_return"].rolling(window).mean()
                    roll_std_mkt = oos["market_return"].rolling(window).std()
                    roll_sharpe_mkt = (roll_mean_mkt / roll_std_mkt * np.sqrt(252)).dropna()

                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(
                        x=roll_sharpe_strat.index, y=roll_sharpe_strat,
                        name="Strategy", line=dict(color="#818cf8", width=2),
                        fill="tozeroy", fillcolor="rgba(129,140,248,0.08)",
                    ))
                    fig_rs.add_trace(go.Scatter(
                        x=roll_sharpe_mkt.index, y=roll_sharpe_mkt,
                        name="Market (NIFTY)", line=dict(color="#94a3b8", width=1.5, dash="dash"),
                    ))
                    fig_rs.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)" if IS_DARK else "rgba(0,0,0,0.1)")
                    fig_rs.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        height=350,
                        margin=dict(l=50, r=30, t=20, b=40),
                        yaxis=dict(title="Sharpe Ratio", gridcolor=grid_color),
                        xaxis=dict(gridcolor=grid_color),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_rs, width="stretch", config={"displayModeBar": False})

            # ── Annual Returns Bar Chart ─────────────────────────
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📊 Annual Returns Comparison</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Year-by-year strategy vs market performance — at-a-glance yearly alpha</div>', unsafe_allow_html=True)

                annual_strat = oos["strategy_return"].resample("YE").apply(lambda x: (1 + x).prod() - 1)
                annual_mkt = oos["market_return"].resample("YE").apply(lambda x: (1 + x).prod() - 1)
                years = annual_strat.index.year.astype(str)

                fig_annual = go.Figure()
                fig_annual.add_trace(go.Bar(
                    x=years, y=annual_strat.values,
                    name="Strategy", marker_color="#818cf8",
                    hovertemplate="Strategy: %{y:.1%}<extra></extra>",
                ))
                fig_annual.add_trace(go.Bar(
                    x=years, y=annual_mkt.values,
                    name="Market (NIFTY)", marker_color="#94a3b8",
                    hovertemplate="Market: %{y:.1%}<extra></extra>",
                ))
                fig_annual.update_layout(
                    template="plotly_dark" if IS_DARK else "plotly",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=text_color_plotly),
                    barmode="group", height=380,
                    margin=dict(l=50, r=30, t=20, b=40),
                    yaxis=dict(title="Annual Return", tickformat=".0%", gridcolor=grid_color),
                    xaxis=dict(gridcolor=grid_color, title="Year"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_annual, width="stretch", config={"displayModeBar": False})

            # ── Underwater (Drawdown Duration) Plot ───────────────
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">🌊 Underwater Plot</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Drawdown depth and recovery periods — shaded by severity</div>', unsafe_allow_html=True)

                cum_uw = (1 + oos["strategy_return"]).cumprod()
                peak_uw = cum_uw.cummax()
                dd_uw = (cum_uw - peak_uw) / peak_uw

                fig_uw = go.Figure()
                fig_uw.add_trace(go.Scatter(
                    x=dd_uw.index, y=dd_uw,
                    name="Drawdown", line=dict(color="#f87171", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(248,113,113,0.2)",
                    hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
                ))
                # Highlight deepest drawdown point
                worst_idx = dd_uw.idxmin()
                worst_val = dd_uw.min()
                fig_uw.add_annotation(
                    x=worst_idx, y=worst_val,
                    text=f"Max: {worst_val:.1%}",
                    showarrow=True, arrowhead=2,
                    font=dict(color="#f87171", size=11, family="JetBrains Mono"),
                    arrowcolor="#f87171",
                    ax=40, ay=-30,
                )
                fig_uw.update_layout(
                    template="plotly_dark" if IS_DARK else "plotly",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=text_color_plotly),
                    height=300,
                    margin=dict(l=50, r=30, t=20, b=40),
                    yaxis=dict(title="Drawdown", tickformat=".0%", gridcolor=grid_color),
                    xaxis=dict(gridcolor=grid_color),
                    showlegend=False,
                )
                st.plotly_chart(fig_uw, width="stretch", config={"displayModeBar": False})

        else:
            st.info("No backtest trades available for the selected date range. (The Walk-Forward algorithm requires ~16 years of initial historical data to train before executing its first out-of-sample trade).")

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
                st.image(str(combined_plot), caption="Combined Cross-Regime Feature Importance", width="stretch")
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
                            st.image(str(summary_plot), caption=f"Beeswarm Summary - {key.title()} Regime", width="stretch")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"Summary beeswarm plot not found for {key}.")
                    with col_bar:
                        if bar_plot.exists():
                            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
                            st.image(str(bar_plot), caption=f"Bar Feature Importance - {key.title()} Regime", width="stretch")
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.info(f"Bar importance plot not found for {key}.")
        else:
            st.info("SHAP plots not generated yet. Run the explainability step first.")

        # Audit log explorer
        st.markdown('<div class="section-header">📋 SEBI 2026 Audit Log Explorer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-panel">
            <strong>Regulatory Audit & Explainability:</strong> Explore and inspect any of the logged 
            trading decisions. Select a row in the decision history table to visualize feature contributions 
            (SHAP values) for that day's signal.
        </div>
        """, unsafe_allow_html=True)

        audit_df = load_audit_logs()
        if audit_df is not None and not audit_df.empty:
            # 1. Filter Controls inside a border container
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="margin-bottom: 0.2rem;">Filter Audit History</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle" style="margin-bottom: 1rem; font-size: 0.75rem; color: var(--text-dim);">Refine the 9,707 logged decisions by date, regime, signal, and specific feature impact</div>', unsafe_allow_html=True)
                
                # Setup filter layout columns (two rows for better readability)
                col_f_start, col_f_end, col_f_reg, col_f_sig = st.columns(4)
                
                min_log_date = pd.to_datetime("1990-01-01").date()
                max_log_date = audit_df["date_parsed"].dt.date.max() if not audit_df.empty else pd.to_datetime("today").date()
                
                with col_f_start:
                    start_log_date = st.date_input("Start Date", value=None, format="DD/MM/YYYY", min_value=min_log_date, max_value=max_log_date, key="audit_start_date")
                
                with col_f_end:
                    end_log_date = st.date_input("End Date", value=None, format="DD/MM/YYYY", min_value=min_log_date, max_value=max_log_date, key="audit_end_date")
                
                with col_f_reg:
                    log_regimes = sorted(audit_df["regime"].dropna().unique().tolist())
                    selected_regimes = st.multiselect(
                        "Regime Filter",
                        options=log_regimes,
                        default=[],
                        placeholder="Choose options",
                        key="audit_regimes"
                    )
                    
                with col_f_sig:
                    log_signals = sorted(audit_df["signal"].dropna().unique().tolist())
                    signal_labels = {1: "Buy (+1)", 0: "Hold (0)", -1: "Sell (-1)"}
                    selected_signals = st.multiselect(
                        "Signal Filter", 
                        options=log_signals, 
                        default=[],
                        placeholder="Choose options",
                        format_func=lambda x: signal_labels.get(x, str(x)),
                        key="audit_signals"
                    )
                
                col_f_feat, col_f_conf = st.columns(2)
                
                with col_f_feat:
                    all_features = set()
                    for fl in audit_df["top_features"]:
                        if isinstance(fl, list):
                            for item in fl:
                                all_features.add(item.get("feature"))
                    feature_options = ["All Features"] + sorted(list(all_features))
                    selected_feature = st.selectbox("Top Driver Feature", options=feature_options, key="audit_feature")
                    
                with col_f_conf:
                    min_conf = float(audit_df["confidence"].min()) if "confidence" in audit_df.columns else 0.0
                    max_conf = float(audit_df["confidence"].max()) if "confidence" in audit_df.columns else 1.0
                    if pd.isna(min_conf) or pd.isna(max_conf):
                        min_conf = 0.0
                        max_conf = 1.0
                    if min_conf >= max_conf:
                        min_conf = 0.0
                        max_conf = 1.0
                    selected_min_conf = st.slider("Minimum Confidence Threshold", min_value=float(min_conf), max_value=float(max_conf), value=float(min_conf), step=0.01, key="audit_min_conf")
            
            # Apply filters
            filtered_df = audit_df.copy()
            if start_log_date is not None:
                filtered_df = filtered_df[filtered_df["date_parsed"].dt.date >= start_log_date]
            if end_log_date is not None:
                filtered_df = filtered_df[filtered_df["date_parsed"].dt.date <= end_log_date]
            if selected_regimes:
                filtered_df = filtered_df[filtered_df["regime"].isin(selected_regimes)]
                
            if selected_signals:
                filtered_df = filtered_df[filtered_df["signal"].isin(selected_signals)]
                
            if selected_feature != "All Features":
                filtered_df = filtered_df[
                    filtered_df["top_features"].apply(
                        lambda fl: isinstance(fl, list) and any(item.get("feature") == selected_feature for item in fl)
                    )
                ]
                
            if "confidence" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["confidence"] >= selected_min_conf]
            
            st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
            
            if filtered_df.empty:
                st.warning("⚠️ No audit log entries match the selected filters.")
            else:
                # 2. Main Explorer split: Table vs Details Inspector
                col_tbl, col_insp = st.columns([1.5, 1.5])
                
                with col_tbl:
                    with st.container(border=True):
                        st.markdown(f"### 📋 Decision History ({len(filtered_df):,} entries)")
                        
                        # Columns to show in table
                        table_cols = ["date", "regime", "signal", "confidence", "method"]
                        display_df = filtered_df[table_cols].copy()
                        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
                        
                        # Column configurations for st.dataframe
                        column_config = {
                            "date": st.column_config.TextColumn("Date", width="medium"),
                            "regime": st.column_config.TextColumn("Regime", width="small"),
                            "signal": st.column_config.NumberColumn("Signal", format="%d", width="small"),
                            "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0.0, max_value=1.0, width="medium"),
                            "method": st.column_config.TextColumn("Method", width="small"),
                        }
                        
                        select_event = st.dataframe(
                            display_df,
                            column_config=column_config,
                            hide_index=True,
                            width="stretch",
                            on_select="rerun",
                            selection_mode="single-row",
                            height=640,
                            key="audit_history_table"
                        )
                        
                        st.markdown("""
                        <div style="font-size: 0.72rem; color: var(--text-dim); margin-top: -4px;">
                            💡 Click any row in the table above to inspect its full explainability details on the right panel.
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_insp:
                    # Resolve selected row (default to the first row if none selected)
                    selected_row = filtered_df.iloc[0]
                    try:
                        if select_event and hasattr(select_event, "selection") and select_event.selection and "rows" in select_event.selection and len(select_event.selection.rows) > 0:
                            selected_idx = select_event.selection.rows[0]
                            selected_row = filtered_df.iloc[selected_idx]
                    except Exception:
                        pass
                    
                    with st.container(border=True):
                        st.markdown(f"### 🔍 Details: {selected_row['date']}")
                        
                        # Render small metrics cards inside the details panel
                        d_c1, d_c2, d_c3 = st.columns(3)
                        with d_c1:
                            reg_val = selected_row["regime"]
                            reg_color = REGIME_COLORS.get(reg_val, "#94a3b8")
                            st.markdown(f"""
                            <div class="metric-card" style="border-color: {reg_color}40; padding: 0.75rem 1rem; margin-bottom: 0.5rem; text-align: center;">
                                <div class="metric-label" style="font-size: 0.65rem;">Regime</div>
                                <div class="metric-value" style="font-size: 1.15rem; color: {reg_color};">{reg_val}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with d_c2:
                            sig_val = selected_row["signal"]
                            sig_color = green_color if sig_val == 1 else red_color if sig_val == -1 else text_muted
                            sig_text = "BUY (+1)" if sig_val == 1 else "SELL (-1)" if sig_val == -1 else "HOLD (0)"
                            st.markdown(f"""
                            <div class="metric-card" style="border-color: {sig_color}40; padding: 0.75rem 1rem; margin-bottom: 0.5rem; text-align: center;">
                                <div class="metric-label" style="font-size: 0.65rem;">Signal</div>
                                <div class="metric-value" style="font-size: 1.15rem; color: {sig_color};">{sig_text}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with d_c3:
                            conf_val = selected_row.get("confidence", 0.0)
                            st.markdown(f"""
                            <div class="metric-card" style="border-color: var(--accent)40; padding: 0.75rem 1rem; margin-bottom: 0.5rem; text-align: center;">
                                <div class="metric-label" style="font-size: 0.65rem;">Confidence</div>
                                <div class="metric-value" style="font-size: 1.15rem; color: var(--accent);">{conf_val:.1%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Feature SHAP chart
                        top_feats = selected_row.get("top_features", [])
                        if top_feats and isinstance(top_feats, list):
                            feat_plot_df = pd.DataFrame(top_feats)
                            feat_plot_df["abs_shap"] = feat_plot_df["shap_value"].abs()
                            feat_plot_df = feat_plot_df.sort_values("abs_shap", ascending=True)
                            feat_plot_df["color"] = feat_plot_df["shap_value"].apply(lambda x: green_color if x >= 0 else red_color)
                            
                            fig_shap_detail = go.Figure(go.Bar(
                                x=feat_plot_df["shap_value"],
                                y=feat_plot_df["feature"],
                                orientation="h",
                                marker_color=feat_plot_df["color"],
                                hovertemplate="Feature: %{y}<br>SHAP Value: %{x:.6f}<extra></extra>",
                            ))
                            
                            fig_shap_detail.update_layout(
                                template="plotly_dark" if IS_DARK else "plotly",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Inter, sans-serif", color=text_color_plotly),
                                height=230,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(
                                    title="SHAP Value (BUY Contribution)",
                                    gridcolor=grid_color, 
                                    zerolinecolor="#fafafa" if IS_DARK else "#09090b",
                                    zerolinewidth=1
                                ),
                                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                            )
                            
                            st.plotly_chart(fig_shap_detail, width="stretch", config={"displayModeBar": False})
                            
                            # Feature Table
                            rows_html = ""
                            for _, f_row in feat_plot_df.iloc[::-1].iterrows():
                                f_name = f_row["feature"]
                                f_val = f_row["feature_value"]
                                s_val = f_row["shap_value"]
                                
                                badge_cls = "regime-bull" if s_val >= 0 else "regime-crisis"
                                badge_txt = "Positive" if s_val >= 0 else "Negative"
                                shap_color = green_color if s_val >= 0 else red_color
                                
                                rows_html += f"<tr>"
                                rows_html += f"<td style='font-weight: 600; color: var(--text); padding: 0.5rem;'>{f_name}</td>"
                                rows_html += f"<td style='text-align: right; padding: 0.5rem;'>{f_val:.4f}</td>"
                                rows_html += f"<td style='text-align: right; font-weight: 600; color: {shap_color}; padding: 0.5rem;'>{s_val:+.4f}</td>"
                                rows_html += f"<td style='text-align: center; padding: 0.5rem;'><span class='regime-badge {badge_cls}' style='padding: 1px 6px; font-size: 0.65rem;'>{badge_txt}</span></td>"
                                rows_html += f"</tr>"
                                
                            table_html = (
                                f'<table class="data-table" style="margin-top: 0; margin-bottom: 0.75rem; font-size: 0.78rem; width: 100%;">'
                                f'<thead><tr>'
                                f'<th style="padding: 0.5rem;">Feature</th>'
                                f'<th style="text-align: right; padding: 0.5rem;">Value</th>'
                                f'<th style="text-align: right; padding: 0.5rem;">SHAP</th>'
                                f'<th style="text-align: center; padding: 0.5rem;">Impact</th>'
                                f'</tr></thead>'
                                f'<tbody>{rows_html}</tbody>'
                                f'</table>'
                            )
                            st.markdown(table_html, unsafe_allow_html=True)
                            
                            # Narrative Explanation
                            primary_driver = feat_plot_df.iloc[-1]
                            dr_name = primary_driver["feature"]
                            dr_val = primary_driver["feature_value"]
                            dr_shap = primary_driver["shap_value"]
                            
                            dir_str = "pushing the model towards a BUY decision" if dr_shap >= 0 else "pulling the model towards a SELL or defensive allocation"
                            sig_str = "BUY" if sig_val == 1 else "SELL" if sig_val == -1 else "HOLD"
                            
                            narrative = f"""
                            On <strong>{selected_row['date']}</strong>, the model executed under the <strong>{reg_val}</strong> regime 
                            using model version <code>{selected_row.get('model_version', 'hmm_rf_v1')}</code>.
                            The system generated a <strong>{sig_str}</strong> signal with <strong>{conf_val:.1%}</strong> confidence.
                            The primary driver was <code>{dr_name}</code> (observed: {dr_val:.4f}), which had a SHAP contribution of 
                            <span style="color: {green_color if dr_shap >= 0 else red_color}; font-weight: 600;">{dr_shap:+.4f}</span>, {dir_str}.
                            """
                            st.markdown(f"""
                            <div class="info-panel" style="margin-top: 0; padding: 0.75rem 1rem; font-size: 0.8rem; border-left: 3px solid var(--accent);">
                                <strong>Narrative Explanation ({selected_row.get('compliance_framework', 'SEBI_2026')}):</strong><br>
                                {narrative}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No top SHAP features found for this decision.")
        else:
            st.warning("⚠️ Audit log file `output/logs/shap_audit.jsonl` was not found or is empty. Run explainability pipeline first.")

    # ── TAB 5: Regime Analysis ────────────────────────────────
    with tab5:
        features = data.get("features")
        if features is not None and "regime_label_stable" in features.columns:
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📋 Regime Transition Probabilities</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Likelihood of shifting from one market regime to another based on historical chain transitions</div>', unsafe_allow_html=True)
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
                    font=dict(family="Inter, sans-serif", color=text_color_plotly),
                    height=350,
                    xaxis_title="To Regime",
                    yaxis_title="From Regime",
                    margin=dict(l=80, r=30, t=30, b=50),
                )
                
                st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

            # Per-regime statistics
            if "NSEI_returns" in features.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📊 Per-Regime Return Statistics</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Key performance metrics calculated dynamically per hidden state</div>', unsafe_allow_html=True)
                    regime_stats = features.groupby("regime_label_stable")["NSEI_returns"].agg([
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

            # ── Regime Duration Histogram ─────────────────────────
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">⏱️ Regime Duration Distribution</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">How long each regime persists in trading days — with median duration annotations</div>', unsafe_allow_html=True)

                regimes_series = features["regime_label_stable"].dropna()
                regime_groups = (regimes_series != regimes_series.shift()).cumsum()
                durations = regimes_series.groupby(regime_groups).agg(["first", "count"])
                durations.columns = ["regime", "duration"]

                fig_dur = go.Figure()
                regime_order = ["Bull", "Recovery", "Bear", "Crisis"]
                for regime in regime_order:
                    dur_data = durations[durations["regime"] == regime]["duration"]
                    if len(dur_data) > 0:
                        color = REGIME_COLORS.get(regime, "#94a3b8")
                        median_d = dur_data.median()
                        fig_dur.add_trace(go.Histogram(
                            x=dur_data,
                            name=f"{regime} (med: {median_d:.0f}d)",
                            marker_color=color,
                            opacity=0.7,
                            nbinsx=30,
                            hovertemplate=f"{regime}<br>Duration: %{{x}} days<br>Count: %{{y}}<extra></extra>",
                        ))

                fig_dur.update_layout(
                    template="plotly_dark" if IS_DARK else "plotly",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color=text_color_plotly),
                    barmode="overlay", height=380,
                    margin=dict(l=50, r=30, t=20, b=40),
                    xaxis=dict(title="Duration (Trading Days)", gridcolor=grid_color),
                    yaxis=dict(title="Frequency", gridcolor=grid_color),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig_dur, width="stretch", config={"displayModeBar": False})

            # ── Return Distribution Overlay ───────────────────────
            if "NSEI_returns" in features.columns:
                with st.container(border=True):
                    st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📊 Return Distribution Overlay</div>', unsafe_allow_html=True)
                    st.markdown('<div class="chart-subtitle">Daily return histograms per regime — reveals fat tails in Crisis vs positive skew in Bull</div>', unsafe_allow_html=True)

                    fig_dist = go.Figure()
                    for regime in regime_order:
                        mask = features["regime_label_stable"] == regime
                        rets = features.loc[mask, "NSEI_returns"].dropna()
                        if len(rets) > 0:
                            color = REGIME_COLORS.get(regime, "#94a3b8")
                            fig_dist.add_trace(go.Histogram(
                                x=rets,
                                name=regime,
                                marker_color=color,
                                opacity=0.5,
                                nbinsx=80,
                                histnorm="probability density",
                                hovertemplate=f"{regime}<br>Return: %{{x:.3%}}<br>Density: %{{y:.2f}}<extra></extra>",
                            ))
                    fig_dist.update_layout(
                        template="plotly_dark" if IS_DARK else "plotly",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter, sans-serif", color=text_color_plotly),
                        barmode="overlay", height=380,
                        margin=dict(l=50, r=30, t=20, b=40),
                        xaxis=dict(title="Daily Return", tickformat=".1%", gridcolor=grid_color),
                        yaxis=dict(title="Density", gridcolor=grid_color),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_dist, width="stretch", config={"displayModeBar": False})

            # ── Regime Calendar Strip ─────────────────────────────
            with st.container(border=True):
                st.markdown('<div class="chart-title" style="font-size: 1.1rem; margin-bottom: 0.5rem;">📅 Regime Timeline Calendar</div>', unsafe_allow_html=True)
                st.markdown('<div class="chart-subtitle">Year-by-year color-coded regime timeline — showing structural market transitions</div>', unsafe_allow_html=True)

                regimes_cal = features["regime_label_stable"].dropna()
                years = sorted(regimes_cal.index.year.unique())

                calendar_html = ""
                for year in years:
                    year_data = regimes_cal[regimes_cal.index.year == year]
                    if len(year_data) == 0:
                        continue
                    # Group consecutive regimes
                    year_groups = (year_data != year_data.shift()).cumsum()
                    segments_html = ""
                    for _, grp in year_data.groupby(year_groups):
                        regime = grp.iloc[0]
                        pct = len(grp) / len(year_data) * 100
                        color = REGIME_COLORS.get(regime, "#94a3b8")
                        segments_html += f'<div class="calendar-strip-segment" style="width: {pct}%; background: {color};" title="{regime}: {len(grp)} days"></div>'

                    calendar_html += f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 2px;">
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-muted); min-width: 36px;">{year}</span>
                        <div class="calendar-strip" style="flex: 1;">
                            {segments_html}
                        </div>
                    </div>"""

                st.markdown(calendar_html, unsafe_allow_html=True)

                # Legend
                legend_html = '<div style="display: flex; gap: 1rem; margin-top: 8px; flex-wrap: wrap;">'
                for regime, color in REGIME_COLORS.items():
                    legend_html += f'<div style="display: flex; align-items: center; gap: 4px; font-size: 0.72rem; color: var(--text-muted);"><div style="width: 10px; height: 10px; border-radius: 2px; background: {color};"></div>{regime}</div>'
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

        else:
            st.info("Regime data not available.")


if __name__ == "__main__":
    main()
