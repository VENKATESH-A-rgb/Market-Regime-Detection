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

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detection — Adaptive Portfolio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global theme */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }
    .metric-value.positive {
        background: linear-gradient(135deg, #34d399, #10b981, #6ee7b7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value.negative {
        background: linear-gradient(135deg, #f87171, #ef4444, #fca5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    .metric-sublabel {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.4);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.1), rgba(59, 130, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #c7d2fe, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 8px 0;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Regime badges */
    .regime-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .regime-bull { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .regime-recovery { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .regime-bear { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .regime-crisis { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }

    /* Info panel */
    .info-panel {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Plotly chart container */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        padding: 8px 20px;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        border-color: rgba(99, 102, 241, 0.4);
    }
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


def render_metric_card(label, value, sublabel="", css_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
        <div class="metric-sublabel">{sublabel}</div>
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

    # ── Hero Banner ───────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <h1>🔮 Market Regime Detection</h1>
        <p>Adaptive Portfolio Allocation powered by HMM regime classification,
        regime-specialist Random Forests, and FFD-stationarized features.
        Walk-Forward validated with transaction costs.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")

        features = data.get("features")
        if features is not None and not features.empty:
            min_date = features.index.min().date()
            max_date = features.index.max().date()
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            date_range = (start_date, end_date)
        else:
            date_range = None

        st.markdown("---")
        st.markdown("### 📊 Regime Legend")
        for regime, color in REGIME_COLORS.items():
            st.markdown(f'<span class="regime-badge regime-{regime.lower()}">{regime}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏗️ Pipeline Info")
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

    # ── Filter by date range ──────────────────────────────────
    def filter_df(df):
        if date_range and len(date_range) == 2 and df is not None:
            mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
            return df[mask]
        return df

    # ── Metrics Cards ─────────────────────────────────────────
    if "metrics" in data:
        m = data["metrics"]
        cols = st.columns(5)
        with cols[0]:
            val = m.get("sharpe_ratio", 0)
            css = "positive" if val > 0.5 else "negative" if val < 0 else ""
            render_metric_card("Sharpe Ratio", f"{val:.2f}", f"Market: {m.get('market_sharpe', 0):.2f}", css)
        with cols[1]:
            val = m.get("max_drawdown", 0)
            render_metric_card("Max Drawdown", f"{val:.1%}", f"Market: {m.get('market_max_drawdown', 0):.1%}", "negative")
        with cols[2]:
            val = m.get("annual_return", 0)
            css = "positive" if val > 0 else "negative"
            render_metric_card("Annual Return", f"{val:.1%}", f"Total: {m.get('total_return', 0):.1%}", css)
        with cols[3]:
            val = m.get("win_rate", 0)
            css = "positive" if val > 0.5 else ""
            render_metric_card("Win Rate", f"{val:.1%}", f"Profit Factor: {m.get('profit_factor', 0):.2f}", css)
        with cols[4]:
            val = m.get("calmar_ratio", 0)
            css = "positive" if val > 0.5 else ""
            render_metric_card("Calmar Ratio", f"{val:.2f}", f"Trades: {m.get('total_trades', 0)}", css)

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
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 14, 26, 0.8)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=700,
                margin=dict(l=60, r=30, t=50, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11),
                    bgcolor="rgba(30, 41, 59, 0.7)",
                ),
                xaxis2=dict(gridcolor="rgba(99, 102, 241, 0.1)"),
                yaxis=dict(gridcolor="rgba(99, 102, 241, 0.1)", title="Price ($)"),
                yaxis2=dict(gridcolor="rgba(99, 102, 241, 0.1)", title="Volatility"),
            )
            st.plotly_chart(fig, use_container_width=True)

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
                regime_css = current_regime.lower() if current_regime in REGIME_COLORS else "bear"

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
                            line=dict(color="#0a0e1a", width=2),
                        ),
                        textinfo="label+percent",
                        textfont=dict(size=13, color="#e2e8f0", family="Inter"),
                        hovertemplate="%{label}: %{percent}<br>Weight: %{value:.1%}<extra></extra>",
                        sort=False,
                    )])
                    fig_pie.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#e2e8f0"),
                        height=380,
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(
                            orientation="h", yanchor="bottom", y=-0.1,
                            xanchor="center", x=0.5,
                            font=dict(size=12, color="#94a3b8"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        annotations=[dict(
                            text=f"<b>{current_regime}</b>",
                            x=0.5, y=0.5, font_size=18,
                            font_color=REGIME_COLORS.get(current_regime, "#94a3b8"),
                            showarrow=False,
                        )],
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

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
                            <div style="display: flex; justify-content: space-between; color: #cbd5e1; font-size: 0.85rem;">
                                <span style="color: {color}; font-weight: 600;">{asset_name}</span>
                                <span>{w_val:.1%}</span>
                            </div>
                            <div style="background: rgba(30,41,59,0.5); border-radius: 4px; height: 6px; margin-top: 4px;">
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
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(10, 14, 26, 0.8)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    height=450,
                    margin=dict(l=60, r=30, t=30, b=30),
                    yaxis=dict(title="Weight", tickformat=".0%", gridcolor="rgba(99,102,241,0.1)"),
                    xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

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
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10, 14, 26, 0.8)",
                        font=dict(family="Inter", color="#e2e8f0"),
                        barmode="stack", height=350,
                        yaxis=dict(title="Weight", tickformat=".0%", gridcolor="rgba(99,102,241,0.1)"),
                        margin=dict(l=60, r=30, t=30, b=30),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
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
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 14, 26, 0.8)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=600,
                margin=dict(l=60, r=30, t=50, b=30),
                yaxis=dict(title="Cumulative Return", gridcolor="rgba(99,102,241,0.1)"),
                yaxis2=dict(title="Drawdown", tickformat=".0%", gridcolor="rgba(99,102,241,0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

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
                    [0.5, "#1e293b"], [0.7, "#34d399"], [1.0, "#10b981"],
                ],
                zmid=0,
                text=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2%}<extra></extra>",
            ))
            fig3.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 14, 26, 0.8)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=max(250, len(pivot) * 30 + 80),
                margin=dict(l=60, r=30, t=20, b=30),
            )
            st.plotly_chart(fig3, use_container_width=True)
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

        # Load SHAP plots
        shap_files = list(OUTPUT_PLOTS.glob("shap_*.png")) if OUTPUT_PLOTS.exists() else []
        if shap_files:
            for plot_file in sorted(shap_files):
                st.image(str(plot_file), caption=plot_file.stem.replace("_", " ").title(), use_container_width=True)
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

            st.markdown("#### Regime Transition Probabilities")
            fig4 = go.Figure(data=go.Heatmap(
                z=transitions.values,
                x=transitions.columns.astype(str),
                y=transitions.index.astype(str),
                colorscale=[[0, "#0f172a"], [0.5, "#6366f1"], [1, "#a78bfa"]],
                text=[[f"{v:.1%}" for v in row] for row in transitions.values],
                texttemplate="%{text}",
                textfont=dict(size=12, color="#e2e8f0"),
                hovertemplate="From: %{y}<br>To: %{x}<br>Prob: %{z:.2%}<extra></extra>",
            ))
            fig4.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(10, 14, 26, 0.8)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=350,
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                margin=dict(l=80, r=30, t=30, b=50),
            )
            st.plotly_chart(fig4, use_container_width=True)

            # Per-regime statistics
            st.markdown("#### Per-Regime Return Statistics")
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

                st.dataframe(display_stats, use_container_width=True)
        else:
            st.info("Regime data not available.")


if __name__ == "__main__":
    main()
