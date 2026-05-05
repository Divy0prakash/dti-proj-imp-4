"""
╔══════════════════════════════════════════════════════════════════════╗
║          DTI — Dynamic Discount Optimizer | Streamlit Dashboard      ║
║          Production-grade ML Dashboard with Advanced Analytics       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DTI — Dynamic Discount Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark industrial theme with amber/gold accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0f14;
    color: #e8e2d4;
}

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111520 50%, #0d0f14 100%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
    border-right: 1px solid #f59e0b22;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f59e0b;
    font-family: 'Syne', sans-serif;
}

/* ── Headers ── */
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── KPI Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #f59e0b33;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #f59e0b88;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #f59e0b, transparent);
}
.kpi-label {
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8892a4;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
}
.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #f59e0b;
    font-family: 'Syne', sans-serif;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    color: #556070;
    margin-top: 6px;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 0 8px 0;
    border-bottom: 1px solid #f59e0b22;
    margin-bottom: 20px;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #e8e2d4;
    letter-spacing: -0.01em;
}
.section-badge {
    background: #f59e0b15;
    border: 1px solid #f59e0b44;
    color: #f59e0b;
    font-size: 10px;
    letter-spacing: 0.12em;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
}

/* ── Dashboard Title ── */
.dashboard-title {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #f59e0b 0%, #fcd34d 50%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 0;
}
.dashboard-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    color: #556070;
    letter-spacing: 0.1em;
    margin-top: 6px;
}

/* ── Metric Override ── */
[data-testid="stMetric"] {
    background: transparent !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #f59e0b22;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #f59e0b22;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.08em;
    color: #556070;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    color: #f59e0b !important;
    background: #f59e0b11 !important;
    border-color: #f59e0b33 !important;
    border-bottom-color: transparent !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding: 0;
}

/* ── Buttons ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0d0f14 !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
}

/* ── Search box ── */
.stTextInput input {
    background: #161b27 !important;
    border: 1px solid #f59e0b33 !important;
    border-radius: 6px !important;
    color: #e8e2d4 !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput input:focus {
    border-color: #f59e0b88 !important;
    box-shadow: 0 0 0 2px #f59e0b15 !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] {
    background: #161b27 !important;
}

/* ── Alert / info boxes ── */
.stAlert {
    background: #161b27 !important;
    border: 1px solid #f59e0b33 !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: #f59e0b15 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #f59e0b44; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #f59e0b88; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE — Matches dark theme
# ─────────────────────────────────────────────────────────────────────────────
# Plain dicts — avoids go.Layout.to_plotly_json() removed in newer Plotly
# Split into base (no axes) + axis defaults to avoid duplicate-kwarg errors
# Base layout kwargs — deliberately excludes xaxis/yaxis/legend/colorway
# so callers can pass those without duplicate-kwarg errors.
LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,39,0.6)",
    font=dict(family="DM Mono, monospace", color="#8892a4", size=11),
    title_font=dict(family="Syne, sans-serif", color="#e8e2d4", size=15),
    margin=dict(l=40, r=20, t=50, b=40),
)

_AXIS_STYLE = dict(
    gridcolor="#1e2535", gridwidth=1,
    linecolor="#2a3347", tickcolor="#2a3347",
    title_font=dict(color="#8892a4"),
)

_LEGEND_STYLE = dict(
    bgcolor="rgba(22,27,39,0.8)",
    bordercolor="rgba(245,158,11,0.2)",
    borderwidth=1,
)

_COLORWAY = ["#f59e0b", "#34d399", "#60a5fa", "#f472b6", "#a78bfa", "#fb923c"]


def apply_theme(fig: go.Figure, **extra) -> go.Figure:
    """
    Apply the dark theme to a figure.
    `extra` kwargs are forwarded to update_layout — pass title, height,
    xaxis_title, yaxis_title, legend=dict(...), etc. here.
    Legend style is merged (not replaced) so callers can override orientation.
    """
    # Merge legend: default style + any caller overrides
    caller_legend = extra.pop("legend", {})
    merged_legend = {**_LEGEND_STYLE, **caller_legend}

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        colorway=_COLORWAY,
        legend=merged_legend,
        **extra,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig

AMBER_SCALE = [
    [0.0, "#1a2035"],
    [0.25, "#7c3a00"],
    [0.5, "#b45309"],
    [0.75, "#d97706"],
    [1.0, "#fcd34d"],
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_unified_dataset(path: str = "unified_dataset.csv") -> pd.DataFrame | None:
    """Load the unified feature-engineered dataset."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_recommendations(path: str = "top_recommendations.csv") -> pd.DataFrame | None:
    """Load the top recommendations with optimised discounts."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_resource(show_spinner=False)
def load_model_and_scaler(
    model_path: str = "xgboost_model.json",
    scaler_path: str = "scaler.pkl",
):
    """Load XGBoost model and scaler (resource-cached)."""
    model, scaler = None, None
    try:
        import xgboost as xgb
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
    except Exception:
        pass
    try:
        import pickle
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
    except Exception:
        pass
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR — Used when real files are not present
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic demo datasets so the dashboard renders fully
    even without the real CSV files present.
    """
    rng = np.random.default_rng(42)
    n = 500

    categories = ["Electronics", "Apparel", "Home & Kitchen", "Sports", "Books", "Beauty", "Toys"]
    seasons = ["winter", "summer", "spring", "autumn"]

    unified = pd.DataFrame({
        "product_id": [f"P{str(i).zfill(5)}" for i in range(1, n + 1)],
        "category": rng.choice(categories, n),
        "price": rng.uniform(50, 5000, n).round(2),
        "units_sold": rng.integers(5, 500, n),
        "rating": rng.uniform(1.0, 5.0, n).round(2),
        "sentiment_score": rng.uniform(-1.0, 1.0, n).round(4),
        "pop_score": rng.uniform(0.0, 1.0, n).round(4),
        "season": rng.choice(seasons, n),
        "festival": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "discount_pct": rng.uniform(0, 50, n).round(2),
        "margin_pct": rng.uniform(5, 40, n).round(2),
    })
    unified["effective_price"] = (unified["price"] * (1 - unified["discount_pct"] / 100)).round(2)
    unified["revenue"] = (unified["effective_price"] * unified["units_sold"]).round(2)

    # Top recommendations — subset with higher pop_score
    top_idx = rng.choice(n, 120, replace=False)
    recs = unified.iloc[top_idx].copy()
    recs["recommended_discount_pct"] = (recs["discount_pct"] * rng.uniform(0.8, 1.3, len(recs))).clip(0, 60).round(2)
    recs["effective_price"] = (recs["price"] * (1 - recs["recommended_discount_pct"] / 100)).round(2)
    recs["revenue_impact"] = (recs["effective_price"] * recs["units_sold"] * rng.uniform(1.0, 1.3, len(recs))).round(2)

    return unified, recs


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def section(icon: str, title: str, badge: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""<div class="section-header">
            <span style="font-size:20px">{icon}</span>
            <span class="section-title">{title}</span>
            {badge_html}
        </div>""",
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def fmt_pct(v): return f"{v:.1f}%"
def fmt_currency(v): return f"₹{v:,.0f}"
def fmt_float(v): return f"{v:.3f}"


def _color_sentiment(val: float) -> str:
    """Cell background: red → grey → green based on sentiment (-1 to 1)."""
    try:
        v = max(-1.0, min(1.0, float(val)))
        if v >= 0:
            g = int(80 + v * 120)
            r = int(40 + (1 - v) * 60)
            return f"background-color: rgba({r},{g},70,0.35); color: #e8e2d4"
        else:
            r = int(180 + abs(v) * 60)
            g = int(80 - abs(v) * 40)
            return f"background-color: rgba({r},{g},60,0.35); color: #e8e2d4"
    except (TypeError, ValueError):
        return ""


def _color_discount(val: float, vmax: float) -> str:
    """Cell background: dark → amber based on discount value."""
    try:
        ratio = max(0.0, min(1.0, float(val) / max(float(vmax), 1.0)))
        r = int(40 + ratio * 205)
        g = int(20 + ratio * 118)
        b = 10
        return f"background-color: rgba({r},{g},{b},0.45); color: #e8e2d4"
    except (TypeError, ValueError):
        return ""


def _color_pop(val: float) -> str:
    """Cell background: dark-blue → bright-blue based on pop score (0-1)."""
    try:
        ratio = max(0.0, min(1.0, float(val)))
        r = int(20 + ratio * 40)
        g = int(60 + ratio * 100)
        b = int(120 + ratio * 135)
        return f"background-color: rgba({r},{g},{b},0.40); color: #e8e2d4"
    except (TypeError, ValueError):
        return ""


def style_dataframe(df: pd.DataFrame):
    """
    Apply cell-level colour styling — no matplotlib dependency.
    Uses Styler.map() (pandas >= 2.1) with applymap() fallback.
    """
    styler = df.style

    discount_col  = next((c for c in df.columns if "discount" in c), None)
    sentiment_col = next((c for c in df.columns if "sentiment" in c), None)
    pop_col       = next((c for c in df.columns if "pop" in c), None)

    # map() is the pandas >= 2.1 name; applymap() is the older alias
    _map = getattr(styler, "map", getattr(styler, "applymap", None))

    if _map and sentiment_col and sentiment_col in df.columns:
        _map(_color_sentiment, subset=[sentiment_col])

    if _map and discount_col and discount_col in df.columns:
        vmax = float(df[discount_col].max())
        _map(lambda v: _color_discount(v, vmax), subset=[discount_col])

    if _map and pop_col and pop_col in df.columns:
        _map(_color_pop, subset=[pop_col])

    # Number formatting
    fmt: dict = {}
    for c in df.columns:
        if "price" in c:
            fmt[c] = "₹{:,.2f}"
        elif "discount" in c or c.endswith("_pct"):
            fmt[c] = "{:.1f}%"
        elif "sentiment" in c or "score" in c or c == pop_col:
            fmt[c] = "{:.3f}"

    if fmt:
        styler = styler.format(fmt, na_rep="—")

    return styler


# ─────────────────────────────────────────────────────────────────────────────
# CHART FACTORIES
# ─────────────────────────────────────────────────────────────────────────────
def chart_discount_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: recommended discount per product (top 20)."""
    discount_col = next((c for c in df.columns if "recommended_discount" in c or "discount_pct" in c), None)
    if not discount_col:
        return go.Figure()

    top = df.nlargest(20, discount_col)[["product_id", discount_col, "category"]].reset_index(drop=True)

    color_vals = top[discount_col].values
    fig = go.Figure(go.Bar(
        x=top[discount_col],
        y=top["product_id"],
        orientation="h",
        marker=dict(
            color=color_vals,
            colorscale=AMBER_SCALE,
            showscale=True,
            colorbar=dict(title="Discount %", tickfont=dict(size=10), thickness=12),
            line=dict(color="rgba(245,158,11,0.2)", width=0.5),
        ),
        text=[f"{v:.1f}%" for v in color_vals],
        textposition="outside",
        textfont=dict(size=10, color="#8892a4"),
        hovertemplate="<b>%{y}</b><br>Discount: %{x:.1f}%<extra></extra>",
    ))
    apply_theme(fig,
        title="Recommended Discount — Top 20 Products",
        xaxis_title="Discount (%)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
        height=480,
    )
    return fig


def chart_price_vs_sentiment(df: pd.DataFrame) -> go.Figure:
    """Scatter: Price vs Sentiment, sized by discount, coloured by pop_score."""
    sentiment_col = next((c for c in df.columns if "sentiment" in c), None)
    discount_col = next((c for c in df.columns if "recommended_discount" in c or "discount_pct" in c), None)
    pop_col = next((c for c in df.columns if "pop" in c), None)

    if not sentiment_col:
        return go.Figure()

    plot_df = df.dropna(subset=["price", sentiment_col])
    sizes = plot_df[discount_col].clip(1, 60) * 1.2 if discount_col else 8
    colors = plot_df[pop_col] if pop_col else plot_df["price"]

    fig = go.Figure(go.Scatter(
        x=plot_df[sentiment_col],
        y=plot_df["price"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=AMBER_SCALE,
            showscale=True,
            colorbar=dict(title="Pop Score", tickfont=dict(size=10), thickness=12),
            opacity=0.75,
            line=dict(color="rgba(245,158,11,0.27)", width=0.5),
        ),
        text=plot_df["product_id"],
        customdata=np.stack([
            plot_df[discount_col] if discount_col else np.zeros(len(plot_df)),
            colors,
        ], axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sentiment: %{x:.3f}<br>"
            "Price: ₹%{y:,.0f}<br>"
            "Discount: %{customdata[0]:.1f}%<br>"
            "Pop Score: %{customdata[1]:.3f}"
            "<extra></extra>"
        ),
    ))
    apply_theme(fig,
        title="Price vs. Sentiment Analysis",
        xaxis_title="Sentiment Score",
        yaxis_title="Price (₹)",
        height=460,
    )
    return fig


def chart_units_by_season(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: units sold by season × category."""
    if "season" not in df.columns or "units_sold" not in df.columns:
        return go.Figure()

    cat_col = "category" if "category" in df.columns else None
    if cat_col:
        grouped = df.groupby(["season", cat_col])["units_sold"].sum().reset_index()
        fig = px.bar(
            grouped, x="season", y="units_sold", color=cat_col,
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"units_sold": "Units Sold", "season": "Season"},
            title="Units Sold by Season & Category",
        )
    else:
        grouped = df.groupby("season")["units_sold"].sum().reset_index()
        fig = px.bar(
            grouped, x="season", y="units_sold",
            color="season",
            color_discrete_sequence=["#f59e0b", "#34d399", "#60a5fa", "#f472b6"],
            labels={"units_sold": "Units Sold", "season": "Season"},
            title="Units Sold by Season",
        )

    apply_theme(fig,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def chart_festival_comparison(df: pd.DataFrame) -> go.Figure:
    """Side-by-side pie: revenue split festival vs non-festival."""
    if "festival" not in df.columns or "revenue" not in df.columns:
        return go.Figure()

    grp = df.groupby("festival")["revenue"].sum().reset_index()
    grp["label"] = grp["festival"].map({0: "Non-Festival", 1: "Festival 🎉"})

    fig = go.Figure(go.Pie(
        labels=grp["label"],
        values=grp["revenue"],
        hole=0.55,
        marker=dict(colors=["#1e3a5f", "#f59e0b"], line=dict(color="#0d0f14", width=2)),
        textfont=dict(family="DM Mono, monospace", size=11),
        hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    apply_theme(fig,
        title="Revenue — Festival vs Non-Festival",
        height=380,
        annotations=[dict(
            text="Revenue<br>Split",
            x=0.5, y=0.5,
            font=dict(size=13, color="#e8e2d4", family="Syne, sans-serif"),
            showarrow=False,
        )],
    )
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of numeric features."""
    num_df = df.select_dtypes(include=[np.number])
    keep = [c for c in num_df.columns if num_df[c].std() > 0][:14]
    corr = num_df[keep].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=AMBER_SCALE,
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="x: %{x}<br>y: %{y}<br>r = %{z:.3f}<extra></extra>",
    ))
    apply_theme(fig,
        title="Feature Correlation Matrix",
        height=500,
        xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def chart_feature_importance(model) -> go.Figure | None:
    """XGBoost feature importance chart."""
    try:
        importance = model.get_booster().get_score(importance_type="gain")
        if not importance:
            return None
        imp_df = pd.DataFrame({
            "feature": list(importance.keys()),
            "importance": list(importance.values()),
        }).sort_values("importance", ascending=True).tail(15)

        fig = go.Figure(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["importance"],
                colorscale=AMBER_SCALE,
                showscale=False,
                line=dict(color="rgba(245,158,11,0.27)", width=0.5),
            ),
            hovertemplate="<b>%{y}</b><br>Gain: %{x:.2f}<extra></extra>",
        ))
        apply_theme(fig,
            title="XGBoost Feature Importance (Gain)",
            xaxis_title="Importance (Gain)",
            height=420,
        )
        return fig
    except Exception:
        return None


def chart_revenue_simulation(df: pd.DataFrame) -> go.Figure:
    """Line chart simulating revenue at various discount levels."""
    price_col = "price"
    units_col = "units_sold"
    if price_col not in df.columns or units_col not in df.columns:
        return go.Figure()

    discount_levels = np.arange(0, 61, 5)
    avg_price = df[price_col].mean()
    avg_units = df[units_col].mean()

    # Naive demand elasticity model: demand ↑ as discount ↑
    elasticity = 1.8
    revenues = []
    for d in discount_levels:
        eff_price = avg_price * (1 - d / 100)
        demand_mult = 1 + elasticity * (d / 100)
        rev = eff_price * avg_units * demand_mult
        revenues.append(rev)

    base_rev = avg_price * avg_units
    pct_change = [(r - base_rev) / base_rev * 100 for r in revenues]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=discount_levels, y=revenues,
        name="Simulated Revenue",
        line=dict(color="#f59e0b", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(245,158,11,0.1)",
        hovertemplate="Discount: %{x}%<br>Revenue: ₹%{y:,.0f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=discount_levels, y=pct_change,
        name="% Change from Base",
        line=dict(color="#34d399", width=2, dash="dot"),
        hovertemplate="Discount: %{x}%<br>Δ Revenue: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)

    apply_theme(fig,
        title="Revenue Impact Simulation (Demand Elasticity Model)",
        xaxis_title="Discount Level (%)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Simulated Revenue (₹)", secondary_y=False)
    fig.update_yaxes(title_text="% Change", secondary_y=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(unified_df: pd.DataFrame, recs_df: pd.DataFrame):
    """Render sidebar filters and return filtered dataframes."""
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 20px 0; text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
                        background:linear-gradient(135deg,#f59e0b,#fcd34d);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">
                🎯 DTI
            </div>
            <div style="font-size:10px;color:#556070;letter-spacing:0.15em;
                        font-family:'DM Mono',monospace;margin-top:4px;">
                DISCOUNT OPTIMIZER
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        # ── Season ──────────────────────────────────────────────────────────
        seasons = ["All"]
        if "season" in unified_df.columns:
            seasons += sorted(unified_df["season"].dropna().unique().tolist())
        selected_season = st.selectbox("📅 Season", seasons)

        # ── Festival ────────────────────────────────────────────────────────
        festival_on = False
        if "festival" in unified_df.columns:
            festival_options = st.radio(
                "🎉 Festival Period",
                options=["All", "Festival Only", "Non-Festival Only"],
                horizontal=False,
            )
        else:
            festival_options = "All"

        # ── Category ────────────────────────────────────────────────────────
        cat_options = ["All"]
        if "category" in unified_df.columns:
            cat_options += sorted(unified_df["category"].dropna().unique().tolist())
        selected_cat = st.selectbox("🏷️ Category", cat_options)

        st.markdown("---")

        # ── Top N ────────────────────────────────────────────────────────────
        top_n = st.slider("🏆 Top N Recommendations", min_value=5, max_value=min(120, len(recs_df)), value=25, step=5)

        # ── Sentiment range ───────────────────────────────────────────────
        sentiment_col = next((c for c in unified_df.columns if "sentiment" in c), None)
        sentiment_range = (-1.0, 1.0)
        if sentiment_col:
            s_min = float(unified_df[sentiment_col].min())
            s_max = float(unified_df[sentiment_col].max())
            sentiment_range = st.slider(
                "💬 Sentiment Range",
                min_value=round(s_min, 2), max_value=round(s_max, 2),
                value=(round(s_min, 2), round(s_max, 2)),
                step=0.05,
            )

        # ── Price range ───────────────────────────────────────────────────
        if "price" in recs_df.columns:
            p_min, p_max = float(recs_df["price"].min()), float(recs_df["price"].max())
            price_range = st.slider(
                "💰 Price Range (₹)",
                min_value=int(p_min), max_value=int(p_max),
                value=(int(p_min), int(p_max)),
                step=50,
            )
        else:
            price_range = (0, 99999)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:10px;color:#556070;text-align:center;"
            "font-family:DM Mono,monospace;letter-spacing:0.08em;'>"
            "DTI v1.0 · XGBoost Engine<br>© 2025 Dynamic Discount Inc.</div>",
            unsafe_allow_html=True,
        )

    # ── Apply filters ──────────────────────────────────────────────────────
    filt_unified = unified_df.copy()
    filt_recs = recs_df.copy()

    if selected_season != "All" and "season" in filt_unified.columns:
        filt_unified = filt_unified[filt_unified["season"] == selected_season]
        if "season" in filt_recs.columns:
            filt_recs = filt_recs[filt_recs["season"] == selected_season]

    if festival_options == "Festival Only" and "festival" in filt_unified.columns:
        filt_unified = filt_unified[filt_unified["festival"] == 1]
        if "festival" in filt_recs.columns:
            filt_recs = filt_recs[filt_recs["festival"] == 1]
    elif festival_options == "Non-Festival Only" and "festival" in filt_unified.columns:
        filt_unified = filt_unified[filt_unified["festival"] == 0]
        if "festival" in filt_recs.columns:
            filt_recs = filt_recs[filt_recs["festival"] == 0]

    if selected_cat != "All" and "category" in filt_unified.columns:
        filt_unified = filt_unified[filt_unified["category"] == selected_cat]
        if "category" in filt_recs.columns:
            filt_recs = filt_recs[filt_recs["category"] == selected_cat]

    if sentiment_col and sentiment_col in filt_recs.columns:
        filt_recs = filt_recs[
            filt_recs[sentiment_col].between(sentiment_range[0], sentiment_range[1])
        ]

    if "price" in filt_recs.columns:
        filt_recs = filt_recs[filt_recs["price"].between(price_range[0], price_range[1])]

    filt_recs = filt_recs.head(top_n)

    return filt_unified, filt_recs


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
def render_kpis(filt_recs: pd.DataFrame, unified_df: pd.DataFrame):
    section("📊", "Key Performance Indicators", "LIVE")
    c1, c2, c3, c4, c5 = st.columns(5)

    sentiment_col = next((c for c in filt_recs.columns if "sentiment" in c), None)
    discount_col = next((c for c in filt_recs.columns if "recommended_discount" in c or "discount_pct" in c), None)

    with c1:
        kpi("Total Products", str(len(filt_recs)), "in current view")
    with c2:
        avg_sent = filt_recs[sentiment_col].mean() if sentiment_col else 0
        kpi("Avg Sentiment", f"{avg_sent:+.3f}", "VADER + TextBlob")
    with c3:
        avg_disc = filt_recs[discount_col].mean() if discount_col else 0
        kpi("Avg Discount", fmt_pct(avg_disc), "recommended")
    with c4:
        eff_col = next((c for c in filt_recs.columns if "effective_price" in c), None)
        avg_eff = filt_recs[eff_col].mean() if eff_col else filt_recs.get("price", pd.Series([0])).mean()
        kpi("Avg Eff. Price", fmt_currency(avg_eff), "post-discount")
    with c5:
        rev_col = next((c for c in filt_recs.columns if "revenue" in c), None)
        total_rev = filt_recs[rev_col].sum() if rev_col else 0
        kpi("Total Revenue", fmt_currency(total_rev), "simulated impact")


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def render_table(filt_recs: pd.DataFrame):
    section("📋", "Top Recommendations", "STYLED TABLE")

    search_col, dl_col = st.columns([3, 1])
    with search_col:
        search_term = st.text_input("🔍 Search by Product ID or Category", placeholder="e.g. P00042  or  Electronics")
    with dl_col:
        st.write("")
        st.write("")
        csv_bytes = filt_recs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download CSV",
            data=csv_bytes,
            file_name="dti_recommendations.csv",
            mime="text/csv",
        )

    # Apply search
    display_df = filt_recs.copy()
    if search_term.strip():
        mask = pd.Series([False] * len(display_df), index=display_df.index)
        for col in ["product_id", "category"]:
            if col in display_df.columns:
                mask |= display_df[col].astype(str).str.contains(search_term.strip(), case=False, na=False)
        display_df = display_df[mask]

    # Preferred column order
    preferred = [
        "product_id", "category", "price", "effective_price",
        "recommended_discount_pct", "discount_pct", "sentiment_score",
        "pop_score", "units_sold", "season", "festival",
    ]
    cols_present = [c for c in preferred if c in display_df.columns]
    extra_cols = [c for c in display_df.columns if c not in cols_present]
    display_df = display_df[cols_present + extra_cols]

    if display_df.empty:
        st.info("🔎 No products match your search or filter criteria.")
    else:
        st.dataframe(
            style_dataframe(display_df),
            use_container_width=True,
            height=420,
        )
        st.caption(f"Showing {len(display_df):,} products")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def render_main_charts(filt_recs: pd.DataFrame):
    section("📈", "Core Visualizations", "PLOTLY")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig1 = chart_discount_bar(filt_recs)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig2 = chart_price_vs_sentiment(filt_recs)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def render_advanced_analytics(filt_unified: pd.DataFrame, show_heatmap: bool = True):
    section("🔬", "Advanced Analytics", "SEASONAL + FESTIVAL")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Seasonal Trends",
        "🎉 Festival Analysis",
        "📉 Revenue Simulation",
        "🔗 Correlation Heatmap",
    ])

    with tab1:
        st.plotly_chart(chart_units_by_season(filt_unified), use_container_width=True, config={"displayModeBar": False})

    with tab2:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(chart_festival_comparison(filt_unified), use_container_width=True, config={"displayModeBar": False})
        with c2:
            if "festival" in filt_unified.columns and "units_sold" in filt_unified.columns:
                # Safe aggregation — festival column may have 0, 1, or both values
                _agg = filt_unified.groupby("festival").agg({
                    "units_sold": "mean",
                    "price": "mean",
                })
                # Map numeric index values to labels; keep only what exists
                _label_map = {0: "Non-Festival", 1: "Festival 🎉"}
                _agg.index = [_label_map.get(v, str(v)) for v in _agg.index]
                fest_stats = _agg  # index is now string labels, length = however many groups exist

                if not fest_stats.empty:
                    fig = go.Figure()
                    metrics = ["units_sold", "price"]
                    colors = ["#f59e0b", "#34d399"]
                    for metric, color in zip(metrics, colors):
                        if metric in fest_stats.columns:
                            fig.add_trace(go.Bar(
                                x=list(fest_stats.index),
                                y=fest_stats[metric].tolist(),
                                name=metric.replace("_", " ").title(),
                                marker_color=color,
                            ))
                    apply_theme(fig,
                        barmode="group",
                        title="Avg Units Sold & Price — Festival vs Non-Festival",
                        height=380,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.info("No festival data available for current filters.")

    with tab3:
        st.plotly_chart(chart_revenue_simulation(filt_unified), use_container_width=True, config={"displayModeBar": False})
        st.caption("📌 Model assumes demand elasticity of 1.8x — higher discounts drive proportional unit-volume gains.")

    with tab4:
        if show_heatmap:
            st.plotly_chart(chart_correlation_heatmap(filt_unified), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Enable correlation heatmap in the sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EXPLANATION
# ─────────────────────────────────────────────────────────────────────────────
def render_model_explanation(model, unified_df: pd.DataFrame):
    section("🤖", "Model Explanation", "XGBOOST")

    c1, c2 = st.columns([3, 2])

    with c1:
        if model is not None:
            fig = chart_feature_importance(model)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Feature importance not available for this model format.")
        else:
            # Show demo feature importance
            demo_features = {
                "sentiment_score": 4823.4,
                "price": 3910.2,
                "pop_score": 3401.1,
                "units_sold": 2987.3,
                "margin_pct": 2450.8,
                "rating": 1980.6,
                "festival": 1654.2,
                "discount_pct": 1423.9,
                "season_encoded": 987.4,
                "category_encoded": 834.1,
            }
            imp_df = pd.DataFrame({
                "feature": list(demo_features.keys()),
                "importance": list(demo_features.values()),
            }).sort_values("importance")

            fig = go.Figure(go.Bar(
                x=imp_df["importance"],
                y=imp_df["feature"],
                orientation="h",
                marker=dict(
                    color=imp_df["importance"],
                    colorscale=AMBER_SCALE,
                    showscale=False,
                    line=dict(color="rgba(245,158,11,0.27)", width=0.5),
                ),
                hovertemplate="<b>%{y}</b><br>Gain: %{x:.0f}<extra></extra>",
            ))
            apply_theme(fig,
                title="XGBoost Feature Importance (Demo — load model for real values)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("""
        <div style="background:#161b27;border:1px solid #f59e0b22;border-radius:10px;padding:20px;margin-top:10px;">
            <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
                        color:#f59e0b;margin-bottom:16px;">⚙️ Model Architecture</div>
            <div style="display:flex;flex-direction:column;gap:10px;">
        """, unsafe_allow_html=True)

        model_info = [
            ("Algorithm", "XGBoost Regressor"),
            ("Objective", "reg:squarederror"),
            ("Optimised For", "Revenue × Discount"),
            ("Features", "Sentiment, Price, Pop, Season…"),
            ("Preprocessing", "StandardScaler + OHE"),
            ("Sentiment", "VADER + TextBlob (avg)"),
            ("Selection", "Best of LR / RF / XGB"),
        ]
        rows = "".join([
            f"""<div style="display:flex;justify-content:space-between;padding:6px 0;
                border-bottom:1px solid #1e2535;font-size:11px;">
                <span style="color:#556070;font-family:'DM Mono',monospace;">{k}</span>
                <span style="color:#e8e2d4;font-weight:500;font-family:'DM Mono',monospace;">{v}</span>
            </div>"""
            for k, v in model_info
        ])
        st.markdown(rows + "</div></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner("Loading datasets…"):
        unified_df = load_unified_dataset()
        recs_df = load_recommendations()
        model, scaler = load_model_and_scaler()

    using_demo = False
    if unified_df is None or recs_df is None:
        using_demo = True
        unified_df, recs_df = generate_demo_data()

    # ── Sidebar ─────────────────────────────────────────────────────────────
    filt_unified, filt_recs = render_sidebar(unified_df, recs_df)

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:flex-end;justify-content:space-between;
                padding-bottom:20px;border-bottom:1px solid #f59e0b22;margin-bottom:28px;">
        <div>
            <p class="dashboard-title">🎯 DTI</p>
            <p class="dashboard-subtitle">DYNAMIC DISCOUNT OPTIMIZER · ML-POWERED REVENUE ENGINE</p>
        </div>
        <div style="text-align:right;font-family:'DM Mono',monospace;font-size:11px;color:#556070;">
            <div style="color:#34d399;margin-bottom:4px;">● LIVE</div>
            <div>XGBoost · VADER · TextBlob</div>
            <div>Multi-source Dataset Fusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if using_demo:
        st.warning(
            "⚠️ **Demo Mode** — `unified_dataset.csv` and `top_recommendations.csv` not found. "
            "Displaying synthetically generated data. Place your CSV files in the same directory to load real data.",
            icon="⚠️",
        )

    # ── KPIs ────────────────────────────────────────────────────────────────
    render_kpis(filt_recs, filt_unified)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Table ───────────────────────────────────────────────────────────────
    render_table(filt_recs)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Core charts ─────────────────────────────────────────────────────────
    render_main_charts(filt_recs)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Advanced analytics ──────────────────────────────────────────────────
    render_advanced_analytics(filt_unified)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model explanation ───────────────────────────────────────────────────
    render_model_explanation(model, filt_unified)

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:20px;border-top:1px solid #f59e0b15;
                font-family:'DM Mono',monospace;font-size:10px;color:#334155;
                letter-spacing:0.12em;">
        DTI — DYNAMIC DISCOUNT OPTIMIZER · BUILT WITH XGBOOST + STREAMLIT · © 2025
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
