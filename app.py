"""
Claim Frequency & Severity Modeling — Interactive Dashboard
French Motor MTPL Insurance
Run: streamlit run app.py
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTPL Pricing Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: #ffffff;
    border-left: 5px solid #d4a017;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 4px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
}
.kpi-card.green  { border-left-color: #27ae60; }
.kpi-card.red    { border-left-color: #c0392b; }
.kpi-card.blue   { border-left-color: #2980b9; }
.kpi-card.purple { border-left-color: #7d3c98; }
.kpi-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #d4a017; }
.kpi-value.green  { color: #27ae60; }
.kpi-value.red    { color: #c0392b; }
.kpi-value.blue   { color: #2980b9; }
.kpi-value.purple { color: #7d3c98; }
.kpi-sub { font-size: 0.78rem; color: #999; margin-top: 2px; }
.note-box {
    background: #eaf4fb;
    border-left: 4px solid #2980b9;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.92rem;
    color: #1a3a4a;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #2980b9;
    border-bottom: 2px solid #eaf4fb;
    padding-bottom: 4px;
    margin: 18px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
CLAIM  = os.path.join(BASE, "claim")
DATA   = os.path.join(BASE, "in")
FIG    = os.path.join(BASE, "fig")
OUT    = os.path.join(BASE, "out")
ASSETS = os.path.join(BASE, "assets")

def find_file(filename, folders):
    for folder in folders:
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            return p
    return None

def find_image(folders):
    """Find any image file (.webp, .jpg, .jpeg, .png) in the given folders."""
    exts = [".webp", ".jpg", ".jpeg", ".png"]
    for folder in folders:
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if any(f.lower().endswith(e) for e in exts):
                return os.path.join(folder, f)
    return None

# ─── LOADERS ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    p = find_file("freMTPL2_clean_for_model.csv", [DATA, CLAIM, BASE])
    if p is None:
        return None
    df = pd.read_csv(p)
    df["Exposure"]     = df["Exposure"].clip(upper=1)
    df["ClaimNb"]      = df["ClaimNb"].clip(upper=4)
    df["VehAgeS"]      = np.minimum(4, df["VehAge"])
    df["DrivAgeGroup"] = pd.cut(
        df["DrivAge"], bins=[17,24,34,49,64,110],
        labels=["18-24","25-34","35-49","50-64","65+"]
    )
    if "Frequency" not in df.columns:
        df["Frequency"] = df["ClaimNb"] / df["Exposure"].clip(lower=1e-6)
    return df

@st.cache_data(show_spinner=False)
def load_metrics():
    p = find_file("glm_metrics.json", [CLAIM, OUT, BASE])
    if p is None:
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_severity_metrics():
    p = find_file("severity_metrics.json", [CLAIM, OUT, BASE])
    if p is None:
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_pure_premium():
    p = find_file("pure_premium.csv", [DATA, CLAIM, BASE])
    if p is None:
        return None
    return pd.read_csv(p)

@st.cache_data(show_spinner=False)
def load_clusters():
    p = find_file("client_clusters.csv", [DATA, CLAIM, BASE])
    if p is None:
        return None
    return pd.read_csv(p)

@st.cache_resource(show_spinner=False)
def load_cann():
    try:
        import tensorflow as tf
        mp = find_file("model_cann_production.keras", [CLAIM, BASE])
        sp = find_file("scaler_nn.joblib",            [CLAIM, BASE])
        cp = find_file("cat_dims.json",               [CLAIM, BASE])
        if not all([mp, sp, cp]):
            return None, None, None
        m = tf.keras.models.load_model(mp, compile=False)
        s = joblib.load(sp)
        with open(cp) as f:
            c = json.load(f)
        return m, s, c
    except Exception:
        return None, None, None

# ─── HELPERS ──────────────────────────────────────────────────────────────────
C = dict(gold="#d4a017", green="#27ae60", red="#c0392b",
         blue="#2980b9", purple="#7d3c98", orange="#e67e22",
         bg="#ffffff", card="#f8f9fa", text="#212529")

LAYOUT = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#f8f9fa",
    font=dict(color="#212529"),
    xaxis=dict(gridcolor="#e9ecef"), yaxis=dict(gridcolor="#e9ecef"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)"),
    margin=dict(t=50, b=30, l=10, r=10),
)

def kpi(label, value, color="gold", sub=None):
    cls = "" if color == "gold" else color
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi-card {cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {cls}">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True
    )

def section(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

MODEL_COLORS = {
    "GLM 1 (Baseline)":      "#7f8c8d",
    "GLM 2 (Raw)":           "#c0392b",
    "GLM 3 (Engineered)":    "#2980b9",
    "GLM 4 (NegBin)":        "#7d3c98",
    "LightGBM Pure":         "#e67e22",
    "LightGBM Constrained":  "#f39c12",
    "XGBoost Pure":          "#1abc9c",
    "XGBoost Constrained":   "#16a085",
    "CANN (Deep Learning)":  "#27ae60",
}

SEV_COLORS = {
    "Gamma GLM":   "#2980b9",
    "LightGBM":    "#e67e22",
    "XGBoost":     "#1abc9c",
    "Neural Net":  "#27ae60",
}

VAR_INFO = {
    "ClaimNb":    ("integer",  "Number of claims filed (capped at 4)"),
    "Exposure":   ("float",    "Fraction of year the policy was active (capped at 1)"),
    "Area":       ("ordinal",  "Driver area: A = rural to F = dense urban"),
    "VehPower":   ("integer",  "Vehicle engine power in horsepower (capped at 9)"),
    "VehAge":     ("integer",  "Age of the vehicle in years"),
    "DrivAge":    ("integer",  "Age of the main driver"),
    "BonusMalus": ("integer",  "Claims experience: below 100 = bonus (safe driver), above 100 = malus (risky driver). Capped at 150."),
    "VehBrand":   ("category", "Vehicle manufacturer brand"),
    "VehGas":     ("binary",   "Fuel type: Diesel or Regular (petrol)"),
    "Density":    ("float",    "Population density of the driver municipality (log-transformed for modeling)"),
    "Region":     ("category", "French administrative region of the policyholder"),
    "Frequency":  ("float",    "TARGET (Frequency) — Annualized claim rate = ClaimNb / Exposure"),
    "ClaimAmount":("float",    "TARGET (Severity) — Individual claim cost in EUR (freMTPL2sev)"),
}

# ─── LOAD ALL DATA ─────────────────────────────────────────────────────────────
df        = load_data()
metrics   = load_metrics()
sev_m     = load_severity_metrics()
pp_df     = load_pure_premium()
clust_df  = load_clusters()
cann_model, scaler, cat_dims = load_cann()

# ─── IMAGE: find .webp first, then .jpg fallback ──────────────────────────────
img = find_image([BASE, ASSETS])

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    if img:
        st.image(img, use_container_width=True)
    st.markdown("## MTPL Pricing Dashboard")
    st.markdown("**French Motor MTPL**")
    st.markdown("Actuarial Data Science")
    st.divider()
    st.markdown("**Frequency models**")
    st.markdown("- GLM Poisson & Negative Binomial\n- LightGBM & XGBoost\n- CANN Deep Learning")
    st.divider()
    st.markdown("**Severity models**")
    st.markdown("- Log-Normal & Gamma adequacy\n- Gamma GLM\n- LightGBM & XGBoost\n- Neural Network\n- Pure Premium\n- Client Clustering")
    st.divider()
    if df is not None:
        st.success(f"Frequency data: {len(df):,} policies")
    else:
        st.error("Frequency data not found")
    if sev_m:
        st.success("Severity metrics loaded")
    if pp_df is not None:
        st.success(f"Pure premium: {len(pp_df):,} policies")
    if clust_df is not None:
        st.success(f"Clusters: {clust_df['Cluster'].nunique()} groups")
    st.divider()
    st.caption("Reference: Schelldorfer & Wuthrich (2019)")

# ─── HEADER ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 5])
with c1:
    if img:
        st.image(img, width=110)
with c2:
    st.title("Claim Frequency & Severity Modeling Dashboard")
    st.markdown("**French Motor Third Party Liability (MTPL) — Actuarial Data Science Pipeline**")
    st.markdown(
        "GLM Poisson/NegBin · LightGBM & XGBoost · CANN · "
        "Gamma GLM Severity · Pure Premium · Client Clustering"
    )

st.markdown("""
<div class="note-box">
<strong>Full pricing pipeline:</strong>
<strong>Pure Premium = Frequency × Severity</strong> — This dashboard covers both stages:
(1) claim frequency modeling (Poisson GLM, gradient boosting, CANN) and
(2) claim severity modeling (Gamma GLM, log-normal, ML, deep learning).
Client clustering groups policyholders into homogeneous risk segments.
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "Overview",
    "Univariate Analysis",
    "Bivariate Analysis",
    "Frequency Benchmark",
    "Severity Modeling",
    "Client Prediction",
    "Explainability & SHAP",
    "Client Clustering",
    "Upload Data",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with t1:
    st.markdown("## Project Overview")

    # ── Banner image ──────────────────────────────────────────────
    if img:
        col_img, col_txt = st.columns([2, 3])
        with col_img:
            st.image(img, use_container_width=True, caption="French Motor MTPL — Actuarial Pricing")
        with col_txt:
            st.markdown("""
### Project Goal
Model the **claim frequency** and **claim severity** for ~678,000 French motor insurance policies,
then combine them into a **pure premium** for actuarial pricing.

$$\\text{Pure Premium}_i = \\hat{\\lambda}_i \\times \\hat{\\mu}_i$$

Full pipeline: GLM Poisson/NegBin · LightGBM · XGBoost · CANN ·
Gamma GLM Severity · Pure Premium · K-Means Client Segmentation
            """)
    else:
        pass  # image already shown in header

    with st.expander("What is this project?", expanded=True):
        st.markdown("""
**Goal:** Model the annual claim frequency and severity for approximately 678,000 French motor insurance policies,
and combine them into a **pure premium** for actuarial pricing.

$$\\text{Pure Premium}_i = \\hat{\\lambda}_i \\times \\hat{\\mu}_i$$

where $\\hat{\\lambda}_i$ is the predicted claim frequency and $\\hat{\\mu}_i$ the predicted average claim cost.

#### Frequency modeling

We model **Y_i ~ Poisson(μ_i)** where **μ_i = Exposure_i × λ_i**.

| Family | Models | Key Strength |
|---|---|---|
| GLM | Poisson, Negative Binomial | Interpretable, regulatory compliant |
| Gradient Boosting | LightGBM, XGBoost (with and without monotone constraints) | Non-linear interactions |
| Deep Learning | CANN (Combined Actuarial Neural Network) | GLM + NN hybrid |

#### Severity modeling

We model the average claim cost **X_i ~ Gamma(α, β)** with a log-link GLM and gradient boosting.

| Step | Method |
|---|---|
| Distribution test | Log-Normal vs Gamma (AIC/BIC, KS test) |
| GLM | Gamma GLM with log-link (statsmodels) |
| ML | LightGBM & XGBoost (Tweedie, p=2) |
| DL | Feedforward NN with Gamma deviance loss |

#### Client clustering

K-Means segmentation on risk features + pure premium to identify homogeneous risk groups.
        """)

    st.markdown("### Variable Dictionary")
    TYPE_COLOR = {
        "integer":  "#2980b9", "float":   "#27ae60", "category": "#d4a017",
        "ordinal":  "#e67e22", "binary":  "#7d3c98",
    }
    for var, (vtype, desc) in VAR_INFO.items():
        ca, cb, cc = st.columns([1.2, 0.7, 4])
        with ca:
            st.markdown(f"**`{var}`**")
        with cb:
            color = TYPE_COLOR.get(vtype, "#888")
            st.markdown(
                f'<span style="background:{color};color:#fff;padding:2px 10px;'
                f'border-radius:12px;font-size:0.72rem;font-weight:600">{vtype}</span>',
                unsafe_allow_html=True
            )
        with cc:
            st.markdown(desc)
        st.markdown('<hr style="margin:3px 0;border-color:#eee">', unsafe_allow_html=True)

    if df is not None:
        st.markdown("### Portfolio KPIs")
        k1,k2,k3,k4,k5 = st.columns(5)
        total_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
        with k1: kpi("Policies",        f"{len(df):,}",                      "blue")
        with k2: kpi("Total Exposure",   f"{df['Exposure'].sum():,.0f} yrs",  "green")
        with k3: kpi("Total Claims",    f"{df['ClaimNb'].sum():,.0f}",        "gold")
        with k4: kpi("Claim Frequency", f"{total_freq:.4f}/yr",               "purple")
        with k5: kpi("Zero-Claim %",    f"{(df['ClaimNb']==0).mean():.1%}",  "red")
    else:
        st.warning("Dataset not loaded. Check that `freMTPL2_clean_for_model.csv` is in `in/` or `claim/`.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — UNIVARIATE
# ═══════════════════════════════════════════════════════════════════
with t2:
    st.markdown("## Univariate Analysis")
    if df is None:
        st.warning("Dataset not loaded.")
    else:
        NUM = ["DrivAge","VehAge","BonusMalus","Density","Exposure","ClaimNb","Frequency"]
        CAT = ["Area","VehBrand","VehGas","Region"]
        NUM = [c for c in NUM if c in df.columns]
        CAT = [c for c in CAT if c in df.columns]

        col_ctrl, col_plot = st.columns([1, 3])
        with col_ctrl:
            kind = st.radio("Variable type", ["Numerical","Categorical"])
            if kind == "Numerical":
                var = st.selectbox("Variable", NUM)
                show_log = st.checkbox("Log scale Y", value=False)
            else:
                var = st.selectbox("Variable", CAT)

        with col_plot:
            if kind == "Numerical":
                fig = px.histogram(df, x=var, nbins=60, marginal="box",
                                   color_discrete_sequence=[C["gold"]],
                                   title=f"Distribution of {var}")
                if show_log:
                    fig.update_yaxes(type="log")
                fig.update_layout(**LAYOUT, title_x=0.5, height=380)
                st.plotly_chart(fig, use_container_width=True)

                s = df[var].describe()
                r1,r2,r3,r4 = st.columns(4)
                with r1: kpi("Mean",   f"{s['mean']:.3f}", "blue")
                with r2: kpi("Median", f"{s['50%']:.3f}",  "green")
                with r3: kpi("Std",    f"{s['std']:.3f}",  "gold")
                with r4: kpi("Max",    f"{s['max']:.1f}",  "red")
            else:
                vc = df[var].value_counts().reset_index()
                vc.columns = [var,"Count"]
                fig1 = px.bar(vc, x=var, y="Count",
                              color_discrete_sequence=[C["blue"]],
                              title=f"Distribution of {var}")
                fig1.update_layout(**LAYOUT, title_x=0.5, height=280)
                st.plotly_chart(fig1, use_container_width=True)

                fc = (df.groupby(var, observed=True)
                      .agg(nb=("ClaimNb","sum"), ex=("Exposure","sum"))
                      .assign(freq=lambda x: x["nb"]/x["ex"])
                      .reset_index())
                fc[var] = fc[var].astype(str)
                fc = fc.sort_values("freq", ascending=False)
                fig2 = px.bar(fc, x=var, y="freq",
                              color="freq", color_continuous_scale="YlOrRd",
                              title=f"Annualized Claim Frequency by {var}")
                fig2.update_layout(**LAYOUT, title_x=0.5, height=280,
                                   coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — BIVARIATE
# ═══════════════════════════════════════════════════════════════════
with t3:
    st.markdown("## Bivariate Analysis")
    if df is None:
        st.warning("Dataset not loaded.")
    else:
        CAT_ALL = [c for c in ["Area","VehBrand","VehGas","Region","DrivAgeGroup","VehAgeS"] if c in df.columns]

        b1, b2 = st.columns(2)
        with b1:
            xv = st.selectbox("X axis (segmentation)", CAT_ALL, index=0)
        with b2:
            cv = st.selectbox("Color by", [c for c in CAT_ALL if c != xv], index=0)

        grp = list({xv, cv})
        biv = (df.groupby(grp, observed=True)
               .agg(nb=("ClaimNb","sum"), ex=("Exposure","sum"))
               .reset_index()
               .assign(freq=lambda x: x["nb"]/x["ex"]))
        biv = biv[biv["ex"] >= 50]
        biv[xv] = biv[xv].astype(str)
        biv[cv] = biv[cv].astype(str)

        fig_biv = px.scatter(
            biv, x=xv, y="freq", color=cv, size="ex", size_max=28,
            title=f"Claim Frequency: {xv} x {cv}",
            labels={"freq":"Annualized Freq","ex":"Exposure"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
            height=420,
        )
        fig_biv.update_layout(**LAYOUT, title_x=0.5)
        st.plotly_chart(fig_biv, use_container_width=True)

        st.divider()
        st.markdown("#### BonusMalus Score vs Claim Frequency")
        st.caption("The BonusMalus system is the strongest pricing signal. Monotone increase is enforced in constrained models.")

        bm_df = df.copy()
        bm_df["BM_bin"] = pd.cut(bm_df["BonusMalus"], bins=20)
        bm_g = (bm_df.groupby("BM_bin", observed=True)
                .agg(nb=("ClaimNb","sum"), ex=("Exposure","sum"))
                .reset_index()
                .assign(freq=lambda x: x["nb"]/x["ex"],
                        mid=lambda x: x["BM_bin"].apply(lambda b: b.mid)))

        fig_bm = go.Figure()
        fig_bm.add_trace(go.Scatter(
            x=bm_g["mid"], y=bm_g["freq"],
            mode="lines+markers",
            line=dict(color=C["gold"], width=2.5),
            marker=dict(size=6),
            fill="tozeroy", fillcolor="rgba(212,160,23,0.12)",
            name="Observed frequency"
        ))
        fig_bm.add_vline(x=100, line_dash="dash", line_color="#888",
                         annotation_text="BM=100 (no claim history)")
        fig_bm.update_layout(**LAYOUT, title="Claim Frequency by BonusMalus Score",
                             xaxis_title="BonusMalus", yaxis_title="Annualized Frequency",
                             title_x=0.5, height=350)
        st.plotly_chart(fig_bm, use_container_width=True)

        st.divider()
        st.markdown("#### Driver Age vs Claim Frequency")
        age_df = df.copy()
        age_df["Age_bin"] = pd.cut(age_df["DrivAge"], bins=15)
        age_g = (age_df.groupby("Age_bin", observed=True)
                 .agg(nb=("ClaimNb","sum"), ex=("Exposure","sum"))
                 .reset_index()
                 .assign(freq=lambda x: x["nb"]/x["ex"],
                         mid=lambda x: x["Age_bin"].apply(lambda b: b.mid)))
        fig_age = go.Figure()
        fig_age.add_trace(go.Scatter(
            x=age_g["mid"], y=age_g["freq"],
            mode="lines+markers",
            line=dict(color=C["blue"], width=2.5),
            fill="tozeroy", fillcolor="rgba(41,128,185,0.12)",
        ))
        fig_age.update_layout(**LAYOUT, title="Claim Frequency by Driver Age",
                              xaxis_title="Driver Age", yaxis_title="Annualized Frequency",
                              title_x=0.5, height=320)
        st.plotly_chart(fig_age, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — FREQUENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════
with t4:
    st.markdown("## Frequency Model Benchmark")

    if not metrics:
        st.warning("No `glm_metrics.json` found. Run notebooks 01–04 to generate metrics.")
    else:
        MAP = [
            ("GLM 1 (Baseline)",     "dev_glm1_int", None),
            ("GLM 2 (Raw)",          "dev_glm2_raw", "gini_glm2"),
            ("GLM 3 (Engineered)",   "dev_glm3_eng", "gini_glm3"),
            ("GLM 4 (NegBin)",       "dev_glm4_nb",  "gini_glm4"),
            ("LightGBM Pure",        "dev_lgb_pure",  "gini_lgb_pure"),
            ("LightGBM Constrained", "dev_lgb_const", "gini_lgb_const"),
            ("XGBoost Pure",         "dev_xgb_pure",  "gini_xgb_pure"),
            ("XGBoost Constrained",  "dev_xgb_const", "gini_xgb_const"),
            ("CANN (Deep Learning)", "dev_cann",       None),
        ]
        d1 = metrics.get("dev_glm1_int", 0)
        d3 = metrics.get("dev_glm3_eng", 1)
        rows = []
        for name, dk, gk in MAP:
            dev = metrics.get(dk)
            if dev is None: continue
            idx = (dev - d1) / (d3 - d1) * 100 if d3 != d1 else 0
            rows.append({"Model": name, "Deviance": round(dev,4),
                         "Gini": round(metrics[gk],4) if gk and gk in metrics else None,
                         "Index %": round(idx,1),
                         "Color": MODEL_COLORS.get(name,"#888")})
        bdf = pd.DataFrame(rows)

        best = bdf.loc[bdf["Index %"].idxmax()]
        m1,m2,m3,m4 = st.columns(4)
        with m1: kpi("Best Model",        best["Model"],             "green")
        with m2: kpi("Improvement Index", f"{best['Index %']:.1f}%", "green")
        with m3: kpi("GLM 3 Deviance",    f"{d3:.4f}",               "blue",  "Reference (100%)")
        with m4: kpi("Null Deviance",     f"{d1:.4f}",               "red",   "Baseline (0%)")

        st.divider()
        l, r = st.columns(2)
        with l:
            fig_imp = go.Figure(go.Bar(
                x=bdf["Index %"], y=bdf["Model"],
                orientation="h",
                marker=dict(color=bdf["Color"], line=dict(width=0)),
                text=[f"{v:.1f}%" for v in bdf["Index %"]],
                textposition="outside",
            ))
            fig_imp.add_vline(x=100, line_dash="dash", line_color=C["blue"],
                              annotation_text="GLM 3 = 100%",
                              annotation_font_color=C["blue"])
            fig_imp.update_layout(**LAYOUT, title="Actuarial Improvement Index",
                                  xaxis_title="Improvement Index (%)",
                                  height=400, title_x=0.5)
            st.plotly_chart(fig_imp, use_container_width=True)

        with r:
            gdf = bdf.dropna(subset=["Gini"])
            if not gdf.empty:
                fig_gini = go.Figure(go.Bar(
                    x=gdf["Gini"], y=gdf["Model"],
                    orientation="h",
                    marker=dict(color=gdf["Color"]),
                    text=[f"{v:.4f}" for v in gdf["Gini"]],
                    textposition="outside",
                ))
                fig_gini.update_layout(**LAYOUT, title="Gini Index by Model",
                                       xaxis_title="Gini Index",
                                       height=400, title_x=0.5)
                st.plotly_chart(fig_gini, use_container_width=True)
            else:
                st.info("Run notebooks to compute Gini Index.")

        st.divider()
        st.markdown("### Full Results Table")
        display_bdf = bdf.drop(columns=["Color"])
        st.dataframe(
            display_bdf.style
            .background_gradient(subset=["Index %"], cmap="YlGn")
            .format({"Deviance":"{:.4f}", "Gini":"{:.4f}", "Index %":"{:.1f}"}),
            use_container_width=True, height=320,
        )

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — SEVERITY MODELING (Notebook 05)
# ═══════════════════════════════════════════════════════════════════
with t5:
    st.markdown("## Claim Severity Modeling")
    st.markdown(
        "Second stage of the pricing pipeline: modeling the **average cost per claim**. "
        "Source: notebook `05_Severity_Modeling_CORRIGE.ipynb`."
    )

    # ── §3.2.1 & §3.2.2 — Distribution adequacy ──────────────────
    section("Distribution Adequacy — Log-Normal vs Gamma (§3.2.1 & §3.2.2)")

    if sev_m:
        d1a, d1b, d1c, d1d = st.columns(4)
        with d1a: kpi("AIC — Log-Normal", f"{sev_m.get('aic_lognormal',0):,.0f}", "red",   "lower = better")
        with d1b: kpi("AIC — Gamma",      f"{sev_m.get('aic_gamma',0):,.0f}",     "green", "lower = better")
        with d1c: kpi("BIC — Log-Normal", f"{sev_m.get('bic_lognormal',0):,.0f}", "red")
        with d1d: kpi("BIC — Gamma",      f"{sev_m.get('bic_gamma',0):,.0f}",     "green")

        st.markdown("")
        d2a, d2b, d2c = st.columns(3)
        with d2a: kpi("Log-Normal mu (log-scale)",    f"{sev_m.get('mu_lognormal',0):.4f}",  "blue")
        with d2b: kpi("Log-Normal sigma (log-scale)",  f"{sev_m.get('sigma_lognormal',0):.4f}", "blue")
        with d2c: kpi("Gamma shape alpha",             f"{sev_m.get('alpha_gamma',0):.4f}",   "purple")

        # AIC comparison chart
        aic_data = pd.DataFrame({
            "Distribution": ["Log-Normal", "Gamma"],
            "AIC":          [sev_m.get("aic_lognormal",0), sev_m.get("aic_gamma",0)],
            "Color":        [C["red"], C["green"]],
        })
        l_dist, r_dist = st.columns(2)
        with l_dist:
            fig_aic = go.Figure(go.Bar(
                x=aic_data["Distribution"], y=aic_data["AIC"],
                marker=dict(color=aic_data["Color"], opacity=0.85),
                text=[f"{v:,.0f}" for v in aic_data["AIC"]],
                textposition="outside",
            ))
            fig_aic.update_layout(**LAYOUT, title="AIC — Log-Normal vs Gamma",
                                  yaxis_title="AIC (lower = better)", height=320, title_x=0.5)
            st.plotly_chart(fig_aic, use_container_width=True)
        with r_dist:
            ks_stat = sev_m.get("ks_stat_gamma", None)
            ks_p    = sev_m.get("ks_p_gamma", None)
            if ks_stat is not None:
                kpi("KS Statistic (Gamma)", f"{ks_stat:.4f}", "blue", "Kolmogorov-Smirnov")
                kpi("KS p-value",           f"{ks_p:.2e}",    "green" if ks_p > 0.05 else "red",
                    "Not rejected (p>0.05)" if ks_p and ks_p > 0.05 else "Rejected (p<0.05)")
                kpi("Dispersion phi",       f"{sev_m.get('dispersion_glm',0):.4f}", "purple",
                    "CoV = sqrt(phi)")
            best_dist = "Gamma" if sev_m.get("aic_gamma",1) < sev_m.get("aic_lognormal",0) else "Log-Normal"
            st.success(f"Best fit: **{best_dist}** (lowest AIC)")
    else:
        st.info("Run `05_Severity_Modeling_CORRIGE.ipynb` to load distribution metrics.")

    # Saved figures from notebook 5
    sev_figs = {
        "Log-Normal Adequacy":       "severity_lognormal_adequacy.png",
        "Gamma Adequacy":            "severity_gamma_adequacy.png",
        "GLM Tariff Relativities":   "severity_glm_relativities.png",
        "GLM Validation":            "severity_glm_validation.png",
    }
    sev_found = {k: os.path.join(FIG, v) for k,v in sev_figs.items()
                 if os.path.exists(os.path.join(FIG,v))}
    if sev_found:
        sel_dist = st.selectbox("View diagnostic figure", list(sev_found.keys()), key="sev_fig")
        st.image(sev_found[sel_dist], use_container_width=True)

    st.divider()

    # ── §3.2.3 — Gamma GLM + ML + DL benchmark ───────────────────
    section("Severity Model Benchmark (§3.2.3)")

    if sev_m:
        SEV_MAP = [
            ("Gamma GLM",  "dev_sev_glm", "gini_sev_glm"),
            ("LightGBM",   "dev_sev_lgb", "gini_sev_lgb"),
            ("XGBoost",    "dev_sev_xgb", "gini_sev_xgb"),
            ("Neural Net", "dev_sev_nn",  "gini_sev_nn"),
        ]
        sev_rows = []
        for name, dk, gk in SEV_MAP:
            dev  = sev_m.get(dk)
            gini = sev_m.get(gk)
            if dev is None: continue
            sev_rows.append({
                "Model": name,
                "Gamma Deviance": round(dev,6),
                "Gini Index":     round(gini,4) if gini else None,
                "Color":          SEV_COLORS.get(name,"#888"),
            })
        sdf = pd.DataFrame(sev_rows)

        if not sdf.empty:
            best_sev = sdf.loc[sdf["Gamma Deviance"].idxmin()]
            s1,s2,s3,s4 = st.columns(4)
            with s1: kpi("Best Severity Model",  best_sev["Model"],                  "green")
            with s2: kpi("Best Deviance",        f"{best_sev['Gamma Deviance']:.5f}", "green", "lower = better")
            with s3: kpi("GLM Gamma Deviance",   f"{sev_m.get('dev_sev_glm',0):.5f}", "blue",  "baseline")
            with s4: kpi("GLM Gini Index",       f"{sev_m.get('gini_sev_glm',0):.4f}", "purple")

            sl, sr = st.columns(2)
            with sl:
                fig_sdev = go.Figure(go.Bar(
                    x=sdf["Gamma Deviance"], y=sdf["Model"],
                    orientation="h",
                    marker=dict(color=sdf["Color"], opacity=0.85),
                    text=[f"{v:.5f}" for v in sdf["Gamma Deviance"]],
                    textposition="outside",
                ))
                fig_sdev.update_layout(**LAYOUT, title="Gamma Deviance (lower = better)",
                                       xaxis_title="Gamma Deviance",
                                       height=300, title_x=0.5)
                st.plotly_chart(fig_sdev, use_container_width=True)

            with sr:
                gdf_s = sdf.dropna(subset=["Gini Index"])
                if not gdf_s.empty:
                    fig_sgini = go.Figure(go.Bar(
                        x=gdf_s["Gini Index"], y=gdf_s["Model"],
                        orientation="h",
                        marker=dict(color=gdf_s["Color"], opacity=0.85),
                        text=[f"{v:.4f}" for v in gdf_s["Gini Index"]],
                        textposition="outside",
                    ))
                    fig_sgini.update_layout(**LAYOUT, title="Gini Index (higher = better)",
                                            xaxis_title="Gini Index",
                                            height=300, title_x=0.5)
                    st.plotly_chart(fig_sgini, use_container_width=True)

            st.markdown("### Full Severity Results")
            disp_sdf = sdf.drop(columns=["Color"])
            st.dataframe(
                disp_sdf.style
                .background_gradient(subset=["Gamma Deviance"], cmap="RdYlGn_r")
                .format({"Gamma Deviance":"{:.6f}", "Gini Index":"{:.4f}"}),
                use_container_width=True,
            )
    else:
        st.info("Run `05_Severity_Modeling_CORRIGE.ipynb` to load severity benchmark.")

    # Neural network learning curve
    nn_fig_path = os.path.join(FIG, "severity_nn_learning_curve.png")
    bench_fig   = os.path.join(FIG, "severity_benchmark.png")
    if os.path.exists(bench_fig) or os.path.exists(nn_fig_path):
        st.divider()
        extra_figs = {}
        if os.path.exists(bench_fig):   extra_figs["Severity Benchmark Chart"]       = bench_fig
        if os.path.exists(nn_fig_path): extra_figs["Neural Network Learning Curve"] = nn_fig_path
        sel_extra = st.selectbox("Additional figures", list(extra_figs.keys()), key="sev_extra")
        st.image(extra_figs[sel_extra], use_container_width=True)

    st.divider()

    # ── Pure Premium ───────────────────────────────────────────────
    section("Pure Premium = Frequency × Severity")

    pp_fig = os.path.join(FIG, "pure_premium.png")
    if os.path.exists(pp_fig):
        st.image(pp_fig, use_container_width=True)

    if pp_df is not None:
        pp_cols = [c for c in ["PP_GLM","PP_LGB","PP_XGB"] if c in pp_df.columns]
        if pp_cols:
            p1,p2,p3 = st.columns(3)
            cols_kpi = [p1,p2,p3]
            col_colors = ["blue","gold","green"]
            for i, col in enumerate(pp_cols[:3]):
                with cols_kpi[i]:
                    kpi(f"Mean {col}", f"{pp_df[col].mean():.2f} EUR/yr", col_colors[i])

            st.markdown("")
            cap = pp_df[pp_cols[0]].quantile(0.99) if pp_cols else 1000
            fig_pp = go.Figure()
            pp_clrs = [C["blue"], C["gold"], C["green"]]
            for col, clr in zip(pp_cols, pp_clrs):
                fig_pp.add_trace(go.Histogram(
                    x=pp_df[col][pp_df[col] <= cap],
                    nbinsx=80, opacity=0.55,
                    marker_color=clr, name=col,
                ))
            fig_pp.update_layout(**LAYOUT,
                                 title="Pure Premium Distribution (capped at 99th pct)",
                                 xaxis_title="Pure Premium (EUR/yr)",
                                 barmode="overlay", height=350, title_x=0.5)
            st.plotly_chart(fig_pp, use_container_width=True)
    else:
        st.info("Run notebook 05 to generate `pure_premium.csv`.")

    st.divider()

    # ── Client Clustering ──────────────────────────────────────────
    section("Client Risk Segmentation — K-Means Clustering")

    clust_fig     = os.path.join(FIG, "clustering_results.png")
    elbow_fig     = os.path.join(FIG, "clustering_elbow.png")

    if os.path.exists(elbow_fig) or os.path.exists(clust_fig):
        cl1, cl2 = st.columns(2)
        with cl1:
            if os.path.exists(elbow_fig):
                st.image(elbow_fig, caption="Elbow method — choosing K", use_container_width=True)
        with cl2:
            if os.path.exists(clust_fig):
                st.image(clust_fig, caption="Cluster profiles & PCA projection", use_container_width=True)

    if clust_df is not None and pp_df is not None:
        merged = clust_df.merge(pp_df, on="IDpol", how="inner")
        if "Cluster" in merged.columns:
            merged["Cluster"] = merged["Cluster"].astype(str)
            pp_col_use = next((c for c in ["PP_GLM","PP_LGB"] if c in merged.columns), None)
            if pp_col_use:
                profile = merged.groupby("Cluster")[pp_col_use].agg(["mean","count"]).reset_index()
                profile.columns = ["Cluster","Mean Pure Premium","Policies"]
                profile = profile.sort_values("Mean Pure Premium")

                fig_cl = go.Figure(go.Bar(
                    x=profile["Cluster"],
                    y=profile["Mean Pure Premium"],
                    marker_color=[C["green"],C["gold"],C["orange"],C["red"],C["purple"]][:len(profile)],
                    text=[f"{v:.0f} EUR" for v in profile["Mean Pure Premium"]],
                    textposition="outside",
                ))
                fig_cl.update_layout(**LAYOUT,
                                     title="Mean Pure Premium by Risk Cluster",
                                     xaxis_title="Cluster ID",
                                     yaxis_title="Mean Pure Premium (EUR/yr)",
                                     height=320, title_x=0.5)
                st.plotly_chart(fig_cl, use_container_width=True)

                st.markdown("**Cluster Profile Summary**")
                st.dataframe(profile, use_container_width=True)
    elif clust_df is None:
        st.info("Run notebook 05 to generate cluster assignments (`client_clusters.csv`).")

# ═══════════════════════════════════════════════════════════════════
# TAB 6 — CLIENT PREDICTION
# ═══════════════════════════════════════════════════════════════════
with t6:
    st.markdown("## Individual Client Prediction")

    if df is None:
        st.warning("Dataset not loaded.")
    else:
        mode = st.radio("Mode", ["Pick from dataset", "Enter manually"], horizontal=True)
        st.divider()

        if mode == "Pick from dataset":
            sample = df["IDpol"].sample(min(500, len(df)), random_state=42).sort_values()
            chosen = st.selectbox("Select Policy ID", sample)
            row = df[df["IDpol"] == chosen].iloc[0]
        else:
            c1,c2,c3 = st.columns(3)
            with c1:
                area     = st.selectbox("Area", sorted(df["Area"].unique()))
                veh_pwr  = st.slider("Vehicle Power", 1, 15, 6)
                veh_age  = st.slider("Vehicle Age (yrs)", 0, 20, 3)
            with c2:
                driv_age = st.slider("Driver Age", 18, 100, 35)
                bonus    = st.slider("BonusMalus", 50, 230, 100)
                density  = st.number_input("Density (inhab/km2)", 1, 50000, 1500)
            with c3:
                brand    = st.selectbox("Vehicle Brand", sorted(df["VehBrand"].unique()))
                gas      = st.selectbox("Fuel Type",     sorted(df["VehGas"].unique()))
                region   = st.selectbox("Region",        sorted(df["Region"].astype(str).unique()))
                exposure = st.slider("Exposure (yrs)", 0.01, 1.0, 1.0)
            row = pd.Series({
                "IDpol":0,"Area":area,"VehPower":veh_pwr,"VehAge":veh_age,
                "DrivAge":driv_age,"BonusMalus":bonus,"Density":density,
                "VehBrand":brand,"VehGas":gas,"Region":region,
                "Exposure":exposure,"ClaimNb":0,
            })

        disp_cols = ["Area","VehPower","VehAge","DrivAge","BonusMalus","Density","VehBrand","VehGas","Region","Exposure"]
        disp_cols = [c for c in disp_cols if c in row.index]
        st.markdown("### Policy Profile")
        st.dataframe(pd.DataFrame([{c: row[c] for c in disp_cols}]), use_container_width=True)

        bm  = float(row.get("BonusMalus", 100))
        age = float(row.get("DrivAge", 35))
        risk = min(100, (bm-50)/180*70 + max(0,25-age)/25*30)

        g1, g2 = st.columns([1,2])
        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(risk,1),
                title={"text":"Risk Score","font":{"color":"#212529","size":14}},
                number={"suffix":"/100","font":{"color":C["gold"],"size":28}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#212529"},
                    "bar":{"color":C["gold"],"thickness":0.25},
                    "steps":[
                        {"range":[0,33],"color":"#d5f5e3"},
                        {"range":[33,66],"color":"#fef9e7"},
                        {"range":[66,100],"color":"#fadbd8"},
                    ],
                    "threshold":{"line":{"color":"#333","width":3},"value":risk},
                },
            ))
            fig_g.update_layout(paper_bgcolor="#f8f9fa", font_color="#212529",
                                height=270, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            global_freq = df["ClaimNb"].sum() / df["Exposure"].sum()
            st.markdown("### Predictions")
            p1,p2 = st.columns(2)
            with p1:
                kpi("Portfolio Frequency", f"{global_freq:.4f}/yr", "blue", "Average all policies")
            with p2:
                if "fit_glm3" in row.index and pd.notna(row.get("fit_glm3",None)):
                    glm3_f = float(row["fit_glm3"]) / max(float(row.get("Exposure",1)),1e-6)
                    kpi("GLM 3 Frequency", f"{glm3_f:.4f}/yr", "green",
                        f"Predicted claims: {float(row['fit_glm3']):.4f}")
                    ratio = glm3_f / global_freq
                    if   ratio > 1.2: st.error(f"{ratio:.1f}x above average — HIGH RISK")
                    elif ratio < 0.8: st.success(f"{ratio:.1f}x below average — LOW RISK")
                    else:             st.info(f"Close to average (ratio: {ratio:.2f})")
                else:
                    kpi("GLM 3 Frequency", "N/A", "gold", "Run Notebook 2 first")

            # Pure premium for this policy
            if pp_df is not None and "IDpol" in row.index and row["IDpol"] != 0:
                policy_pp = pp_df[pp_df["IDpol"] == row["IDpol"]]
                if not policy_pp.empty and "PP_GLM" in policy_pp.columns:
                    st.markdown("---")
                    kpi("Pure Premium (GLM)",
                        f"{policy_pp['PP_GLM'].values[0]:.2f} EUR/yr",
                        "purple", "Frequency x Severity")

            st.markdown("---")
            bm_pct = (df["BonusMalus"] <= bm).mean()
            st.markdown(f"This policy's BonusMalus **{bm:.0f}** is higher than **{bm_pct:.0%}** of the portfolio.")
            fig_bm_ctx = go.Figure()
            fig_bm_ctx.add_trace(go.Histogram(
                x=df["BonusMalus"], nbinsx=50,
                marker_color=C["blue"], opacity=0.7, name="Portfolio"
            ))
            fig_bm_ctx.add_vline(x=bm, line_color=C["gold"], line_width=2.5,
                                  annotation_text=f"This policy: {bm:.0f}",
                                  annotation_font_color=C["gold"])
            fig_bm_ctx.update_layout(**LAYOUT, height=200,
                                     xaxis_title="BonusMalus",
                                     showlegend=False)
            st.plotly_chart(fig_bm_ctx, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 7 — EXPLAINABILITY & SHAP
# ═══════════════════════════════════════════════════════════════════
with t7:
    st.markdown("## Explainability & SHAP")
    st.markdown(
        "Model interpretability for both frequency and severity models: "
        "feature importance, SHAP values, tariff relativities, and observed vs predicted."
    )

    ex1, ex2, ex3 = st.tabs([
        "Model Comparison",
        "SHAP & Feature Importance",
        "Observed vs Predicted",
    ])

    # ── Sub-tab A : Model Comparison ──────────────────────────────
    with ex1:
        section("Frequency Models — Full Benchmark")

        if metrics:
            ALL_FREQ = [
                ("GLM 1 — Null",          "dev_glm1_int", None,           "#7f8c8d"),
                ("GLM 2 — Raw",           "dev_glm2_raw", "gini_glm2",    "#c0392b"),
                ("GLM 3 — Engineered",    "dev_glm3_eng", "gini_glm3",    "#2980b9"),
                ("GLM 4 — NegBin",        "dev_glm4_nb",  "gini_glm4",    "#7d3c98"),
                ("LightGBM Pure",         "dev_lgb_pure", "gini_lgb_pure","#e67e22"),
                ("LightGBM Constrained",  "dev_lgb_const","gini_lgb_const","#f39c12"),
                ("XGBoost Pure",          "dev_xgb_pure", "gini_xgb_pure","#1abc9c"),
                ("XGBoost Constrained",   "dev_xgb_const","gini_xgb_const","#16a085"),
                ("CANN",                  "dev_cann",     None,            "#27ae60"),
            ]
            d1 = metrics.get("dev_glm1_int", 0)
            d3 = metrics.get("dev_glm3_eng", 1)
            freq_rows = []
            for name, dk, gk, col in ALL_FREQ:
                dev = metrics.get(dk)
                if dev is None: continue
                idx = (dev - d1) / (d3 - d1) * 100 if d3 != d1 else 0
                freq_rows.append({
                    "Model": name, "Poisson Deviance": round(dev,5),
                    "Gini":  round(metrics[gk],4) if gk and gk in metrics else None,
                    "Improvement Index %": round(idx,1), "Color": col,
                })
            fdf = pd.DataFrame(freq_rows)

            # Grouped bar: deviance + improvement side by side
            fa, fb = st.columns(2)
            with fa:
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(
                    y=fdf["Model"], x=fdf["Poisson Deviance"],
                    orientation="h", name="Poisson Deviance",
                    marker_color=fdf["Color"], opacity=0.85,
                    text=[f"{v:.5f}" for v in fdf["Poisson Deviance"]],
                    textposition="outside",
                ))
                fig_cmp.update_layout(**LAYOUT, height=420,
                                      title="Poisson Deviance (lower = better)",
                                      xaxis_title="Deviance", title_x=0.5)
                st.plotly_chart(fig_cmp, use_container_width=True)
            with fb:
                fig_idx = go.Figure()
                fig_idx.add_trace(go.Bar(
                    y=fdf["Model"], x=fdf["Improvement Index %"],
                    orientation="h", name="Improvement Index",
                    marker_color=fdf["Color"], opacity=0.85,
                    text=[f"{v:.1f}%" for v in fdf["Improvement Index %"]],
                    textposition="outside",
                ))
                fig_idx.add_vline(x=100, line_dash="dash", line_color=C["blue"],
                                  annotation_text="GLM 3 = 100%",
                                  annotation_font_color=C["blue"])
                fig_idx.update_layout(**LAYOUT, height=420,
                                      title="Actuarial Improvement Index",
                                      xaxis_title="Index (%)", title_x=0.5)
                st.plotly_chart(fig_idx, use_container_width=True)

            # Gini radar-like bar
            gdf_f = fdf.dropna(subset=["Gini"])
            if not gdf_f.empty:
                fig_gin = go.Figure(go.Bar(
                    x=gdf_f["Model"], y=gdf_f["Gini"],
                    marker_color=gdf_f["Color"], opacity=0.85,
                    text=[f"{v:.4f}" for v in gdf_f["Gini"]],
                    textposition="outside",
                ))
                fig_gin.update_layout(**LAYOUT, title="Gini Index by Model (higher = better)",
                                      yaxis_title="Gini Index", height=320, title_x=0.5)
                st.plotly_chart(fig_gin, use_container_width=True)

            st.markdown("**Frequency Results Table**")
            disp_f = fdf.drop(columns=["Color"])
            st.dataframe(
                disp_f.style
                .background_gradient(subset=["Improvement Index %"], cmap="YlGn")
                .format({"Poisson Deviance":"{:.5f}", "Gini":"{:.4f}",
                         "Improvement Index %":"{:.1f}"}),
                use_container_width=True,
            )
        else:
            st.info("Run notebooks 01–04 to generate frequency metrics.")

        st.divider()
        section("Severity Models — Gamma Deviance vs Gini")

        if sev_m:
            SEV_BENCH = [
                ("Gamma GLM",  "dev_sev_glm","gini_sev_glm","#2980b9"),
                ("LightGBM",   "dev_sev_lgb","gini_sev_lgb","#e67e22"),
                ("XGBoost",    "dev_sev_xgb","gini_sev_xgb","#1abc9c"),
                ("Neural Net", "dev_sev_nn", "gini_sev_nn", "#27ae60"),
            ]
            sev_bench = []
            for name, dk, gk, col in SEV_BENCH:
                dev  = sev_m.get(dk)
                gini = sev_m.get(gk)
                if dev is None: continue
                sev_bench.append({"Model":name,"Gamma Deviance":round(dev,6),
                                   "Gini Index": round(gini,4) if gini else None,
                                   "Color":col})
            sbdf = pd.DataFrame(sev_bench)
            if not sbdf.empty:
                sa, sb = st.columns(2)
                with sa:
                    fig_sd = go.Figure(go.Bar(
                        x=sbdf["Model"], y=sbdf["Gamma Deviance"],
                        marker_color=sbdf["Color"], opacity=0.85,
                        text=[f"{v:.5f}" for v in sbdf["Gamma Deviance"]],
                        textposition="outside",
                    ))
                    fig_sd.update_layout(**LAYOUT, title="Gamma Deviance — Severity",
                                         yaxis_title="Deviance", height=320, title_x=0.5)
                    st.plotly_chart(fig_sd, use_container_width=True)
                with sb:
                    gdf_sv = sbdf.dropna(subset=["Gini Index"])
                    if not gdf_sv.empty:
                        fig_sg = go.Figure(go.Bar(
                            x=gdf_sv["Model"], y=gdf_sv["Gini Index"],
                            marker_color=gdf_sv["Color"], opacity=0.85,
                            text=[f"{v:.4f}" for v in gdf_sv["Gini Index"]],
                            textposition="outside",
                        ))
                        fig_sg.update_layout(**LAYOUT, title="Gini Index — Severity",
                                             yaxis_title="Gini Index", height=320, title_x=0.5)
                        st.plotly_chart(fig_sg, use_container_width=True)
                disp_sv = sbdf.drop(columns=["Color"])
                st.dataframe(
                    disp_sv.style
                    .background_gradient(subset=["Gamma Deviance"], cmap="RdYlGn_r")
                    .format({"Gamma Deviance":"{:.6f}","Gini Index":"{:.4f}"}),
                    use_container_width=True,
                )
        else:
            st.info("Run notebook 05 to generate severity metrics.")

    # ── Sub-tab B : SHAP & Feature Importance ─────────────────────
    with ex2:
        section("SHAP Values — Frequency Models")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) quantifies the marginal contribution "
            "of each feature to the model prediction for each policy."
        )

        SHAP_FIGS = {
            "SHAP — LightGBM Pure (Frequency)":        "shap_lgbm_comparison.png",
            "SHAP — CANN (Frequency)":                 "cann_shap_summary.png",
            "Feature Importance — CANN":               "cann_feature_importance.png",
            "Feature Importance — Severity (ML)":      "severity_ml_importance.png",
        }
        shap_found = {k: os.path.join(FIG, v) for k,v in SHAP_FIGS.items()
                      if os.path.exists(os.path.join(FIG,v))}

        if shap_found:
            sel_shap = st.selectbox("Select SHAP figure", list(shap_found.keys()), key="shap_sel")
            st.image(shap_found[sel_shap], use_container_width=True)
        else:
            st.info("SHAP figures not found. Run notebooks 03 and 04 — SHAP plots are saved to `fig/`.")

        st.divider()
        section("Tariff Relativities — GLM Coefficients")

        RELAT_FIGS = {
            "GLM 3 — Tariff Relativities (Frequency)": "glm3_relativites.png",
            "Gamma GLM — Tariff Relativities (Severity)": "severity_glm_relativities.png",
            "Gamma GLM — Residuals & Validation":       "severity_glm_validation.png",
        }
        relat_found = {k: os.path.join(FIG, v) for k,v in RELAT_FIGS.items()
                       if os.path.exists(os.path.join(FIG,v))}

        if relat_found:
            sel_rel = st.selectbox("Select figure", list(relat_found.keys()), key="relat_sel")
            st.image(relat_found[sel_rel], use_container_width=True)
        else:
            st.info("Tariff relativity figures not found. Run notebooks 02 and 05.")

        st.divider()
        section("CANN — Learning Curve & Architecture")

        CANN_FIGS = {
            "CANN — Learning Curve (Frequency)":    "cann_learning_curve.png",
            "NN — Learning Curve (Severity)":       "severity_nn_learning_curve.png",
        }
        cann_found = {k: os.path.join(FIG, v) for k,v in CANN_FIGS.items()
                      if os.path.exists(os.path.join(FIG,v))}
        if cann_found:
            sel_cann = st.selectbox("Select figure", list(cann_found.keys()), key="cann_sel")
            st.image(cann_found[sel_cann], use_container_width=True)

        # Inline CANN explanation
        with st.expander("How does SHAP work for the CANN?"):
            st.markdown("""
The **CANN** (Combined Actuarial Neural Network) uses a **KernelExplainer** for SHAP
because TreeExplainer does not apply to neural networks.

KernelExplainer approximates SHAP values by:
1. Sampling a background dataset (reference = portfolio mean)
2. Perturbing each input feature and measuring the prediction change
3. Solving a weighted linear regression to assign contributions

The result is a SHAP value per feature per policy, showing:
- **Positive SHAP** → feature pushes frequency above the baseline
- **Negative SHAP** → feature pulls frequency below the baseline
- **BonusMalus** always dominates because it encodes claims history directly
            """)

    # ── Sub-tab C : Observed vs Predicted ─────────────────────────
    with ex3:
        section("Frequency — Observed vs Predicted")

        if df is not None and "fit_glm3" in df.columns:
            seg = st.selectbox("Segment by", ["BonusMalus","DrivAge","VehAge","Area"], key="seg_live")
            bins_n = st.slider("Number of bins", 5, 30, 15, key="bins_live")
            df_s = df.copy()
            if df_s[seg].dtype.name in ["object","category"]:
                sg = (df_s.groupby(seg, observed=True)
                      .agg(obs=("ClaimNb","sum"), ex=("Exposure","sum"), pred=("fit_glm3","sum"))
                      .reset_index()
                      .assign(obs_f=lambda x: x["obs"]/x["ex"],
                              pred_f=lambda x: x["pred"]/x["ex"]))
                x_vals = sg[seg].astype(str)
            else:
                df_s["bin"] = pd.cut(df_s[seg], bins=bins_n)
                sg = (df_s.groupby("bin", observed=True)
                      .agg(obs=("ClaimNb","sum"), ex=("Exposure","sum"), pred=("fit_glm3","sum"))
                      .reset_index()
                      .assign(obs_f=lambda x: x["obs"]/x["ex"],
                              pred_f=lambda x: x["pred"]/x["ex"],
                              mid=lambda x: x["bin"].apply(lambda b: b.mid)))
                x_vals = sg["mid"]

            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=x_vals, y=sg["obs_f"], mode="lines+markers",
                name="Observed", line=dict(color=C["gold"], width=2.5),
                marker=dict(size=7),
            ))
            fig_ov.add_trace(go.Scatter(
                x=x_vals, y=sg["pred_f"], mode="lines+markers",
                name="GLM 3 Predicted", line=dict(color=C["blue"], width=2, dash="dash"),
                marker=dict(size=6),
            ))
            fig_ov.update_layout(**LAYOUT,
                                 title=f"Observed vs GLM 3 Predicted — by {seg}",
                                 xaxis_title=seg, yaxis_title="Annualized Frequency",
                                 title_x=0.5, height=380)
            st.plotly_chart(fig_ov, use_container_width=True)

            # Residual chart: (obs - pred) / pred
            sg["rel_err"] = (sg["obs_f"] - sg["pred_f"]) / sg["pred_f"].clip(lower=1e-9) * 100
            fig_err = go.Figure(go.Bar(
                x=x_vals, y=sg["rel_err"],
                marker_color=[C["green"] if v <= 0 else C["red"] for v in sg["rel_err"]],
                text=[f"{v:+.1f}%" for v in sg["rel_err"]],
                textposition="outside",
            ))
            fig_err.add_hline(y=0, line_dash="dash", line_color="#888")
            fig_err.update_layout(**LAYOUT,
                                  title=f"Relative Error (Observed - Predicted) / Predicted — by {seg}",
                                  xaxis_title=seg, yaxis_title="Relative Error (%)",
                                  title_x=0.5, height=300)
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.info("Load frequency data and run notebook 02 to enable this chart.")

        st.divider()
        section("All Benchmark Figures")

        BENCH_FIGS = {
            "GLM Benchmark (Frequency)":         "glm_benchmark.png",
            "ML Benchmark (Frequency)":          "ml_benchmark_complet.png",
            "Final Benchmark — All Freq Models": "benchmark_final_complet.png",
            "Severity Benchmark":                "severity_benchmark.png",
            "Pure Premium Distribution":         "pure_premium.png",
        }
        bench_found = {k: os.path.join(FIG,v) for k,v in BENCH_FIGS.items()
                       if os.path.exists(os.path.join(FIG,v))}
        if bench_found:
            sel_b = st.selectbox("Select benchmark figure", list(bench_found.keys()), key="bench_sel")
            st.image(bench_found[sel_b], use_container_width=True)
        else:
            st.info("No benchmark figures yet. Run all notebooks.")

# ═══════════════════════════════════════════════════════════════════
# TAB 8 — CLIENT CLUSTERING
# ═══════════════════════════════════════════════════════════════════
with t8:
    st.markdown("## Client Clustering — Segmentation du Risque")
    st.markdown(
        "K-Means regroupe les assurés en segments homogènes à partir de leur profil "
        "et de leur prime pure. Source : notebook `05_Severity_Modeling_CORRIGE.ipynb`."
    )

    if clust_df is None:
        st.info("Lance `05_Severity_Modeling_CORRIGE.ipynb` pour générer `client_clusters.csv`.")
    else:
        # ── Detect risk columns ──────────────────────────────────
        HAS_RISK   = "Risk_Label" in clust_df.columns
        HAS_COLOR  = "Risk_Color" in clust_df.columns
        HAS_DESC   = "Risk_Desc"  in clust_df.columns
        HAS_FREQ   = "freq_glm3"  in clust_df.columns
        HAS_SEV    = "sev_glm"    in clust_df.columns

        CLUSTER_PALETTE = [C["blue"],C["gold"],C["green"],C["red"],C["purple"],C["orange"]]
        n_cl = clust_df["Cluster"].nunique()
        clusters_sorted_raw = sorted(clust_df["Cluster"].unique())

        # Build a color map: cluster → color (Risk_Color if available)
        if HAS_COLOR:
            cl_color_map = clust_df.groupby("Cluster")["Risk_Color"].first().to_dict()
            palette_cl   = [cl_color_map.get(c, CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)])
                            for i, c in enumerate(clusters_sorted_raw)]
        else:
            cl_color_map = {}
            palette_cl   = CLUSTER_PALETTE[:n_cl]

        # Build label map: cluster → risk label
        if HAS_RISK:
            cl_label_map = clust_df.groupby("Cluster")["Risk_Label"].first().to_dict()
        else:
            cl_label_map = {c: f"Cluster {c}" for c in clusters_sorted_raw}

        def cl_label(c):
            return cl_label_map.get(c, f"Cluster {c}")

        # ── Top KPIs ──────────────────────────────────────────────
        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1: kpi("Clusters", str(n_cl), "blue")
        with kc2: kpi("Polices", f"{len(clust_df):,}", "green")
        with kc3:
            lrg = clust_df["Cluster"].value_counts().idxmax()
            kpi("Cluster + grand", cl_label(lrg), "gold")
        with kc4:
            if HAS_FREQ:
                overall_freq = clust_df["freq_glm3"].mean()
                kpi("Fréquence moyenne", f"{overall_freq:.4f}", "purple")
            elif HAS_SEV:
                overall_sev = clust_df["sev_glm"].mean()
                kpi("Sévérité moyenne", f"{overall_sev:,.0f} €", "purple")
            else:
                sml = clust_df["Cluster"].value_counts().idxmin()
                kpi("Cluster + petit", cl_label(sml), "purple")

        # ── Risk Class Cards ──────────────────────────────────────
        if HAS_RISK:
            st.divider()
            section("Classes Tarifaires — Description par Segment")
            risk_summary = clust_df.groupby("Cluster").agg(
                Risk_Label=("Risk_Label","first"),
                Risk_Desc =("Risk_Desc","first") if HAS_DESC else ("Cluster","count"),
                Risk_Color=("Risk_Color","first") if HAS_COLOR else ("Cluster","count"),
                Policies  =("Cluster","count"),
                PP_GLM_mean=("PP_GLM","mean") if "PP_GLM" in clust_df.columns else ("Cluster","count"),
                freq_mean  =("freq_glm3","mean") if HAS_FREQ else ("Cluster","count"),
                sev_mean   =("sev_glm","mean")   if HAS_SEV  else ("Cluster","count"),
            ).reset_index().sort_values("PP_GLM_mean" if "PP_GLM" in clust_df.columns else "Policies")

            card_cols = st.columns(n_cl)
            for idx, row in enumerate(risk_summary.itertuples()):
                with card_cols[idx]:
                    color = row.Risk_Color if HAS_COLOR else CLUSTER_PALETTE[idx % len(CLUSTER_PALETTE)]
                    desc  = row.Risk_Desc  if HAS_DESC  else ""
                    pp_txt = f"{row.PP_GLM_mean:,.0f} €/an" if "PP_GLM" in clust_df.columns else "—"
                    fr_txt = f"{row.freq_mean:.4f}" if HAS_FREQ else "—"
                    sv_txt = f"{row.sev_mean:,.0f} €" if HAS_SEV else "—"
                    st.markdown(
                        f"""<div style="background:#fff;border-left:6px solid {color};
                        border-radius:8px;padding:14px 16px;box-shadow:0 1px 5px rgba(0,0,0,.10);
                        margin-bottom:6px">
                        <div style="font-size:.72rem;text-transform:uppercase;color:#888;
                        letter-spacing:.06em">Cluster {row.Cluster}</div>
                        <div style="font-size:1.05rem;font-weight:800;color:{color};margin:4px 0">
                        {row.Risk_Label}</div>
                        <div style="font-size:.78rem;color:#555;margin-bottom:8px">{desc}</div>
                        <hr style="border-color:#eee;margin:6px 0">
                        <div style="font-size:.78rem;color:#444">
                        <b>{row.Policies:,}</b> polices<br>
                        Prime pure : <b>{pp_txt}</b><br>
                        Fréquence : <b>{fr_txt}</b><br>
                        Sévérité : <b>{sv_txt}</b>
                        </div></div>""",
                        unsafe_allow_html=True,
                    )

        # ── PCA & Elbow figures ───────────────────────────────────
        cl_figs = {
            "Méthode Elbow (choix K)":       "clustering_elbow.png",
            "Projection PCA 2D des clusters": "clustering_results.png",
        }
        cl_found = {k: os.path.join(FIG, v) for k, v in cl_figs.items()
                    if os.path.exists(os.path.join(FIG, v))}
        if cl_found:
            st.divider()
            section("Visualisations (notebook 05)")
            cf_cols = st.columns(len(cl_found))
            for col_w, (title, path) in zip(cf_cols, cl_found.items()):
                with col_w:
                    st.image(path, caption=title, use_container_width=True)

        st.divider()

        # ── Merge for interactive plots ───────────────────────────
        # clust_df already has freq_glm3, sev_glm, PP_GLM; merge with df only for extra cols
        merged_cl = clust_df.copy()
        merged_cl["Cluster_str"] = merged_cl["Cluster"].astype(str)
        clusters_sorted = sorted(merged_cl["Cluster"].unique())
        lbl_x = lambda c: cl_label(c)  # noqa

        # ── Fréquence par cluster ─────────────────────────────────
        if HAS_FREQ:
            section("Fréquence du Sinistre par Segment (freq_glm3)")
            fa, fb = st.columns(2)
            with fa:
                fig_fv = go.Figure()
                for i, cl in enumerate(clusters_sorted):
                    vals = merged_cl.loc[merged_cl["Cluster"]==cl, "freq_glm3"].dropna()
                    cap  = vals.quantile(0.99)
                    vals = vals[vals <= cap]
                    color_v = palette_cl[i % len(palette_cl)]
                    fig_fv.add_trace(go.Violin(
                        y=vals, x=[lbl_x(cl)]*len(vals),
                        name=lbl_x(cl), box_visible=True, meanline_visible=True,
                        fillcolor=color_v, opacity=0.6, line_color="grey",
                    ))
                fig_fv.update_layout(**LAYOUT, violinmode="group",
                                     title="Fréquence sinistre — Violin par segment",
                                     yaxis_title="Fréquence (sin/police/an)",
                                     height=380, title_x=0.5)
                st.plotly_chart(fig_fv, use_container_width=True)

            with fb:
                freq_agg = (merged_cl.groupby("Cluster")
                            .agg(mean_freq=("freq_glm3","mean"),
                                 median_freq=("freq_glm3","median"))
                            .reset_index().sort_values("Cluster"))
                fig_fb = go.Figure()
                fig_fb.add_trace(go.Bar(
                    x=[lbl_x(c) for c in freq_agg["Cluster"]],
                    y=freq_agg["mean_freq"],
                    name="Moyenne",
                    marker_color=palette_cl[:len(freq_agg)],
                    text=[f"{v:.4f}" for v in freq_agg["mean_freq"]],
                    textposition="outside", opacity=0.85,
                ))
                fig_fb.update_layout(**LAYOUT,
                                     title="Fréquence moyenne par segment",
                                     yaxis_title="Fréquence moyenne",
                                     height=380, title_x=0.5)
                st.plotly_chart(fig_fb, use_container_width=True)

        # ── Sévérité par cluster ──────────────────────────────────
        if HAS_SEV:
            st.divider()
            section("Coût Moyen du Sinistre par Segment (sev_glm)")
            sa, sb = st.columns(2)
            with sa:
                fig_sv = go.Figure()
                for i, cl in enumerate(clusters_sorted):
                    vals = merged_cl.loc[merged_cl["Cluster"]==cl, "sev_glm"].dropna()
                    cap  = vals.quantile(0.99)
                    vals = vals[vals <= cap]
                    color_v = palette_cl[i % len(palette_cl)]
                    fig_sv.add_trace(go.Violin(
                        y=vals, x=[lbl_x(cl)]*len(vals),
                        name=lbl_x(cl), box_visible=True, meanline_visible=True,
                        fillcolor=color_v, opacity=0.6, line_color="grey",
                    ))
                fig_sv.update_layout(**LAYOUT, violinmode="group",
                                     title="Coût sinistre — Violin par segment",
                                     yaxis_title="Coût moyen (€)",
                                     height=380, title_x=0.5)
                st.plotly_chart(fig_sv, use_container_width=True)

            with sb:
                sev_agg = (merged_cl.groupby("Cluster")
                           .agg(mean_sev=("sev_glm","mean"),
                                median_sev=("sev_glm","median"))
                           .reset_index().sort_values("Cluster"))
                fig_sb = go.Figure()
                fig_sb.add_trace(go.Bar(
                    x=[lbl_x(c) for c in sev_agg["Cluster"]],
                    y=sev_agg["mean_sev"],
                    name="Moyenne",
                    marker_color=palette_cl[:len(sev_agg)],
                    text=[f"{v:,.0f} €" for v in sev_agg["mean_sev"]],
                    textposition="outside", opacity=0.85,
                ))
                fig_sb.update_layout(**LAYOUT,
                                     title="Sévérité moyenne par segment",
                                     yaxis_title="Coût moyen (€)",
                                     height=380, title_x=0.5)
                st.plotly_chart(fig_sb, use_container_width=True)

        # ── Prime Pure par cluster ────────────────────────────────
        st.divider()
        pp_col_cl = next((c for c in ["PP_GLM","PP_LGB"] if c in merged_cl.columns), None)
        if pp_col_cl:
            section(f"Prime Pure ({pp_col_cl}) par Segment")
            fig_ppv = go.Figure()
            cap_pp = merged_cl[pp_col_cl].quantile(0.99)
            for i, cl in enumerate(clusters_sorted):
                vals = merged_cl.loc[merged_cl["Cluster"]==cl, pp_col_cl].dropna()
                vals = vals[vals <= cap_pp]
                fig_ppv.add_trace(go.Violin(
                    y=vals, x=[lbl_x(cl)]*len(vals),
                    name=lbl_x(cl), box_visible=True, meanline_visible=True,
                    fillcolor=palette_cl[i % len(palette_cl)],
                    opacity=0.6, line_color="grey",
                ))
            fig_ppv.update_layout(**LAYOUT, violinmode="group",
                                  title=f"{pp_col_cl} — Distribution par Segment",
                                  yaxis_title="Prime Pure (€/an)", height=400, title_x=0.5)
            st.plotly_chart(fig_ppv, use_container_width=True)

        # ── Profil statistique par cluster ────────────────────────
        st.divider()
        section("Profil Statistique par Segment")
        profile_cols = [c for c in
            ["BonusMalus","DrivAge","VehAge","Density","Exposure",
             "freq_glm3","sev_glm","PP_GLM"]
            if c in merged_cl.columns]
        profile = (merged_cl.groupby("Cluster")[profile_cols]
                   .agg(["mean","median"]).round(3))
        profile.columns = [f"{col}_{stat}" for col, stat in profile.columns]
        profile = profile.reset_index()
        profile["Segment"] = profile["Cluster"].apply(cl_label)
        profile = profile.drop(columns=["Cluster"])
        st.dataframe(profile, use_container_width=True, height=250)

        # ── Distribution des features par cluster ────────────────
        st.divider()
        section("Distribution d'une Variable par Segment")
        feat_choices = [c for c in
            ["BonusMalus","DrivAge","VehAge","Density","freq_glm3","sev_glm","PP_GLM"]
            if c in merged_cl.columns]
        feat_sel = st.selectbox("Variable à comparer", feat_choices, key="clust_feat")

        fl, fr = st.columns(2)
        with fl:
            fig_box = go.Figure()
            for i, cl in enumerate(clusters_sorted):
                vals = merged_cl.loc[merged_cl["Cluster"]==cl, feat_sel].dropna()
                cap  = vals.quantile(0.99)
                fig_box.add_trace(go.Box(
                    y=vals[vals<=cap], name=lbl_x(cl),
                    marker_color=palette_cl[i % len(palette_cl)],
                    boxmean=True,
                ))
            fig_box.update_layout(**LAYOUT, title=f"{feat_sel} — Boxplot par segment",
                                  yaxis_title=feat_sel, height=400, title_x=0.5)
            st.plotly_chart(fig_box, use_container_width=True)

        with fr:
            means = merged_cl.groupby("Cluster")[feat_sel].mean().reset_index()
            means.columns = ["Cluster", "Mean"]
            means = means.sort_values("Cluster")
            fig_mean = go.Figure(go.Bar(
                x=[lbl_x(c) for c in means["Cluster"]],
                y=means["Mean"],
                marker_color=palette_cl[:len(means)],
                text=[f"{v:.3f}" for v in means["Mean"]],
                textposition="outside", opacity=0.85,
            ))
            fig_mean.update_layout(**LAYOUT, title=f"Moyenne {feat_sel} par segment",
                                   yaxis_title=f"Moyenne {feat_sel}", height=400, title_x=0.5)
            st.plotly_chart(fig_mean, use_container_width=True)

        # ── Scatter BonusMalus vs DrivAge ─────────────────────────
        st.divider()
        section("Nuage de Points — Profil du Conducteur par Segment")
        sample_cl = merged_cl.sample(min(15000, len(merged_cl)), random_state=42)
        sc_color  = "Risk_Label" if HAS_RISK else "Cluster_str"
        fig_sc = px.scatter(
            sample_cl, x="DrivAge", y="BonusMalus",
            color=sc_color,
            color_discrete_sequence=palette_cl,
            opacity=0.4,
            title="BonusMalus vs Âge conducteur — appartenance au segment (15k polices)",
            labels={"DrivAge":"Âge conducteur","BonusMalus":"BonusMalus",
                    "Risk_Label":"Segment","Cluster_str":"Cluster"},
            height=450,
        )
        fig_sc.update_traces(marker=dict(size=3))
        fig_sc.update_layout(**LAYOUT, title_x=0.5)
        st.plotly_chart(fig_sc, use_container_width=True)

        # ── Client Lookup ─────────────────────────────────────────
        st.divider()
        section("Recherche par Police — Profil de Risque Client")
        st.markdown("Sélectionne un numéro de police pour afficher son segment de risque et ses indicateurs.")

        id_col = "IDpol"
        if id_col in clust_df.columns:
            idpol_list = sorted(clust_df[id_col].dropna().astype(int).unique())
            sel_id = st.selectbox(
                "Numéro de police (IDpol)",
                idpol_list[:5000],   # cap to avoid huge dropdown
                key="client_lookup",
                format_func=lambda x: f"Police #{x}",
            )
            row = clust_df[clust_df[id_col] == sel_id]
            if len(row):
                r = row.iloc[0]
                cl_val   = int(r["Cluster"])
                r_label  = r.get("Risk_Label",  f"Cluster {cl_val}")
                r_desc   = r.get("Risk_Desc",   "—")
                r_color  = r.get("Risk_Color",  CLUSTER_PALETTE[cl_val % len(CLUSTER_PALETTE)])
                r_freq   = r.get("freq_glm3",   None)
                r_sev    = r.get("sev_glm",     None)
                r_pp     = r.get("PP_GLM",      None)
                r_bm     = r.get("BonusMalus",  None)
                r_dage   = r.get("DrivAge",     None)
                r_vage   = r.get("VehAge",      None)

                lu1, lu2 = st.columns([1,2])
                with lu1:
                    st.markdown(
                        f"""<div style="background:#fff;border-left:8px solid {r_color};
                        border-radius:10px;padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.12)">
                        <div style="font-size:.78rem;color:#888;text-transform:uppercase;
                        letter-spacing:.06em">Police #{sel_id}</div>
                        <div style="font-size:1.5rem;font-weight:900;color:{r_color};margin:8px 0">
                        {r_label}</div>
                        <div style="font-size:.88rem;color:#555;margin-bottom:12px">{r_desc}</div>
                        <hr style="border-color:#eee">
                        <div style="font-size:.85rem;color:#444;margin-top:10px">
                        {"<br>".join([
                            f"<b>Cluster :</b> {cl_val}",
                            f"<b>BonusMalus :</b> {int(r_bm) if r_bm is not None else '—'}",
                            f"<b>Âge conducteur :</b> {int(r_dage) if r_dage is not None else '—'}",
                            f"<b>Âge véhicule :</b> {int(r_vage) if r_vage is not None else '—'}",
                        ])}
                        </div></div>""",
                        unsafe_allow_html=True,
                    )
                with lu2:
                    ind_cols = st.columns(3)
                    with ind_cols[0]:
                        kpi("Fréquence GLM", f"{r_freq:.4f}" if r_freq is not None else "—", "blue")
                    with ind_cols[1]:
                        kpi("Sévérité GLM", f"{r_sev:,.0f} €" if r_sev is not None else "—", "orange")
                    with ind_cols[2]:
                        kpi("Prime Pure", f"{r_pp:,.0f} €" if r_pp is not None else "—", "gold")

                    # Show cluster average for comparison
                    cl_grp = clust_df[clust_df["Cluster"] == cl_val]
                    st.markdown("**Moyennes du segment :**")
                    avgs = {}
                    if HAS_FREQ:   avgs["Fréquence moy."]  = f"{cl_grp['freq_glm3'].mean():.4f}"
                    if HAS_SEV:    avgs["Sévérité moy."]   = f"{cl_grp['sev_glm'].mean():,.0f} €"
                    if "PP_GLM" in clust_df.columns:
                        avgs["Prime Pure moy."] = f"{cl_grp['PP_GLM'].mean():,.0f} €"
                    avgs["Polices"] = f"{len(cl_grp):,}"
                    avg_df = pd.DataFrame(list(avgs.items()), columns=["Indicateur","Valeur"])
                    st.dataframe(avg_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Police non trouvée dans les données de clustering.")

        # ── Download ──────────────────────────────────────────────
        st.divider()
        st.download_button(
            "⬇ Télécharger les données de clustering (CSV)",
            data=clust_df.to_csv(index=False),
            file_name="client_clusters.csv", mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════
# TAB 9 — UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════
with t9:
    st.markdown("## Upload and Score Your Portfolio")
    st.markdown("Upload a CSV file with the same structure as the MTPL dataset to get risk predictions.")

    with st.expander("Required CSV format"):
        st.markdown("""
| Column | Type | Example |
|---|---|---|
| `Exposure` | float 0–1 | 0.75 |
| `Area` | string A–F | B |
| `VehPower` | integer | 6 |
| `VehAge` | integer | 3 |
| `DrivAge` | integer | 35 |
| `BonusMalus` | integer 50–350 | 95 |
| `VehBrand` | string | B1 |
| `VehGas` | string | Regular |
| `Density` | integer | 1500 |
| `Region` | string | R11 |
        """)
        if df is not None:
            template_cols = ["Exposure","Area","VehPower","VehAge","DrivAge",
                             "BonusMalus","VehBrand","VehGas","Density","Region"]
            template_cols = [c for c in template_cols if c in df.columns]
            st.download_button(
                "Download sample template (10 rows)",
                data=df[template_cols].head(10).to_csv(index=False),
                file_name="mtpl_template.csv", mime="text/csv",
            )

    uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            st.success(f"Loaded: {len(udf):,} policies x {udf.shape[1]} columns")

            req = ["Exposure","Area","VehPower","VehAge","DrivAge","BonusMalus",
                   "VehBrand","VehGas","Density","Region"]
            missing = [c for c in req if c not in udf.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                udf["Exposure"]     = udf["Exposure"].clip(upper=1)
                udf["BonusMalusGLM"] = np.minimum(150, udf["BonusMalus"])

                global_freq = df["ClaimNb"].sum() / df["Exposure"].sum() if df is not None else 0.073
                udf["Predicted_Frequency"] = global_freq
                udf["Predicted_Claims"]    = global_freq * udf["Exposure"]
                udf["Risk_Tier"] = pd.cut(
                    udf["BonusMalus"],
                    bins=[0,80,100,120,999],
                    labels=["Preferred","Standard","High","Very High"]
                )

                st.divider()
                st.markdown("### Scoring Results")
                k1,k2,k3,k4 = st.columns(4)
                with k1: kpi("Policies Scored",   f"{len(udf):,}",                            "blue")
                with k2: kpi("Avg Frequency",      f"{udf['Predicted_Frequency'].mean():.4f}/yr","gold")
                with k3: kpi("Expected Claims",    f"{udf['Predicted_Claims'].sum():.1f}",      "purple")
                with k4: kpi("High Risk Policies", f"{(udf['BonusMalus']>120).sum():,}",        "red")

                st.dataframe(
                    udf[["Exposure","BonusMalus","DrivAge","VehAge",
                          "Predicted_Frequency","Predicted_Claims","Risk_Tier"]].head(50),
                    use_container_width=True, height=300,
                )

                l,r = st.columns(2)
                with l:
                    fig_dist = px.histogram(udf, x="BonusMalus", nbins=40,
                                            color_discrete_sequence=[C["gold"]],
                                            title="BonusMalus Distribution")
                    fig_dist.update_layout(**LAYOUT, title_x=0.5, height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)
                with r:
                    tc = udf["Risk_Tier"].value_counts().reset_index()
                    tc.columns = ["Tier","Count"]
                    fig_pie = px.pie(tc, names="Tier", values="Count",
                                    color_discrete_sequence=[C["green"],C["gold"],C["orange"],C["red"]],
                                    title="Risk Segmentation")
                    fig_pie.update_layout(paper_bgcolor="#f8f9fa", font_color="#212529",
                                          title_x=0.5, height=300)
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.download_button(
                    "Download scored portfolio CSV",
                    data=udf.to_csv(index=False),
                    file_name="scored_portfolio.csv", mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error: {e}")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#999;font-size:0.8rem">'
    'Claim Frequency & Severity Modeling — French Motor MTPL — '
    'Schelldorfer & Wuthrich (2019) — Built with Streamlit & Plotly'
    '</p>', unsafe_allow_html=True
)
