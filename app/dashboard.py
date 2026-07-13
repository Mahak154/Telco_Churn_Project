"""
Retention Radar — Customer Churn Analytics Dashboard
Run:  streamlit run dashboard.py

Layout:
  Tab 1 — Overview: fleet-wide analytics computed from data/telco_churn.csv
          run through the trained model (real numbers, not placeholders).
  Tab 2 — Scan Customer: ad-hoc single-account prediction with SHAP explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import plotly.graph_objects as go
import plotly.express as px

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "churn_model.joblib")
ENCODERS_PATH = os.path.join(BASE, "models", "label_encoders.joblib")
FEATURES_PATH = os.path.join(BASE, "models", "feature_columns.joblib")
DATA_PATH = os.path.join(BASE, "data", "telco_churn.csv")   # add this file to your repo

st.set_page_config(page_title="Retention Radar", page_icon="◎", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens — light, corporate SaaS theme
# ---------------------------------------------------------------------------
BG_BASE = "#000000A1"
BG_PANEL = "#FFFFFF"

TEXT_PRIMARY = "#DEEB28"
TEXT_MUTED = "#FFFFFF"

ACCENT = "#2563EB"

ACCENT_SOFT = "#DBEAFE"

RISK_LOW = "#16A34A"
RISK_MED = "#F59E0B"
RISK_HIGH = "#DC2626"

BORDER = "#F8F8F8"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

.stApp {{
    background: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', sans-serif;
}}
#MainMenu, footer, header {{visibility: hidden;}}
h1, h2, h3 {{ color: {TEXT_PRIMARY} !important; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {BG_PANEL};
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
    padding: 4px;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.05);
}}

.kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {TEXT_MUTED};
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 0.75rem;
    color: {TEXT_MUTED};
    margin-top: 4px;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; color: {TEXT_MUTED}; }}
.stTabs [data-baseweb="tab-list"] button {{
    font-family: 'Inter'; font-weight: 600; font-size: 0.9rem;color: {TEXT_MUTED}; 
}}

label, .stSelectbox label, .stSlider label, .stNumberInput label {{
    color: {TEXT_MUTED} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
div[data-baseweb="select"] > div, .stNumberInput input {{
    background: {BG_BASE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 6px !important;
}}
.stButton > button {{
    background: {ACCENT} !important;
    color: {TEXT_MUTED} !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.2rem !important;
}}
.stButton > button:hover {{ opacity: 0.92; }}
hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, encoders, feature_columns

model, encoders, feature_columns = load_artifacts()

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

def tenure_bucket(t):
    if t <= 12: return "0-1yr"
    elif t <= 24: return "1-2yr"
    elif t <= 48: return "2-4yr"
    else: return "4yr+"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering used at training time — applied here to a
    full dataframe (batch) rather than a single row."""
    df = df.copy()
    df["TenureBucket"] = df["tenure"].apply(tenure_bucket)
    df["AvgMonthlySpend"] = (df["TotalCharges"] / df["tenure"].clip(lower=1)).round(2)
    df["NumAddonServices"] = df[addon_cols].apply(lambda r: (r == "Yes").sum(), axis=1)
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        if col in df.columns and col != "Churn":
            df[col] = le.transform(df[col].astype(str))
    return df[feature_columns]

def build_feature_row(row: dict) -> pd.DataFrame:
    df = engineer_features(pd.DataFrame([row]))
    return encode_features(df)

@st.cache_data
def load_and_score_customers():
    """Loads the real customer dataset and scores every row with the
    trained model. This is what powers every number in the Overview tab —
    nothing here is a placeholder."""
    raw = pd.read_csv(DATA_PATH)
    feats = engineer_features(raw)
    encoded = encode_features(feats)
    proba = model.predict_proba(encoded)[:, 1]
    out = raw.copy()
    out["ChurnProbability"] = proba
    out["RiskBand"] = pd.cut(
        proba, bins=[-0.01, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"]
    )
    return out

customers = load_and_score_customers()


def kpi_card(label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div style="padding:14px 4px;">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def build_shap_chart(df_row: pd.DataFrame, top_n: int = 6) -> go.Figure:
    shap_values = explainer.shap_values(df_row)
    values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    contrib = pd.Series(values, index=feature_columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top_n]
    top = top.sort_values()
    colors = [RISK_HIGH if v > 0 else RISK_LOW for v in top.values]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker=dict(color=colors), text=[f"{v:+.2f}" for v in top.values],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color=TEXT_PRIMARY),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color=TEXT_MUTED, size=11),
        margin=dict(l=10, r=40, t=10, b=10), height=260,
        xaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=True,
                   zerolinecolor=TEXT_MUTED, showticklabels=False),
        yaxis=dict(showgrid=False), showlegend=False,
    )
    return fig


@st.cache_data
def build_importance_bars(_model, feature_columns, top_n=8):
    importances = pd.Series(_model.feature_importances_, index=feature_columns)
    top = importances.sort_values(ascending=False)[:top_n]
    max_val = top.max()
    return [(name, val / max_val, val) for name, val in top.items()]


def build_quadrant_scatter(df: pd.DataFrame):
    """Tenure vs. Monthly Spend, colored by predicted churn probability.
    Short-tenure + high-spend is the classic high-risk quadrant — this
    is a real, documented churn pattern, not decorative axis labels."""
    sample = df.sample(min(400, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="tenure", y="MonthlyCharges", color="ChurnProbability",
        color_continuous_scale=[RISK_LOW, RISK_MED, RISK_HIGH],
        labels={"tenure": "TENURE (MONTHS)", "MonthlyCharges": "MONTHLY SPEND (₹)"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color=TEXT_MUTED, size=10),
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER),
        coloraxis_colorbar=dict(title="Risk"),
    )
    return fig


def find_priority_segment(df: pd.DataFrame):
    """Groups by Contract + InternetService, finds the segment with the
    highest (size x avg risk) — a real, data-backed suggestion rather
    than invented copy."""
    grouped = df.groupby(["Contract", "InternetService"], observed=True).agg(
        count=("ChurnProbability", "size"),
        avg_risk=("ChurnProbability", "mean"),
        monthly_value=("MonthlyCharges", "sum"),
    )
    grouped = grouped[grouped["count"] >= 20]
    grouped["priority_score"] = grouped["count"] * grouped["avg_risk"]
    return grouped.sort_values("priority_score", ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:18px;">
    <div style="display:flex; align-items:baseline; gap:10px;">
        <span style="font-family:'Chakra Petch'; font-size:1.6rem; font-weight:700; color:{TEXT_PRIMARY};">◎ RETENTION RADAR</span>
    </div>
    <div style="font-family:'IBM Plex Mono'; font-size:0.72rem; color:{TEXT_MUTED};">
        MODEL: GRADIENT BOOSTING · TEST ROC-AUC 0.834
    </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_scan = st.tabs(["Overview", "Scan Customer"])

# ---------------------------------------------------------------------------
# TAB 1 — Overview (real, batch-computed analytics)
# ---------------------------------------------------------------------------
with tab_overview:
    actual_churn_rate = (customers["Churn"] == "Yes").mean() * 100
    predicted_churn_rate = customers["ChurnProbability"].mean() * 100
    high_risk = customers[customers["RiskBand"] == "High"]
    mrr_at_risk = high_risk["MonthlyCharges"].sum()
    retention_health = round(100 - predicted_churn_rate)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            kpi_card("Predicted Churn Rate", f"{predicted_churn_rate:.1f}%",
                     f"Observed (labeled data): {actual_churn_rate:.1f}%")
    with k2:
        with st.container(border=True):
            kpi_card("MRR at Risk", f"₹{mrr_at_risk:,.0f}",
                     f"{len(high_risk)} high-risk accounts")
    with k3:
        with st.container(border=True):
            kpi_card("Retention Health", f"{retention_health}/100")
    with k4:
        with st.container(border=True):
            kpi_card("Customers Scanned", f"{len(customers):,}",
                     f"{len(high_risk)} flagged high-risk ({len(high_risk)/len(customers)*100:.1f}%)")

    st.write("")
    left, right = st.columns([1.5, 1])

    with left:
        with st.container(border=True):
            st.markdown(f"<div style='font-weight:700; margin-bottom:2px;'>Retention Radar</div>"
                        f"<div style='color:{TEXT_MUTED}; font-size:0.78rem; margin-bottom:8px;'>"
                        f"Tenure vs. monthly spend, colored by predicted churn risk. "
                        f"Short-tenure + high-spend accounts cluster as highest risk.</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(build_quadrant_scatter(customers), use_container_width=True,
                            config={"displayModeBar": False})

    with right:
        with st.container(border=True):
            st.markdown(f"<div style='font-weight:700; margin-bottom:10px;'>Top Churn Indicators</div>",
                        unsafe_allow_html=True)
            for name, frac, val in build_importance_bars(model, feature_columns, top_n=6):
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:3px;">
                        <span>{name}</span>
                        <span style="color:{TEXT_MUTED}; font-family:'IBM Plex Mono';">{val:.3f}</span>
                    </div>
                    <div style="background:{BORDER}; border-radius:4px; height:6px;">
                        <div style="background:{ACCENT}; width:{frac*100:.0f}%; height:6px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")
        seg = find_priority_segment(customers)
        seg_name = seg.name
        with st.container(border=True):
            st.markdown(f"""
            <div style="background:{ACCENT_SOFT}; border-radius:8px; padding:16px; margin:2px;">
                <div style="color:{ACCENT}; font-size:0.7rem; font-weight:700; letter-spacing:0.06em; margin-bottom:6px;">
                    PRIORITY SEGMENT
                </div>
                <div style="font-weight:700; margin-bottom:6px;">
                    {seg_name[0]} · {seg_name[1]}
                </div>
                <div style="color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.5;">
                    {int(seg['count'])} customers, {seg['avg_risk']*100:.0f}% average predicted churn risk.
                    Combined monthly value: ₹{seg['monthly_value']:,.0f}.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown(f"<div style='font-weight:700; margin-bottom:10px;'>Critical Priority List</div>",
                    unsafe_allow_html=True)
        priority = customers.sort_values("ChurnProbability", ascending=False).head(10)
        display_df = priority[["customerID", "Contract", "InternetService",
                               "tenure", "MonthlyCharges", "ChurnProbability", "RiskBand"]].copy()
        display_df["ChurnProbability"] = (display_df["ChurnProbability"] * 100).round(1).astype(str) + "%"
        display_df.columns = ["Customer ID", "Contract", "Internet", "Tenure (mo)",
                              "Monthly (₹)", "Risk Score", "Risk Band"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2 — Scan Customer (single ad-hoc prediction, light-themed)
# ---------------------------------------------------------------------------
with tab_scan:
    st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem; margin-bottom:16px;'>"
                f"Enter account parameters for an ad-hoc churn risk prediction.</div>",
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        with st.container(border=True):
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    with col3:
        with st.container(border=True):
            st.markdown("**Account**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges (₹)", min_value=18.0, max_value=150.0, value=65.0, step=0.5)
            total_charges = st.number_input("Total Charges (₹)", min_value=0.0, max_value=10000.0,
                                     value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0)

    st.write("")
    predict_clicked = st.button("Run Scan", type="primary", use_container_width=True)

    if predict_clicked:
        row = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech_support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
        }
        df_row = build_feature_row(row)
        proba = float(model.predict_proba(df_row)[0][1])
        risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
        risk_color = {"High": RISK_HIGH, "Medium": RISK_MED, "Low": RISK_LOW}[risk]

        st.write("")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            with st.container(border=True):
                kpi_card("Churn Probability", f"{proba*100:.1f}%")
        with rcol2:
            with st.container(border=True):
                kpi_card("Risk Level", risk.upper())

        st.write("")
        with st.container(border=True):
            st.markdown(f"<div style='font-weight:700; margin-bottom:8px;'>Risk Factors For This Account</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(build_shap_chart(df_row), use_container_width=True,
                            config={"displayModeBar": False})