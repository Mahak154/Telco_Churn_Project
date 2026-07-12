"""
Retention Radar — Customer Churn Prediction Dashboard
Run:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import joblib
import os
import math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "models", "churn_model.joblib")
ENCODERS_PATH = os.path.join(BASE, "models", "label_encoders.joblib")
FEATURES_PATH = os.path.join(BASE, "models", "feature_columns.joblib")

st.set_page_config(page_title="Retention Radar", page_icon="◎", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG_BASE = "#0B1220"
BG_PANEL = "#141C2B"
BORDER = "#232D3F"
TEXT_PRIMARY = "#E7ECF3"
TEXT_MUTED = "#8B97AC"
ACCENT = "#23C4B7"
RISK_LOW = "#34D399"
RISK_MED = "#F5A623"
RISK_HIGH = "#F0575A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

.stApp {{
    background: {BG_BASE};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', sans-serif;
}}

#MainMenu, footer, header {{visibility: hidden;}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    color: {TEXT_PRIMARY} !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {BG_PANEL};
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
    padding: 4px;
}}

label, .stSelectbox label, .stSlider label, .stNumberInput label {{
    color: {TEXT_MUTED} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}

div[data-baseweb="select"] > div, .stNumberInput input {{
    background: {BG_BASE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 6px !important;
}}

.stButton > button {{
    background: {ACCENT} !important;
    color: #06231F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.7rem 1.2rem !important;
    transition: transform 0.12s ease, opacity 0.12s ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    opacity: 0.92;
}}

hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, encoders, feature_columns

model, encoders, feature_columns = load_artifacts()

addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]

def tenure_bucket(t):
    if t <= 12: return "0-1yr"
    elif t <= 24: return "1-2yr"
    elif t <= 48: return "2-4yr"
    else: return "4yr+"

def predict(row: dict) -> float:
    row = dict(row)
    row["TenureBucket"] = tenure_bucket(row["tenure"])
    row["AvgMonthlySpend"] = round(row["TotalCharges"] / max(row["tenure"], 1), 2)
    row["NumAddonServices"] = sum(1 for c in addon_cols if row[c] == "Yes")
    df = pd.DataFrame([row])
    for col, le in encoders.items():
        if col in df.columns and col != "Churn":
            df[col] = le.transform(df[col].astype(str))
    df = df[feature_columns]
    return float(model.predict_proba(df)[0][1])


def build_gauge_svg(proba: float) -> str:
    """Semicircular risk gauge — the dashboard's signature visual element."""
    cx, cy, r = 160, 170, 130

    def point(theta_deg):
        rad = math.radians(theta_deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)

    def theta_for(p):
        return 180 - 180 * p

    zones = [(0.0, 0.4, RISK_LOW), (0.4, 0.7, RISK_MED), (0.7, 1.0, RISK_HIGH)]
    arcs = ""
    for p_start, p_end, color in zones:
        x1, y1 = point(theta_for(p_start))
        x2, y2 = point(theta_for(p_end))
        arcs += (f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
                 f'stroke="{color}" stroke-width="22" fill="none" opacity="0.85"/>')

    needle_theta = theta_for(min(max(proba, 0), 1))
    nx, ny = point(needle_theta)
    needle_x = cx + (nx - cx) * 0.82
    needle_y = cy - (cy - ny) * 0.82

    risk_color = RISK_HIGH if proba >= 0.7 else RISK_MED if proba >= 0.4 else RISK_LOW

    svg = f'''
    <svg viewBox="0 0 320 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;">
        {arcs}
        <line x1="{cx}" y1="{cy}" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
              stroke="{TEXT_PRIMARY}" stroke-width="3.5" stroke-linecap="round"/>
        <circle cx="{cx}" cy="{cy}" r="8" fill="{TEXT_PRIMARY}"/>
        <circle cx="{cx}" cy="{cy}" r="4" fill="{BG_PANEL}"/>
        <text x="{cx}" y="{cy - 34}" text-anchor="middle"
              font-family="JetBrains Mono" font-size="34" font-weight="700"
              fill="{risk_color}">{proba*100:.1f}%</text>
        <text x="{cx}" y="{cy - 10}" text-anchor="middle"
              font-family="Inter" font-size="12" letter-spacing="1"
              fill="{TEXT_MUTED}">CHURN PROBABILITY</text>
    </svg>
    '''
    return svg


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:14px; margin-bottom:4px;">
    <span style="font-family:'Space Grotesk'; font-size:2.1rem; font-weight:700; color:{TEXT_PRIMARY};">
        Retention Radar
    </span>
    <span style="display:inline-flex; align-items:center; gap:6px; color:{ACCENT}; font-size:0.8rem; font-weight:600;">
        <span style="width:7px; height:7px; border-radius:50%; background:{ACCENT}; display:inline-block;"></span>
        LIVE MODEL
    </span>
</div>
<div style="color:{TEXT_MUTED}; font-size:0.95rem; margin-bottom:22px;">
    Enter a customer's account details to estimate churn risk and get a retention recommendation.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input panels
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown(f"<div style='color:{ACCENT}; font-weight:600; font-family:Space Grotesk; margin-bottom:10px;'>◈ Demographics</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    with st.container(border=True):
        st.markdown(f"<div style='color:{ACCENT}; font-weight:600; font-family:Space Grotesk; margin-bottom:10px;'>◈ Services</div>", unsafe_allow_html=True)
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
        st.markdown(f"<div style='color:{ACCENT}; font-weight:600; font-family:Space Grotesk; margin-bottom:10px;'>◈ Account</div>", unsafe_allow_html=True)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=150.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                         value=float(round(monthly_charges * max(tenure, 1), 2)), step=1.0)

st.write("")
predict_clicked = st.button("▸  Predict Churn Risk", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
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

    proba = predict(row)
    prediction = "Yes" if proba >= 0.5 else "No"
    risk = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
    risk_color = {"High": RISK_HIGH, "Medium": RISK_MED, "Low": RISK_LOW}[risk]

    messages = {
        "High": "This customer is at high risk of churning. Recommend proactive outreach — "
                "a loyalty discount, contract upgrade offer, or a personal check-in call.",
        "Medium": "This customer shows some churn risk signals. Light-touch monitoring or "
                  "a retention offer may help.",
        "Low": "This customer looks stable. No immediate action needed."
    }

    st.write("")
    gcol, rcol = st.columns([1, 1.4])

    with gcol:
        with st.container(border=True):
            st.markdown(build_gauge_svg(proba), unsafe_allow_html=True)

    with rcol:
        with st.container(border=True):
            st.markdown(f"""
            <div style="padding:6px 4px;">
                <div style="display:flex; gap:28px; margin-bottom:18px;">
                    <div>
                        <div style="color:{TEXT_MUTED}; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;">PREDICTION</div>
                        <div style="font-family:'JetBrains Mono'; font-size:1.4rem; font-weight:700; color:{TEXT_PRIMARY};">{prediction}</div>
                    </div>
                    <div>
                        <div style="color:{TEXT_MUTED}; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;">RISK LEVEL</div>
                        <div style="font-family:'JetBrains Mono'; font-size:1.4rem; font-weight:700; color:{risk_color};">{risk}</div>
                    </div>
                </div>
                <div style="border-left:3px solid {risk_color}; padding:10px 16px; background:{BG_BASE}; border-radius:0 6px 6px 0; color:{TEXT_PRIMARY}; font-size:0.92rem; line-height:1.5;">
                    {messages[risk]}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.write("")
st.markdown(f"""
<div style="color:{TEXT_MUTED}; font-size:0.78rem; border-top:1px solid {BORDER}; padding-top:12px; margin-top:10px;">
    Model: Tuned Gradient Boosting Classifier · Test-set ROC-AUC 0.834 · Internal tool
</div>
""", unsafe_allow_html=True)
