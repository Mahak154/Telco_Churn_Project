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
# Design tokens — HUD / radar-console theme
# ---------------------------------------------------------------------------
BG_BASE = "#080B10"
BG_PANEL = "#0F151D"
BG_PANEL_2 = "#0B1017"
BORDER = "#1E2A33"
TEXT_PRIMARY = "#E8EDF2"
TEXT_MUTED = "#65788A"
SIGNAL = "#FFB020"        # phosphor amber — primary signal color
SIGNAL_DIM = "#7A5A1E"
COOL = "#4CD3C2"           # secondary cyan — calm / low-risk data
RISK_LOW = "#4CD3C2"
RISK_MED = "#FFB020"
RISK_HIGH = "#FF4B4B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

.stApp {{
    background:
        linear-gradient(180deg, rgba(255,176,32,0.03) 0%, transparent 12%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(255,255,255,0.012) 40px),
        {BG_BASE};
    color: {TEXT_PRIMARY};
    font-family: 'IBM Plex Sans', sans-serif;
}}

#MainMenu, footer, header {{visibility: hidden;}}

h1, h2, h3 {{
    font-family: 'Chakra Petch', sans-serif !important;
    color: {TEXT_PRIMARY} !important;
}}

/* HUD panel with corner brackets */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_PANEL_2} 100%);
    border: 1px solid {BORDER} !important;
    border-radius: 3px;
    padding: 6px;
    position: relative;
}}
div[data-testid="stVerticalBlockBorderWrapper"]::before {{
    content: '';
    position: absolute; top: -1px; left: -1px;
    width: 16px; height: 16px;
    border-top: 2px solid {SIGNAL};
    border-left: 2px solid {SIGNAL};
    pointer-events: none;
}}
div[data-testid="stVerticalBlockBorderWrapper"]::after {{
    content: '';
    position: absolute; bottom: -1px; right: -1px;
    width: 16px; height: 16px;
    border-bottom: 2px solid {SIGNAL};
    border-right: 2px solid {SIGNAL};
    pointer-events: none;
}}

label, .stSelectbox label, .stSlider label, .stNumberInput label {{
    color: {TEXT_MUTED} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

div[data-baseweb="select"] > div, .stNumberInput input {{
    background: {BG_BASE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_PRIMARY} !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}

div[data-baseweb="select"] > div:hover {{
    border-color: {SIGNAL_DIM} !important;
}}

.stSlider [role="slider"] {{
    background-color: {SIGNAL} !important;
}}
.stSlider > div > div > div > div {{
    background: {SIGNAL} !important;
}}

.stButton > button {{
    background: {SIGNAL} !important;
    color: #1A1204 !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Chakra Petch', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.75rem 1.2rem !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
    box-shadow: 0 0 0 rgba(255,176,32,0);
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 18px rgba(255,176,32,0.25);
}}

hr {{ border-color: {BORDER} !important; }}

@keyframes sweep-rotate {{
    from {{ transform: rotate(0deg); }}
    to {{ transform: rotate(360deg); }}
}}
@keyframes blip-pulse {{
    0%, 100% {{ opacity: 1; r: 6; }}
    50% {{ opacity: 0.4; r: 10; }}
}}
@media (prefers-reduced-motion: reduce) {{
    .sweep-layer {{ animation: none !important; }}
}}
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


def build_radar_svg(proba: float) -> str:
    """Signature visual: a radar scope. Distance from center = churn risk."""
    cx, cy = 150, 150
    rings = [30, 62, 94, 126]
    zone_colors = [RISK_LOW, RISK_LOW, RISK_MED, RISK_HIGH]
    risk_color = RISK_HIGH if proba >= 0.7 else RISK_MED if proba >= 0.4 else RISK_LOW

    ring_circles = ""
    for i, r in enumerate(rings):
        ring_circles += (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" '
                          f'stroke="{zone_colors[i]}" stroke-width="1" opacity="0.28"/>')

    # crosshair
    crosshair = (
        f'<line x1="{cx-126}" y1="{cy}" x2="{cx+126}" y2="{cy}" stroke="{TEXT_MUTED}" stroke-width="0.6" opacity="0.35"/>'
        f'<line x1="{cx}" y1="{cy-126}" x2="{cx}" y2="{cy+126}" stroke="{TEXT_MUTED}" stroke-width="0.6" opacity="0.35"/>'
    )

    # blip: distance from center encodes risk, angle fixed toward upper-right for consistency
    angle_deg = 58
    dist = 20 + proba * 108
    rad = math.radians(angle_deg)
    bx = cx + dist * math.cos(rad)
    by = cy - dist * math.sin(rad)

    zone_labels = [
        (30, "SECURE"), (62, "STABLE"), (94, "WATCH"), (126, "CRITICAL")
    ]
    labels_svg = ""
    for r, txt in zone_labels:
        labels_svg += (f'<text x="{cx}" y="{cy - r - 4}" text-anchor="middle" '
                        f'font-family="IBM Plex Mono" font-size="8" letter-spacing="1.5" '
                        f'fill="{TEXT_MUTED}" opacity="0.55">{txt}</text>')

    svg = f'''
    <div style="position:relative; width:100%; max-width:320px; margin:0 auto;">
      <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" style="width:100%; display:block;">
        <circle cx="{cx}" cy="{cy}" r="126" fill="{BG_BASE}" opacity="0.4"/>
        {ring_circles}
        {crosshair}
        {labels_svg}
        <circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="{risk_color}" style="animation: blip-pulse 1.8s ease-in-out infinite;"/>
        <circle cx="{bx:.1f}" cy="{by:.1f}" r="2.5" fill="{TEXT_PRIMARY}"/>
        <circle cx="{cx}" cy="{cy}" r="2" fill="{SIGNAL}"/>
        <text x="{cx}" y="{cy + 145}" text-anchor="middle" font-family="IBM Plex Mono"
              font-size="11" font-weight="700" letter-spacing="2" fill="{risk_color}">
              {proba*100:.1f}% CHURN PROBABILITY
        </text>
      </svg>
      <div class="sweep-layer" style="
          position:absolute; top:0; left:0; width:100%; height:100%;
          border-radius:50%;
          background: conic-gradient(from 0deg, transparent 0deg, transparent 300deg, rgba(255,176,32,0.16) 340deg, rgba(255,176,32,0.32) 360deg);
          animation: sweep-rotate 5s linear infinite;
          pointer-events:none;
      "></div>
    </div>
    '''
    return svg


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:14px; margin-bottom:4px;">
    <span style="font-family:'Chakra Petch'; font-size:2.1rem; font-weight:700; color:{TEXT_PRIMARY}; letter-spacing:0.01em;">
        RETENTION RADAR
    </span>
    <span style="display:inline-flex; align-items:center; gap:6px; color:{SIGNAL}; font-size:0.75rem; font-weight:600; font-family:'IBM Plex Mono'; letter-spacing:0.08em;">
        <span style="width:7px; height:7px; border-radius:50%; background:{SIGNAL}; display:inline-block; box-shadow:0 0 6px {SIGNAL};"></span>
        SIGNAL: LIVE
    </span>
</div>
<div style="color:{TEXT_MUTED}; font-size:0.92rem; margin-bottom:22px; font-family:'IBM Plex Mono';">
    &gt; Enter account parameters to scan for churn risk and retrieve a retention directive.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input panels
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown(f"<div style='color:{SIGNAL}; font-weight:700; font-family:Chakra Petch; letter-spacing:0.04em; margin-bottom:10px;'>01 · DEMOGRAPHICS</div>", unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    with st.container(border=True):
        st.markdown(f"<div style='color:{SIGNAL}; font-weight:700; font-family:Chakra Petch; letter-spacing:0.04em; margin-bottom:10px;'>02 · SERVICES</div>", unsafe_allow_html=True)
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
        st.markdown(f"<div style='color:{SIGNAL}; font-weight:700; font-family:Chakra Petch; letter-spacing:0.04em; margin-bottom:10px;'>03 · ACCOUNT</div>", unsafe_allow_html=True)
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
predict_clicked = st.button("▸  RUN SCAN", type="primary", use_container_width=True)

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
        "High": "TARGET LOCKED — high churn probability. Recommend immediate outreach: "
                "loyalty discount, contract upgrade offer, or a personal check-in call.",
        "Medium": "ELEVATED SIGNAL — some churn risk detected. Light-touch monitoring or "
                  "a retention offer is advised.",
        "Low": "CLEAR — customer reads as stable. No immediate action required."
    }

    st.write("")
    gcol, rcol = st.columns([1, 1.4])

    with gcol:
        with st.container(border=True):
            st.markdown(build_radar_svg(proba), unsafe_allow_html=True)

    with rcol:
        with st.container(border=True):
            st.markdown(f"""
            <div style="padding:6px 4px;">
                <div style="display:flex; gap:28px; margin-bottom:18px;">
                    <div>
                        <div style="color:{TEXT_MUTED}; font-size:0.72rem; font-family:'IBM Plex Mono'; letter-spacing:0.08em; margin-bottom:4px;">WILL CHURN</div>
                        <div style="font-family:'IBM Plex Mono'; font-size:1.4rem; font-weight:700; color:{TEXT_PRIMARY};">{prediction}</div>
                    </div>
                    <div>
                        <div style="color:{TEXT_MUTED}; font-size:0.72rem; font-family:'IBM Plex Mono'; letter-spacing:0.08em; margin-bottom:4px;">RISK LEVEL</div>
                        <div style="font-family:'IBM Plex Mono'; font-size:1.4rem; font-weight:700; color:{risk_color};">{risk.upper()}</div>
                    </div>
                </div>
                <div style="border-left:3px solid {risk_color}; padding:10px 16px; background:{BG_BASE}; border-radius:0 2px 2px 0; color:{TEXT_PRIMARY}; font-size:0.9rem; line-height:1.55; font-family:'IBM Plex Mono';">
                    {messages[risk]}
                </div>
            </div>
            """, unsafe_allow_html=True)

st.write("")
st.markdown(f"""
<div style="color:{TEXT_MUTED}; font-size:0.75rem; font-family:'IBM Plex Mono'; border-top:1px solid {BORDER}; padding-top:12px; margin-top:10px; letter-spacing:0.03em;">
    MODEL: TUNED GRADIENT BOOSTING CLASSIFIER · TEST-SET ROC-AUC 0.834 · INTERNAL TOOL
</div>
""", unsafe_allow_html=True)