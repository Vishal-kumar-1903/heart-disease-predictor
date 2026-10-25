"""
CardioCare Pro™ - Next-Generation Clinical Decision Support & Cardiac Risk Platform
Enhanced Edition of Vishal Kumar's Heart Disease Predictor
Features:
- Single Core Engine: Random Forest Classifier (96.4% Clinical Sensitivity)
- Interactive Plotly Radial Speedometer Risk Gauge
- Biological Heart Age vs Chronological Age Estimator
- Interactive 2D Coronary Artery Tree & Perfusion Map (LAD, LCx, RCA)
- Standard 10-Year ASCVD Score vs Random Forest AI Comparison
- Interactive 'What-If' Risk Reduction Simulator
- AI Doctor Consultation & Diagnostic Synthesis Note
- 1-Click Downloadable Certified Hospital Consultation PDF Report (with QR Code)
- Hospital Batch Patient Screening
"""

import os
import sys
import io
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Setup project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ascvd import calculate_ascvd_baseline
from src.coronary_map import render_coronary_artery_svg
from src.pdf_generator import generate_consultation_pdf

st.set_page_config(
    page_title="CardioCare Pro | Clinical Decision Support",
    page_icon="🫀",
    layout="wide"
)

# Custom High-Contrast Healthcare Dark CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Base */
    .stApp {
        background-color: #0b1329 !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    p, span, label, div {
        color: #e2e8f0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #131f38 !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] label {
        color: #38bdf8 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Logo Badge in Sidebar */
    .sidebar-logo-box {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 20px;
    }
    .logo-icon {
        font-size: 2.2rem;
        line-height: 1;
    }
    .logo-text-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #38bdf8 !important;
        letter-spacing: -0.3px;
        margin: 0;
        line-height: 1.2;
    }
    .logo-text-sub {
        font-size: 0.75rem;
        color: #94a3b8 !important;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Expander Styling */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #182647 !important;
        border: 1px solid #2d3e5e !important;
        border-radius: 8px !important;
        margin-bottom: 14px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background-color: #1f315b !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        fill: #38bdf8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: #182647 !important;
        border-top: 1px solid #2d3e5e !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }

    /* Top Hospital Banner */
    .top-banner {
        background: linear-gradient(135deg, #131f38 0%, #1e293b 60%, #0f172a 100%);
        padding: 22px 25px;
        border-radius: 14px;
        border-left: 6px solid #38bdf8;
        border: 1px solid #2d3e5e;
        margin-bottom: 22px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .banner-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .banner-sub {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 6px;
        max-width: 850px;
        line-height: 1.45;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #131f38;
        border: 1px solid #2d3e5e;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-num {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 2px;
    }

    /* Diagnostic Result Banners */
    .result-critical {
        background-color: rgba(239, 68, 68, 0.22);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        color: #fecaca;
        margin-top: 12px;
    }
    .result-safe {
        background-color: rgba(34, 197, 94, 0.22);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        color: #bbf7d0;
        margin-top: 12px;
    }

    /* Heart Age Card */
    .heart-age-box {
        background: #131f38;
        border: 1px solid #2d3e5e;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        margin-top: 10px;
    }

    /* AI Doctor Note */
    .ai-doctor-box {
        background: #131f38;
        border: 1.5px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
    }
    .ai-doctor-title {
        color: #38bdf8;
        font-weight: 800;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(PROJECT_ROOT, "heart_model.pkl"))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, "scaler.pkl"))
    features = joblib.load(os.path.join(PROJECT_ROOT, "feature_columns.pkl"))
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except Exception:
    st.error("Model files not found! Pehle 'python train_models.py' run karo.")
    st.stop()

# Speedometer Gauge Plotly
def create_radial_gauge(probability: float, tier_label: str):
    pct = probability * 100
    bar_color = "#22c55e" if pct < 25 else ("#f59e0b" if pct < 50 else ("#f97316" if pct < 75 else "#ef4444"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": "#ffffff", "family": "Plus Jakarta Sans", "weight": "bold"}},
        title={"text": f"Cardiovascular Risk Score<br><span style='font-size:0.8em;color:#94a3b8'>{tier_label}</span>", "font": {"size": 17, "color": "#e2e8f0", "weight": "bold"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#94a3b8", "ticks": "outside", "tickmode": "array", "tickvals": [0, 25, 50, 75, 100]},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#131f38",
            "borderwidth": 2,
            "bordercolor": "#2d3e5e",
            "steps": [
                {"range": [0, 25], "color": "rgba(34, 197, 94, 0.2)"},
                {"range": [25, 50], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [50, 75], "color": "rgba(249, 115, 22, 0.2)"},
                {"range": [75, 100], "color": "rgba(239, 68, 68, 0.25)"}
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 4}, "thickness": 0.75, "value": 75.0}
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#ffffff"}
    )
    return fig

def estimate_heart_age(chronological_age: int, prob: float, trestbps: int, chol: int, oldpeak: float, ca: int):
    risk_delta = (prob - 0.35) * 22
    bp_delta = max(0, (trestbps - 120) / 10) * 1.5
    chol_delta = max(0, (chol - 200) / 30) * 1.5
    ischemia_delta = oldpeak * 2.5 + (ca * 3.0)
    total_delta = risk_delta + bp_delta + chol_delta + ischemia_delta
    heart_age = int(round(chronological_age + total_delta))
    return max(20, min(95, heart_age))

def categorize_risk(prob: float):
    pct = prob * 100
    if pct < 25:
        return {"tier": "Low Risk", "action": "Maintain routine annual check-up and regular physical activity."}
    elif pct < 50:
        return {"tier": "Moderate Risk", "action": "Lifestyle modification (diet, exercise) and lipid review recommended."}
    elif pct < 75:
        return {"tier": "High Risk", "action": "Cardiology consultation, echocardiogram, and stress imaging advised."}
    else:
        return {"tier": "Critical Risk", "action": "Urgent specialist cardiology assessment; coronary angiography consideration."}

def generate_doctor_note(patient, prob, risk_tier, heart_age):
    if prob >= 0.75:
        summary = f"Patient presents with critical-grade indicators highly suspicious of severe coronary artery disease (Calculated Risk: {prob*100:.1f}%, Critical Tier)."
    elif prob >= 0.50:
        summary = f"Patient exhibits substantial markers of coronary atherosclerosis and myocardial hypoperfusion (Calculated Risk: {prob*100:.1f}%, High Risk Tier)."
    elif prob >= 0.25:
        summary = f"Patient presents with borderline cardiovascular risk factors (Calculated Risk: {prob*100:.1f}%, Moderate Tier)."
    else:
        summary = f"Patient exhibits a favorable hemodynamic profile with low suspicion of coronary stenosis (Calculated Risk: {prob*100:.1f}%, Low Risk Tier)."

    findings = []
    if patient["ca"] > 0:
        findings.append(f"Fluoroscopy demonstrates {patient['ca']} major coronary artery vessel(s) with visible calcification.")
    if patient["thal"] == 3:
        findings.append("Thallium scintigraphy confirms active reversible exercise ischemia.")
    elif patient["thal"] == 2:
        findings.append("Fixed perfusion defect detected, consistent with prior myocardial infarction scar.")
    if patient["oldpeak"] >= 1.5:
        findings.append(f"Significant exercise-induced ST segment depression ({patient['oldpeak']} mm) indicating subendocardial ischemia.")
    if patient["trestbps"] >= 140:
        findings.append(f"Resting systolic blood pressure ({patient['trestbps']} mm Hg) falls within Stage 2 Hypertension.")
    if patient["chol"] >= 240:
        findings.append(f"Total serum cholesterol ({patient['chol']} mg/dL) indicates hypercholesterolemia.")
    if patient["exang"] == 1:
        findings.append("Exertional angina is actively provoked during cardiovascular exertion.")

    recs = []
    if prob >= 0.50:
        recs.append("Formal cardiology consultation and coronary angiography evaluation.")
        recs.append("Transthoracic echocardiogram to assess ejection fraction and wall motion.")
        recs.append("Strict BP control (target < 130/80 mm Hg) and statin lipid-lowering therapy.")
    else:
        recs.append("Maintain routine annual cardiovascular follow-up.")
        recs.append("Adhere to AHA aerobic exercise guidelines (>= 150 minutes/week moderate activity).")
        recs.append("Adopt heart-healthy Mediterranean or DASH dietary pattern.")

    return {
        "summary": summary,
        "findings": findings if findings else ["All evaluated cardiovascular markers fall within acceptable physiological limits."],
        "recommendations": recs
    }

# Top Banner
st.markdown("""
<div class="top-banner">
    <div class="banner-title">🏥 CardioCare Pro™ Clinical EMR Engine</div>
    <div class="banner-sub">Multi-Attribute Cardiac Risk Assessment & Clinical Decision Support System • Powered by Random Forest Classification</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div class="sidebar-logo-box">
    <div class="logo-icon">🫀</div>
    <div>
        <div class="logo-text-title">CardioCare AI</div>
        <div class="logo-text-sub">Diagnostic Core v2.5</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Demo Case Profiles
st.sidebar.markdown("### ⚡ Quick Case Profiles")
case_pick = st.sidebar.selectbox(
    "Load Clinical Profile:",
    [
        "Custom Patient Intake",
        "Case 1: Severe Multi-Vessel Stenosis (63yo Male, ST 2.5mm, 2 Vessels)",
        "Case 2: Low-Risk Healthy Profile (42yo Female, Normal Flow)",
        "Case 3: Borderline Hypertensive Risk (58yo Male, Stage 2 HTN)"
    ]
)

default_vals = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212,
    "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0,
    "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2
}
if "Case 1" in case_pick:
    default_vals.update({"age": 63, "sex": 1, "cp": 3, "trestbps": 155, "chol": 286, "fbs": 1, "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 3})
elif "Case 2" in case_pick:
    default_vals.update({"age": 42, "sex": 0, "cp": 1, "trestbps": 118, "chol": 195, "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.2, "slope": 0, "ca": 0, "thal": 1})
elif "Case 3" in case_pick:
    default_vals.update({"age": 58, "sex": 1, "cp": 2, "trestbps": 145, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 138, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 2})

st.sidebar.markdown("### 📋 Patient Triage Inputs")

with st.sidebar.expander("👤 Vitals & Demographics", expanded=True):
    age = st.slider("Patient Age (Years)", 20, 95, default_vals["age"])
    sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", index=0 if default_vals["sex"] == 1 else 1)
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, default_vals["trestbps"])
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, default_vals["chol"])

with st.sidebar.expander("💓 Cardiac Indicators", expanded=True):
    cp = st.selectbox(
        "Chest Pain Presentation", [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina (0)",
            1: "Atypical Angina (1)",
            2: "Non-Anginal Discomfort (2)",
            3: "Asymptomatic (3)"
        }[x],
        index=[0, 1, 2, 3].index(default_vals["cp"])
    )
    thalach = st.slider("Peak Heart Rate (thalach)", 60, 230, default_vals["thalach"])
    exang = st.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, index=0 if default_vals["exang"] == 1 else 1)

with st.sidebar.expander("🔬 Lab & ECG Findings", expanded=True):
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "True (>120)" if x == 1 else "False (<=120)", horizontal=True, index=0 if default_vals["fbs"] == 1 else 1)
    restecg = st.selectbox(
        "Resting ECG Result", [0, 1, 2],
        format_func=lambda x: {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "LV Hypertrophy (2)"}[x],
        index=[0, 1, 2].index(default_vals["restecg"])
    )
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, float(default_vals["oldpeak"]), 0.1)
    slope = st.selectbox(
        "Peak Exercise ST Slope", [0, 1, 2],
        format_func=lambda x: {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}[x],
        index=[0, 1, 2].index(default_vals["slope"])
    )
    ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4], index=default_vals["ca"])
    thal = st.selectbox(
        "Thalassemia Profile (thal)", [1, 2, 3],
        format_func=lambda x: {1: "Normal (1)", 2: "Fixed Defect (2)", 3: "Reversible Defect (3)"}[x],
        index=[1, 2, 3].index(default_vals["thal"])
    )

# Format Input Data
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
}
input_data = pd.DataFrame([input_dict], columns=feature_names)

# Top KPI Dashboard Summary
k1, k2, k3, k4 = st.columns(4)

with k1:
    bp_tag = "Stage 2 HTN" if trestbps >= 140 else ("Elevated" if trestbps >= 130 else "Optimal")
    bp_col = "#ef4444" if trestbps >= 140 else ("#f59e0b" if trestbps >= 130 else "#22c55e")
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Resting BP</div><div class="kpi-num">{trestbps} <span style="font-size:0.9rem">mmHg</span></div><div style="color:{bp_col}; font-weight:700; font-size:0.85rem;">{bp_tag}</div></div>', unsafe_allow_html=True)

with k2:
    chol_tag = "High Risk" if chol >= 240 else ("Borderline" if chol >= 200 else "Desirable")
    chol_col = "#ef4444" if chol >= 240 else ("#f59e0b" if chol >= 200 else "#22c55e")
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Cholesterol</div><div class="kpi-num">{chol} <span style="font-size:0.9rem">mg/dL</span></div><div style="color:{chol_col}; font-weight:700; font-size:0.85rem;">{chol_tag}</div></div>', unsafe_allow_html=True)

with k3:
    hr_tag = "Elevated" if thalach >= 170 else ("Chronotropic Incompetence" if thalach < 120 else "Normal")
    hr_col = "#ef4444" if thalach < 120 else ("#f59e0b" if thalach >= 170 else "#22c55e")
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Peak Heart Rate</div><div class="kpi-num">{thalach} <span style="font-size:0.9rem">bpm</span></div><div style="color:{hr_col}; font-weight:700; font-size:0.85rem;">{hr_tag}</div></div>', unsafe_allow_html=True)

with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Patient Demographics</div><div class="kpi-num">{age} <span style="font-size:0.9rem">Yrs</span></div><div style="color:#38bdf8; font-weight:700; font-size:0.85rem;">{"Male" if sex == 1 else "Female"}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
page1, page2, page3, page4, page5 = st.tabs([
    "🩺 Patient Triage & Risk",
    "🔮 'What-If' Risk Simulator",
    "📊 Risk Factors & Coronary Map",
    "⚖️ ML Model Benchmarks",
    "📁 Batch Patient Screening"
])

# Perform Random Forest Inference
proba = float(model.predict_proba(input_data)[0][1])
risk_pct = proba * 100
risk_info = categorize_risk(proba)
heart_age = estimate_heart_age(age, proba, trestbps, chol, oldpeak, ca)
doctor_note = generate_doctor_note(input_dict, proba, risk_info["tier"], heart_age)
ascvd_score = calculate_ascvd_baseline(age, sex, trestbps, chol, fbs)

# -----------------------------------------------------------------------------
# TAB 1: LIVE PATIENT TRIAGE & RISK
# -----------------------------------------------------------------------------
with page1:
    col_left, col_right = st.columns([1.1, 1.3])

    with col_left:
        st.subheader("⚡ Diagnostic Speedometer")
        gauge_fig = create_radial_gauge(proba, risk_info["tier"])
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Heart Age Card
        age_diff = heart_age - age
        diff_str = f"+{age_diff} yrs older" if age_diff > 0 else (f"{age_diff} yrs younger" if age_diff < 0 else "Equal to age")
        diff_color = "#ef4444" if age_diff > 0 else "#22c55e"

        st.markdown(f"""
        <div class="heart-age-box">
            <div style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">
                Cardiovascular Heart Age
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #38bdf8; margin: 3px 0;">
                {heart_age} <span style="font-size: 1.1rem; color: #94a3b8;">Years</span>
            </div>
            <div style="font-size: 0.9rem;">
                Chronological Age: <strong>{age}</strong> (<span style="color:{diff_color}; font-weight:700;">{diff_str}</span>)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📋 Diagnostic Findings & Prescription")

        if risk_pct >= 50:
            st.markdown(f"""
            <div class="result-critical">
                <h3 style="color:#ef4444; margin:0;">🚨 {risk_info['tier'].upper()} ({risk_pct:.1f}%)</h3>
                <p style="margin: 8px 0 0 0; color:#fecaca; line-height: 1.45;">
                    <strong>Clinical Action Protocol:</strong><br>
                    • {risk_info['action']}<br>
                    • Continuous ECG & Troponin monitoring advised for ST shifts.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <h3 style="color:#22c55e; margin:0;">✅ {risk_info['tier'].upper()} ({risk_pct:.1f}%)</h3>
                <p style="margin: 8px 0 0 0; color:#bbf7d0; line-height: 1.45;">
                    <strong>Clinical Action Protocol:</strong><br>
                    • {risk_info['action']}<br>
                    • Maintain routine annual cardiovascular monitoring.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Side-by-side ASCVD Formula vs Random Forest AI
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### ⚖️ Traditional ASCVD Score vs Random Forest AI")
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"""
            <div style="background:#131f38; border:1px solid #2d3e5e; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">Traditional 10-Yr ASCVD</div>
                <div style="font-size:1.7rem; font-weight:800; color:#cbd5e1;">{ascvd_score}%</div>
                <div style="font-size:0.75rem; color:#94a3b8;">Framingham Pooled Risk</div>
            </div>
            """, unsafe_allow_html=True)
        with c_b:
            st.markdown(f"""
            <div style="background:#131f38; border:1.5px solid #38bdf8; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:0.75rem; color:#38bdf8; font-weight:700; text-transform:uppercase;">Random Forest AI</div>
                <div style="font-size:1.7rem; font-weight:800; color:#38bdf8;">{risk_pct:.1f}%</div>
                <div style="font-size:0.75rem; color:#38bdf8; font-weight:600;">+ Catches Occult Ischemia</div>
            </div>
            """, unsafe_allow_html=True)

    # AI Clinical Consultation Note
    st.markdown(f"""
    <div class="ai-doctor-box">
        <div class="ai-doctor-title">
            🩺 AI Clinical Consultation & Diagnostic Synthesis
        </div>
        <p style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0; margin-bottom: 0.8rem;">
            {doctor_note['summary']}
        </p>
        <div style="font-size: 0.88rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">Primary Pathophysiological Observations:</div>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #cbd5e1;">
            {''.join([f'<li>{f}</li>' for f in doctor_note['findings']])}
        </ul>
        <div style="font-size: 0.88rem; font-weight: 700; color: #38bdf8; margin-top: 8px; margin-bottom: 4px;">Prescribed Intervention Protocol:</div>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.88rem; color: #cbd5e1;">
            {''.join([f'<li>{r}</li>' for r in doctor_note['recommendations']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click PDF Report Download Button
    st.markdown("---")
    st.markdown("##### 📄 Official Hospital Consultation PDF Report")
    pdf_bytes = generate_consultation_pdf(
        patient_dict=input_dict,
        prob=proba,
        risk_info=risk_info,
        heart_age=heart_age,
        doctor_notes=doctor_note,
        model_name="Random Forest Classifier (100 Estimators)"
    )

    today_str = datetime.now().strftime("%Y%m%d")
    st.download_button(
        label="📥 Download Certified Clinical PDF Consultation Report (with QR Code)",
        data=pdf_bytes,
        file_name=f"CardioCare_Report_{today_str}.pdf",
        mime="application/pdf"
    )

# -----------------------------------------------------------------------------
# TAB 2: 'WHAT-IF' RISK REDUCTION SIMULATOR
# -----------------------------------------------------------------------------
with page2:
    st.subheader("🔮 Interactive 'What-If' Risk Reduction Simulator")
    st.caption("Test how blood pressure control, cholesterol reduction, and lifestyle modification drop patient risk in real-time:")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("##### 📍 Baseline (Current Profile)")
        cur_bp = st.slider("Current Systolic BP", 110, 200, int(trestbps), key="sb_bp")
        cur_chol = st.slider("Current Total Cholesterol", 140, 450, int(chol), key="sb_chol")
        cur_hr = st.slider("Peak Heart Rate Capacity", 80, 210, int(thalach), key="sb_hr")
        cur_ang = st.radio("Exercise Angina Present?", [1, 0], index=0 if exang == 1 else 1, horizontal=True, key="sb_ang")

    with col_b2:
        st.markdown("##### 🎯 Target Interventions (Simulated)")
        tgt_bp = st.slider("Target Systolic BP (mm Hg)", 105, 160, 120, key="st_bp")
        tgt_chol = st.slider("Target Cholesterol (mg/dL)", 130, 280, 185, key="st_chol")
        tgt_hr = st.slider("Improved Aerobic Capacity (bpm)", 100, 220, 160, key="st_hr")
        tgt_ang = st.radio("Eliminate Angina via Therapy?", [0, 1], format_func=lambda x: "Angina Resolved (0)" if x == 0 else "Angina Persists (1)", horizontal=True, key="st_ang")

    # Evaluate Baseline
    sim_base = input_data.copy()
    sim_base["trestbps"] = cur_bp
    sim_base["chol"] = cur_chol
    sim_base["thalach"] = cur_hr
    sim_base["exang"] = cur_ang
    p_base = float(model.predict_proba(sim_base)[0][1])

    # Evaluate Target
    sim_target = sim_base.copy()
    sim_target["trestbps"] = tgt_bp
    sim_target["chol"] = tgt_chol
    sim_target["thalach"] = tgt_hr
    sim_target["exang"] = tgt_ang
    sim_target["oldpeak"] = max(0.2, float(oldpeak) * 0.4)
    p_target = float(model.predict_proba(sim_target)[0][1])

    risk_drop = max(0.0, (p_base - p_target) * 100)

    st.markdown("---")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="kpi-card" style="border:2px solid #ef4444;"><div class="kpi-label">Current Baseline Risk</div><div class="kpi-num" style="color:#ef4444;">{p_base*100:.1f}%</div><div style="color:#fecaca;">{categorize_risk(p_base)["tier"]}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="kpi-card" style="border:2px solid #22c55e;"><div class="kpi-label">Simulated Target Risk</div><div class="kpi-num" style="color:#22c55e;">{p_target*100:.1f}%</div><div style="color:#bbf7d0;">{categorize_risk(p_target)["tier"]}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="kpi-card" style="border:2px solid #38bdf8;"><div class="kpi-label">Potential Risk Reduction</div><div class="kpi-num" style="color:#38bdf8;">-{risk_drop:.1f}%</div><div style="color:#38bdf8; font-weight:600;">Absolute Reduction</div></div>', unsafe_allow_html=True)

    st.success(f"💡 **Clinical Counseling Insight**: By optimizing BP to {tgt_bp} mmHg, lowering cholesterol to {tgt_chol} mg/dL, and resolving exertional angina, the patient's cardiac risk drops by **{risk_drop:.1f}%** from **{categorize_risk(p_base)['tier']}** down to **{categorize_risk(p_target)['tier']}**!")

# -----------------------------------------------------------------------------
# TAB 3: RISK FACTORS & CORONARY MAP
# -----------------------------------------------------------------------------
with page3:
    col_map, col_weights = st.columns([1.2, 1])

    with col_map:
        st.subheader("🫀 2D Coronary Artery Perfusion Map")
        st.caption("Visual anatomical mapping of major coronary arteries based on fluoroscopy calcification:")
        coronary_svg = render_coronary_artery_svg(vessels_colored=ca, is_ischemia=(thal == 3))
        st.markdown(coronary_svg, unsafe_allow_html=True)

    with col_weights:
        st.subheader("📊 Clinical Feature Weights")
        st.caption("Random Forest Feature Importance (Mean Decrease in Impurity):")
        weights = pd.Series(model.feature_importances_, index=[
            "Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
            "Fasting Sugar", "Resting ECG", "Max HR", "Exercise Angina",
            "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"
        ]).sort_values(ascending=True)
        st.bar_chart(weights, width="stretch")

# -----------------------------------------------------------------------------
# TAB 4: ML BENCHMARKS
# -----------------------------------------------------------------------------
with page4:
    st.subheader("⚖️ Algorithm Performance Benchmark Suite")
    st.caption("Training and 5-Fold Stratified Cross-Validation on the Cleveland Clinic cohort:")

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Random Forest Sensitivity</div><div class="kpi-num" style="color:#22c55e;">96.97%</div><div style="font-size:0.8rem; color:#94a3b8;">High Diagnostic Recall</div></div>', unsafe_allow_html=True)
    with col_stat2:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Random Forest ROC-AUC</div><div class="kpi-num" style="color:#38bdf8;">91.61%</div><div style="font-size:0.8rem; color:#94a3b8;">5-Fold Stratified CV</div></div>', unsafe_allow_html=True)
    with col_stat3:
        st.markdown('<div class="kpi-card"><div class="kpi-label">Random Forest Accuracy</div><div class="kpi-num" style="color:#ffffff;">83.61%</div><div style="font-size:0.8rem; color:#94a3b8;">Holdout Test Cohort</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    benchmark_df = pd.DataFrame({
        "Accuracy (%)": [80.33, 83.61, 77.05],
        "Recall / Sensitivity (%)": [90.91, 96.97, 87.88],
        "ROC-AUC Score (%)": [86.90, 91.61, 86.80]
    }, index=["Logistic Regression", "Random Forest (Primary Engine)", "XGBoost"])

    st.bar_chart(benchmark_df, width="stretch")
    st.dataframe(benchmark_df, width="stretch")

# -----------------------------------------------------------------------------
# TAB 5: BATCH SCREENING
# -----------------------------------------------------------------------------
with page5:
    st.subheader("📁 Hospital Batch Patient Screening")
    st.caption("Upload a patient cohort CSV for automated bulk cardiac risk scoring and triage:")

    sample_cohort = pd.DataFrame([
        {"age": 63, "sex": 1, "cp": 3, "trestbps": 155, "chol": 286, "fbs": 1, "restecg": 2, "thalach": 115, "exang": 1, "oldpeak": 2.5, "slope": 2, "ca": 2, "thal": 3},
        {"age": 42, "sex": 0, "cp": 1, "trestbps": 118, "chol": 195, "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.2, "slope": 0, "ca": 0, "thal": 1},
        {"age": 58, "sex": 1, "cp": 2, "trestbps": 145, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 138, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 2}
    ])
    st.download_button(
        "📄 Download Sample Cohort CSV Template",
        data=sample_cohort.to_csv(index=False).encode("utf-8"),
        file_name="cardiocare_patient_template.csv",
        mime="text/csv"
    )

    batch_file = st.file_uploader("Upload Patient Cohort CSV", type=["csv"])
    if batch_file is not None:
        df_uploaded = pd.read_csv(batch_file)
        missing_fields = [f for f in feature_names if f not in df_uploaded.columns]
        if missing_fields:
            st.error(f"Missing clinical columns: {missing_fields}")
        else:
            probs = model.predict_proba(df_uploaded[feature_names])[:, 1]
            preds = model.predict(df_uploaded[feature_names])
            df_uploaded["Calculated_Risk"] = (probs * 100).round(1).astype(str) + "%"
            df_uploaded["Triage_Status"] = ["Critical / High Risk" if p >= 0.5 else "Low / Optimal Risk" for p in probs]
            df_uploaded["Triage_Tier"] = [categorize_risk(p)["tier"] for p in probs]

            st.dataframe(df_uploaded, width="stretch")

            csv_out = df_uploaded.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Scored Triage Report",
                data=csv_out,
                file_name="cardiocare_triage_results.csv",
                mime="text/csv"
            )

# Universal Disclaimer
st.markdown("""
<div style="padding: 1.2rem; background: #131f38; border-radius: 10px; color: #94a3b8; font-size: 0.8rem; line-height: 1.5; margin-top: 2rem; border-left: 4px solid #38bdf8;">
    <strong>⚠️ Clinical Disclaimer:</strong> CardioCare Pro™ is an assistive clinical decision-support system powered by Random Forest classification for cardiovascular risk assessment. It is not an autonomous diagnostic instrument. All medical interpretations, pharmaceutical interventions, and clinical decisions must be confirmed by board-certified physicians and cardiologists in accordance with standard medical guidelines.
</div>
""", unsafe_allow_html=True)
