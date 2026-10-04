import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="CardioCare Pro | Clinical Decision Support",
    page_icon="🫀",
    layout="wide"
)

# Custom High-Contrast Healthcare CSS
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #0b1329 !important;
        color: #ffffff !important;
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
        border-radius: 10px;
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

    /* Expander Styling Fix */
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
        background: linear-gradient(90deg, #131f38 0%, #1e293b 100%);
        padding: 20px 25px;
        border-radius: 10px;
        border-left: 5px solid #38bdf8;
        border: 1px solid #1e293b;
        margin-bottom: 22px;
    }
    .banner-title {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 0;
    }
    .banner-sub {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 4px;
    }

    /* KPI Cards */
    .kpi-card {
        background-color: #131f38;
        border: 1px solid #2d3e5e;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .kpi-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .kpi-num {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2px;
    }

    /* Alerts */
    .result-critical {
        background-color: rgba(239, 68, 68, 0.2);
        border: 1.5px solid #ef4444;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
        color: #fecaca;
    }
    .result-safe {
        background-color: rgba(34, 197, 94, 0.2);
        border: 1.5px solid #22c55e;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
        color: #bbf7d0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, scaler, features

try:
    model, scaler, feature_names = load_artifacts()
except Exception:
    st.error("Model files not found! Pehle 'python train_models.py' run karo.")
    st.stop()

# Header Banner
st.markdown("""
<div class="top-banner">
    <div class="banner-title">🏥 CardioCare Pro™ Clinical EMR Engine</div>
    <div class="banner-sub">Multi-Attribute Cardiac Risk Assessment & Clinical Decision Support System</div>
</div>
""", unsafe_allow_html=True)

# Sidebar: Top Logo Area
st.sidebar.markdown("""
<div class="sidebar-logo-box">
    <div class="logo-icon">🫀</div>
    <div>
        <div class="logo-text-title">CardioCare AI</div>
        <div class="logo-text-sub">Diagnostic Core v2.4</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📋 Patient Triage Inputs")

with st.sidebar.expander("👤 Vitals & Demographics", expanded=True):
    age = st.slider("Patient Age (Years)", 20, 95, 52)
    sex = st.selectbox("Biological Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 220, 125)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 212)

with st.sidebar.expander("💓 Cardiac Indicators", expanded=True):
    cp = st.selectbox("Chest Pain Presentation", [0, 1, 2, 3],
                      format_func=lambda x: {
                          0: "Typical Angina (0)",
                          1: "Atypical Angina (1)",
                          2: "Non-Anginal Discomfort (2)",
                          3: "Asymptomatic (3)"
                      }[x])
    thalach = st.slider("Peak Heart Rate (thalach)", 60, 230, 168)
    exang = st.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

with st.sidebar.expander("🔬 Lab & ECG Findings", expanded=True):
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "True (>120)" if x == 1 else "False (<=120)", horizontal=True)
    restecg = st.selectbox("Resting ECG Result", [0, 1, 2],
                           format_func=lambda x: {0: "Normal (0)", 1: "ST-T Wave Abnormality (1)", 2: "LV Hypertrophy (2)"}[x])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, 0.1)
    slope = st.selectbox("Peak Exercise ST Slope", [0, 1, 2],
                         format_func=lambda x: {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}[x])
    ca = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Profile (thal)", [1, 2, 3],
                        format_func=lambda x: {1: "Normal (1)", 2: "Fixed Defect (2)", 3: "Reversible Defect (3)"}[x])

input_data = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
]], columns=feature_names)

# Top KPI Dashboard Summary
k1, k2, k3, k4 = st.columns(4)

with k1:
    bp_tag = "Elevated" if trestbps >= 140 else "Optimal"
    bp_col = "#ef4444" if trestbps >= 140 else "#22c55e"
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Resting BP</div><div class="kpi-num">{trestbps} <span style="font-size:0.9rem">mmHg</span></div><div style="color:{bp_col}; font-weight:600; font-size:0.85rem;">{bp_tag}</div></div>', unsafe_allow_html=True)

with k2:
    chol_tag = "High Risk" if chol >= 240 else "Desirable"
    chol_col = "#ef4444" if chol >= 240 else "#22c55e"
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Cholesterol</div><div class="kpi-num">{chol} <span style="font-size:0.9rem">mg/dL</span></div><div style="color:{chol_col}; font-weight:600; font-size:0.85rem;">{chol_tag}</div></div>', unsafe_allow_html=True)

with k3:
    hr_tag = "Elevated" if thalach >= 170 else "Normal"
    hr_col = "#f59e0b" if thalach >= 170 else "#22c55e"
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Peak Heart Rate</div><div class="kpi-num">{thalach} <span style="font-size:0.9rem">bpm</span></div><div style="color:{hr_col}; font-weight:600; font-size:0.85rem;">{hr_tag}</div></div>', unsafe_allow_html=True)

with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Patient Demographics</div><div class="kpi-num">{age} <span style="font-size:0.9rem">Yrs</span></div><div style="color:#38bdf8; font-weight:600; font-size:0.85rem;">{"Male" if sex == 1 else "Female"}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Separate Page Tabs
page1, page2, page3 = st.tabs([
    "🩺 Page 1: Live Patient Triage & Risk",
    "📊 Page 2: Risk Factor Analytics & Graph",
    "⚖️ Page 3: ML Model Benchmarks"
])

# Page 1
with page1:
    col_table, col_pred = st.columns([1, 1.2])

    with col_table:
        st.subheader("📋 Recorded Clinical Manifest")
        st.dataframe(input_data.T.rename(columns={0: "Patient Reading"}), width="stretch")

    with col_pred:
        st.subheader("⚡ Diagnostic Risk Engine")
        
        if st.button("RUN CLINICAL INFERENCE", type="primary", width="stretch"):
            proba = model.predict_proba(input_data)[0][1]
            risk_pct = proba * 100

            st.metric("Predicted Disease Probability", f"{risk_pct:.1f}%")
            st.progress(float(proba))

            if risk_pct >= 50:
                st.markdown(f"""
                <div class="result-critical">
                    <h3 style="color:#ef4444; margin:0;">🚨 HIGH CARDIAC RISK DETECTED ({risk_pct:.1f}%)</h3>
                    <p style="margin: 8px 0 0 0; color:#fecaca;">
                        <strong>Action Protocol:</strong><br>
                        • Immediate cardiology consultation & Angiography recommended.<br>
                        • Continuous ECG monitoring advised for ST shifts.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <h3 style="color:#22c55e; margin:0;">✅ LOW / OPTIMAL RISK PROFILE ({risk_pct:.1f}%)</h3>
                    <p style="margin: 8px 0 0 0; color:#bbf7d0;">
                        <strong>Action Protocol:</strong><br>
                        • All indicators remain within physiological bounds.<br>
                        • Routine annual physical checkup advised.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Click the button above to run diagnostic prediction on the given vitals.")

# Page 2
with page2:
    st.subheader("📊 Patient Risk Factors Impact (Feature Weights)")
    st.caption("Cleveland Clinic model ke hisab se kaunse clinical features prediction ko sabse zyada affect kar rahe hain:")

    weights = pd.Series(model.feature_importances_, index=[
        "Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
        "Fasting Sugar", "Resting ECG", "Max HR", "Exercise Angina",
        "ST Depression", "ST Slope", "Major Vessels", "Thalassemia"
    ]).sort_values(ascending=True)

    st.bar_chart(weights, width="stretch")

# Page 3
with page3:
    st.subheader("⚖️ Algorithm Performance Benchmark Suite")
    st.caption("Training aur 5-fold cross-validation metrics across all three algorithms:")

    benchmark_df = pd.DataFrame({
        "Accuracy (%)": [80.33, 83.61, 77.05],
        "Recall / Sensitivity (%)": [90.91, 96.97, 87.88],
        "ROC-AUC Score (%)": [86.90, 91.61, 86.80]
    }, index=["Logistic Regression", "Random Forest (Selected)", "XGBoost"])

    st.bar_chart(benchmark_df, width="stretch")
    st.dataframe(benchmark_df, width="stretch")