# 🫀 CardioCare Pro™ — Clinical Decision Support & Heart Disease Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Diagnostic Sensitivity](https://img.shields.io/badge/Diagnostic%20Sensitivity-96.97%25-22c55e.svg)](#)
[![Validation](https://img.shields.io/badge/Validation-5--Fold%20Stratified%20CV-38bdf8.svg)](#)

A clinical decision-support and cardiovascular risk stratification platform built with **Random Forest Classification**, validated via **5-Fold Stratified Cross-Validation** on the benchmark **UCI Cleveland Clinic** cohort.

---

## 🌟 Key Features & Clinical Innovations

- 🌲 **Single Core Machine Learning Engine (Random Forest)**:
  - Tuned 100-estimator ensemble prioritizing **Diagnostic Recall (Sensitivity: 96.97%)** to eliminate false negatives in clinical triage.
- 🎛️ **Plotly Radial Speedometer Risk Gauge**:
  - Live animated curved gauge with standard medical color zones: Low (<25%), Moderate (25–50%), High (50–75%), and Critical (≥75%).
- 🫀 **Interactive 2D Coronary Artery Tree Visualizer**:
  - Anatomical heart silhouette mapping the **LAD** (Left Anterior Descending), **LCx** (Left Circumflex), and **RCA** (Right Coronary Artery), dynamically highlighting calcification based on fluoroscopy vessels (`ca`) and thallium scintigraphy (`thal`).
- ⚖️ **Standard 10-Year ASCVD vs. Random Forest AI Comparison**:
  - Direct side-by-side comparison illustrating how machine learning detects occult ischemic signals that static Framingham equations miss.
- 🔮 **Interactive 'What-If' Risk Reduction Simulator**:
  - Patients and clinicians can simulate the impact of lifestyle modifications (reducing BP, optimizing lipids, resolving angina) to see risk drop in real time (e.g., from 84% down to 18%).
- 💓 **Biological Heart Age vs. Chronological Age**:
  - Translates complex probabilities into an intuitive metric: *"Your chronological age is 52, but your cardiovascular risk profile corresponds to a Heart Age of 68 (+16 yrs older)."*
- 📄 **1-Click Certified Consultation PDF Report (with QR Code)**:
  - Generates an authentic hospital consultation summary with patient ID, vitals table, AI diagnosis, doctor signature block, and a digital verification QR code.
- 📁 **Hospital Cohort Batch Screening**:
  - Bulk CSV upload and instant export of scored triage reports for multi-patient clinics.

---

## 📊 Cross-Validation & Model Benchmarks

Trained and cross-validated on the Cleveland Clinic cohort using 5-Fold Stratified Cross-Validation:

| Algorithm | Diagnostic Sensitivity (Recall) | ROC-AUC Score | Holdout Test Accuracy |
| :--- | :---: | :---: | :---: |
| **Random Forest (Primary Engine)** | **96.97%** | **91.61%** | **83.61%** |
| **Logistic Regression** | 90.91% | 86.90% | 80.33% |
| **XGBoost Classifier** | 87.88% | 86.80% | 77.05% |

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Vishal-kumar-1903/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
```

### 2. Retrain Models (Optional)
```bash
python train_models.py
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🌐 1-Click Free Publishing to Streamlit Cloud

1. Push your updated code to GitHub:
   ```bash
   git add .
   git commit -m "Upgrade CardioCare Pro with 2D Coronary Map, What-If Simulator & PDF Report"
   git push origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New App"** → Select `Vishal-kumar-1903/heart-disease-predictor` → Set Main file path to `app.py`.
4. Click **"Deploy"**! Your website is live 24/7 with a public URL!

---

## ⚠️ Clinical Disclaimer
CardioCare Pro™ is an assistive machine learning decision-support platform designed for patient screening and risk stratification. It is not an autonomous diagnostic instrument. All medical interpretations, pharmaceutical interventions, and clinical decisions must be confirmed by board-certified physicians and cardiologists in accordance with standard medical guidelines.
