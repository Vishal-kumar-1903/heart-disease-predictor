@echo off
title CardioCare Pro - Heart Disease Predictor
echo ==============================================================
echo    Starting CardioCare Pro Web Application...
echo ==============================================================
cd /d "%~dp0"
start "" "http://localhost:8501"
"C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
pause
