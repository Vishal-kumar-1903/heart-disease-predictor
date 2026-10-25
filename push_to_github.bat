@echo off
title Push Updates to GitHub
echo ==============================================================
echo    Pushing Upgraded CardioCare Pro to your GitHub Repo
echo    Repo: https://github.com/Vishal-kumar-1903/heart-disease-predictor
echo ==============================================================
cd /d "%~dp0"

git add .
git commit -m "Upgrade CardioCare Pro: Add 2D Coronary Map, What-If Simulator, Speedometer Gauge, and PDF Report"
git push origin main

echo.
echo ==============================================================
echo    Updates pushed successfully to GitHub!
echo ==============================================================
pause
