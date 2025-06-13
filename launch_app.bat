@echo off
title Water Leak Detection App
echo Launching the Streamlit app...
cd /d %~dp0
streamlit run app.py
pause
