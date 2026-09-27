@echo off
REM Launch the ExplainIDS web app at http://localhost:8501
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
