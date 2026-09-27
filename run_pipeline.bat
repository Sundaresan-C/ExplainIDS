@echo off
REM Full pipeline on the real CIC IoT-DIAD 2024 CSVs placed in data\raw\
cd /d "%~dp0"
".venv\Scripts\python.exe" src\data_loader.py --per-class 25000 || goto :err
".venv\Scripts\python.exe" src\train.py || goto :err
".venv\Scripts\python.exe" docs\make_architecture.py
".venv\Scripts\python.exe" docs\build_report.py
".venv\Scripts\python.exe" docs\build_slides.py
echo Done. Now run run_app.bat
goto :eof
:err
echo Pipeline failed - see messages above.
