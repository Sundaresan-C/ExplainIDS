@echo off
REM One-time setup: creates the Python environment and installs all libraries.
REM Needs Python 3.11 installed (https://www.python.org/downloads/ - tick "Add to PATH").
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv || goto :err
)
echo Installing libraries (this can take 10-15 minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
echo.
echo Setup complete. Double-click run_app.bat to start the web app.
pause
goto :eof
:err
echo Setup failed - check that Python 3.11 is installed and on PATH.
pause
