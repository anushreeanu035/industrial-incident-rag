@echo off
echo =====================================================================
echo  Industrial Incident Investigator - Full Stack Launcher
echo =====================================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Using project virtual environment: .venv
    ".venv\Scripts\python.exe" app.py all
) else (
    echo Using system Python
    python app.py all
)

pause
