@echo off
REM KANAME [要] - Windows environment bootstrapper
REM Requires Python 3.10, matching requirements.txt and the CI workflow.

cd /d "%~dp0"

echo.
echo ========================================
echo   KANAME - Environment Setup
echo ========================================
echo.

set "PYTHON_CMD="

REM Prefer the Windows Python Launcher and explicitly select Python 3.10.
where py >nul 2>&1
if not errorlevel 1 (
    py -3.10 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=py -3.10"
)

REM Fall back to python on PATH, but still enforce Python 3.10.
if not defined PYTHON_CMD (
    where python >nul 2>&1
    if errorlevel 1 goto :python_missing
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1
    if errorlevel 1 goto :python_version
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :setup_failed
) else (
    echo [1/4] Existing .venv found. Reusing it...
)

echo [2/4] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :setup_failed

echo [3/4] Updating pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :setup_failed

echo [4/4] Installing project dependencies...
if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found in the project root.
    goto :setup_failed
)
python -m pip install -r requirements.txt
if errorlevel 1 goto :setup_failed

if not exist ".env" if exist ".env.example" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example. Add your own API key if needed.
)

echo.
echo ========================================
echo   Setup completed successfully!
echo ========================================
echo The .venv environment is active in this window.
echo Start the dashboard with:
echo   streamlit run frontend\streamlit_app\app.py
echo Start the API with:
echo   uvicorn backend.main:app --host 127.0.0.1 --port 8000
echo.
echo Type 'deactivate' to leave the virtual environment.
echo This window will stay open so the environment remains active.
echo.
cmd /k
exit /b 0

:python_missing
echo ERROR: Python 3.10 was not found. Install Python 3.10 and enable the Python Launcher or add python.exe to PATH.
goto :failed

:python_version
echo ERROR: The 'python' command is not Python 3.10. KANAME requires Python 3.10.
goto :failed

:setup_failed
echo.
echo ERROR: Setup failed. Review the message above and try again.
goto :failed

:failed
pause
exit /b 1
