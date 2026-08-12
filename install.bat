@echo off
REM Rose - Windows Installer
REM Phase 8 - Packaging and Installation

echo ============================================
echo   Rose - Autonomous AI Agent Installer
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.10+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo       Virtual environment created.
) else (
    echo       Virtual environment already exists.
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing core dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)

echo [4/4] Installing optional dependencies...
echo.
echo   Select installation type:
echo   [1] Core only (LLM + basic tools)
echo   [2] With UI (PySide6 desktop app)
echo   [3] Full (UI + browser + vision)
echo   [4] Development (all + testing)
echo.
set /p INSTALL_TYPE="Enter choice (1-4): "

if "%INSTALL_TYPE%"=="1" (
    echo       Installing core only...
) else if "%INSTALL_TYPE%"=="2" (
    echo       Installing with UI...
    pip install "PySide6>=6.5.0" --quiet
) else if "%INSTALL_TYPE%"=="3" (
    echo       Installing full...
    pip install "PySide6>=6.5.0" --quiet
    pip install "playwright>=1.40.0" --quiet
    pip install "opencv-python>=4.8.0" --quiet
    echo       Installing Playwright browsers...
    python -m playwright install chromium
) else if "%INSTALL_TYPE%"=="4" (
    echo       Installing development...
    pip install "PySide6>=6.5.0" --quiet
    pip install "playwright>=1.40.0" --quiet
    pip install "opencv-python>=4.8.0" --quiet
    pip install pytest pytest-cov --quiet
    echo       Installing Playwright browsers...
    python -m playwright install chromium
) else (
    echo       Invalid choice, installing core only...
)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo   To run Rose:
echo     venv\Scripts\activate
echo     python run.py
echo.
echo   To run with web UI:
echo     python run.py --web
echo.
echo   To check system status:
echo     python run.py status
echo.
echo   For more options:
echo     python run.py --help
echo.
pause
