@echo off
REM ============================================
REM   Rose - Autonomous AI Agent Installer
REM   Production-grade installer with
REM   system detection and fallback strategies
REM ============================================
setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Rose - Autonomous AI Agent Installer
echo ============================================
echo.

REM ============================================
REM STEP 1: Check Python
REM ============================================
echo [Step 1/9] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found.
    echo.
    echo   Please install Python 3.10, 3.11, or 3.12 from:
    echo   https://www.python.org/downloads/
    echo.
    echo   Note: Python 3.13+ may require compilation tools.
    echo         Python 3.10-3.12 have prebuilt packages available.
    echo.
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo   Found Python %PY_VERSION%

REM Check Python version (need 3.10-3.12 for prebuilt wheels)
for /f "tokens=1,2,3 delims=." %%a in ("%PY_VERSION%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
    set PY_PATCH=%%c
)

if %PY_MAJOR% lss 3 (
    echo   ERROR: Python 3.10 or later required.
    pause
    exit /b 1
)
if %PY_MINOR% lss 10 (
    echo   ERROR: Python 3.10 or later required.
    pause
    exit /b 1
)

if %PY_MINOR% gtr 12 (
    echo   WARNING: Python 3.13+ detected.
    echo   Prebuilt wheels may not be available.
    echo   Will attempt source compilation as fallback.
    echo.
)

REM ============================================
REM STEP 2: Detect System Capabilities
REM ============================================
echo [Step 2/9] Detecting system capabilities...

REM Check for NVIDIA GPU
set HAS_NVIDIA=0
set CUDA_VERSION=
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_NVIDIA=1
    echo   NVIDIA GPU detected.

    REM Try to get CUDA version from nvidia-smi
    for /f "tokens=3 delims= " %%c in ('nvidia-smi ^| findstr /C:"CUDA Version"') do (
        set CUDA_VERSION=%%c
    )
    if defined CUDA_VERSION (
        echo   CUDA Version: !CUDA_VERSION!
    ) else (
        echo   CUDA Version: Unknown
    )
) else (
    echo   No NVIDIA GPU detected. Using CPU mode.
)

REM Check for compiler (Visual Studio Build Tools)
set HAS_COMPILER=0
where cl >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_COMPILER=1
    echo   MSVC compiler found.
) else (
    echo   No MSVC compiler found.
)

REM Check for CMake
set HAS_CMAKE=0
where cmake >nul 2>&1
if %errorlevel% equ 0 (
    set HAS_CMAKE=1
    echo   CMake found.
) else (
    echo   No CMake found.
)

REM Check available disk space (need ~5GB for model + deps)
echo   Checking disk space...

echo.

REM ============================================
REM STEP 3: Select Installation Strategy
REM ============================================
echo [Step 3/9] Selecting installation strategy...

if %HAS_NVIDIA% equ 1 (
    if defined CUDA_VERSION (
        REM Parse CUDA version to select wheel
        for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
            set CUDA_MAJOR=%%a
            set CUDA_MINOR=%%b
        )

        REM Select wheel based on CUDA version
        if %CUDA_MAJOR% geq 13 (
            if %CUDA_MINOR% geq 2 (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu132
                echo   Strategy: CUDA 13.2+ GPU acceleration
            ) else (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu130
                echo   Strategy: CUDA 13.0 GPU acceleration
            )
        ) else if %CUDA_MAJOR% geq 12 (
            if %CUDA_MINOR% geq 5 (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu125
                echo   Strategy: CUDA 12.5+ GPU acceleration
            ) else if %CUDA_MINOR% geq 4 (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu124
                echo   Strategy: CUDA 12.4 GPU acceleration
            ) else if %CUDA_MINOR% geq 3 (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu123
                echo   Strategy: CUDA 12.3 GPU acceleration
            ) else if %CUDA_MINOR% geq 2 (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu122
                echo   Strategy: CUDA 12.2 GPU acceleration
            ) else (
                set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu121
                echo   Strategy: CUDA 12.1 GPU acceleration
            )
        ) else if %CUDA_MAJOR% equ 11 (
            set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu118
            echo   Strategy: CUDA 11.8 GPU acceleration
        ) else (
            set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu
            echo   Strategy: CPU mode (CUDA version too old)
        )
    ) else (
        set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu
        echo   Strategy: CPU mode (CUDA version unknown)
    )
) else (
    set WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu
    echo   Strategy: CPU mode (no NVIDIA GPU)
)

echo.

REM ============================================
REM STEP 4: Create Virtual Environment
REM ============================================
echo [Step 4/9] Creating virtual environment...

if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo   ERROR: Failed to create virtual environment.
        echo   Try: python -m venv --clear venv
        pause
        exit /b 1
    )
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists.
)

REM ============================================
REM STEP 5: Upgrade pip
REM ============================================
echo [Step 5/9] Upgrading pip...
call venv\Scripts\activate.bat
python -m pip upgrade pip --quiet 2>nul
if %errorlevel% neq 0 (
    echo   WARNING: pip upgrade failed, continuing with existing version.
)

REM ============================================
REM STEP 6: Install Core Dependencies
REM ============================================
echo [Step 6/9] Installing core dependencies...
echo   Using wheel index: %WHEEL_INDEX%
echo.

REM Try prebuilt wheel first
pip install llama-cpp-python>=0.3.0 ^
    --extra-index-url %WHEEL_INDEX% ^
    --only-binary=llama-cpp-python ^
    --quiet 2>nul

if %errorlevel% neq 0 (
    echo   Prebuilt wheel not available for this configuration.
    echo   Attempting source compilation...
    echo.

    REM Check if compiler is available
    if %HAS_COMPILER% equ 0 (
        echo   ERROR: No compiler found for source compilation.
        echo.
        echo   Options:
        echo   1. Install Visual Studio Build Tools:
        echo      https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo.
        echo   2. Use Python 3.10, 3.11, or 3.12 with prebuilt wheels.
        echo.
        echo   3. Install w64devkit and set CMAKE_ARGS:
        echo      $env:CMAKE_GENERATOR = "MinGW Makefiles"
        echo      $env:CMAKE_ARGS = "-DCMAKE_C_COMPILER=C:/w64devkit/bin/gcc.exe"
        echo.
        pause
        exit /b 1
    )

    REM Try source compilation with compiler
    pip install llama-cpp-python>=0.3.0 --quiet
    if %errorlevel% neq 0 (
        echo.
        echo   ERROR: Failed to install llama-cpp-python.
        echo.
        echo   The prebuilt wheel was not available and source
        echo   compilation failed. Please check the error above.
        echo.
        pause
        exit /b 1
    )
)

echo   llama-cpp-python installed successfully.

REM Install other core dependencies
pip install ^
    "Pillow>=10.0.0" ^
    "numpy>=1.24.0" ^
    "python-dotenv>=1.0.0" ^
    "PyYAML>=6.0" ^
    "tqdm>=4.65.0" ^
    "rich>=13.0.0" ^
    "pydantic>=2.0.0" ^
    "typing-extensions>=4.5.0" ^
    "diskcache>=5.6.0" ^
    "Jinja2>=3.1.0" ^
    "markdown-it-py>=3.0.0" ^
    "packaging>=23.0" ^
    "pathspec>=0.11.0" ^
    --quiet

if %errorlevel% neq 0 (
    echo   ERROR: Failed to install core dependencies.
    pause
    exit /b 1
)

echo   Core dependencies installed.

REM ============================================
REM STEP 7: Install Optional Features
REM ============================================
echo [Step 7/9] Installing optional features...
echo.
echo   Select installation type:
echo   [1] Core only (LLM + basic tools)
echo   [2] With UI (PySide6 desktop app)
echo   [3] Full (UI + browser + vision)
echo   [4] Development (all + testing)
echo.
set /p INSTALL_TYPE="Enter choice (1-4): "

if "%INSTALL_TYPE%"=="1" (
    echo       Core only installed.
) else if "%INSTALL_TYPE%"=="2" (
    echo       Installing UI support...
    pip install "PySide6>=6.5.0" --quiet
) else if "%INSTALL_TYPE%"=="3" (
    echo       Installing full features...
    pip install "PySide6>=6.5.0" --quiet
    pip install "playwright>=1.40.0" --quiet
    pip install "opencv-python>=4.8.0" --quiet
    echo       Installing Playwright browsers...
    python -m playwright install chromium
) else if "%INSTALL_TYPE%"=="4" (
    echo       Installing development tools...
    pip install "PySide6>=6.5.0" --quiet
    pip install "playwright>=1.40.0" --quiet
    pip install "opencv-python>=4.8.0" --quiet
    pip install pytest pytest-cov pytest-asyncio --quiet
    echo       Installing Playwright browsers...
    python -m playwright install chromium
) else (
    echo       Invalid choice, installing core only...
)

REM ============================================
REM STEP 8: Create Directories
REM ============================================
echo [Step 8/9] Setting up directories...

if not exist "config" mkdir config
if not exist "models" mkdir models
if not exist "workspace" mkdir workspace
if not exist "workspace\media" mkdir workspace\media
if not exist "logs" mkdir logs

echo   Directories created.

REM ============================================
REM STEP 9: Verify Installation
REM ============================================
echo [Step 9/9] Verifying installation...

python -c "import llama_cpp; print('  llama-cpp-python: OK')" 2>nul
if %errorlevel% neq 0 (
    echo   ERROR: llama-cpp-python import failed.
    pause
    exit /b 1
)

python -c "import PIL; print('  Pillow: OK')" 2>nul
if %errorlevel% neq 0 (
    echo   WARNING: Pillow not available. Vision features may be limited.
)

python -c "import numpy; print('  numpy: OK')" 2>nul
if %errorlevel% neq 0 (
    echo   WARNING: numpy not available.
)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo   Installation type: %INSTALL_TYPE%
echo   Python: %PY_VERSION%
if %HAS_NVIDIA% equ 1 (
    echo   GPU: NVIDIA (accelerated)
) else (
    echo   GPU: CPU mode
)
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
echo   Models directory: models/
echo   Place .gguf model files there.
echo.
pause
