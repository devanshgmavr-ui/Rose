@echo off
REM ============================================
REM   Rose - Autonomous AI Agent Installer
REM   Single merged installer with smart
REM   dependency checking and GPU detection
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

REM Activate venv
call venv\Scripts\activate.bat

REM ============================================
REM STEP 5: Check Installed Packages
REM ============================================
echo [Step 5/9] Checking installed packages...

REM Use PowerShell to get installed packages as JSON
set "INSTALLED_FILE=%TEMP%\rose_installed.json"
python -m pip list --format=json > "%INSTALLED_FILE%" 2>nul

REM Check each critical package
set "MISSING_PACKAGES="

REM Check llama-cpp-python
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'llama-cpp-python' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! llama-cpp-python"
    echo   llama-cpp-python: MISSING
) else (
    echo   llama-cpp-python: installed
)

REM Check Pillow
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'pillow' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! Pillow"
    echo   Pillow: MISSING
) else (
    echo   Pillow: installed
)

REM Check numpy
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'numpy' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! numpy"
    echo   numpy: MISSING
) else (
    echo   numpy: installed
)

REM Check pydantic
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'pydantic' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! pydantic"
    echo   pydantic: MISSING
) else (
    echo   pydantic: installed
)

REM Check PyYAML
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'pyyaml' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! PyYAML"
    echo   PyYAML: MISSING
) else (
    echo   PyYAML: installed
)

REM Check rich
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'rich' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    set "MISSING_PACKAGES=!MISSING_PACKAGES! rich"
    echo   rich: MISSING
) else (
    echo   rich: installed
)

echo.

REM ============================================
REM STEP 6: Install Missing Dependencies
REM ============================================
echo [Step 6/9] Installing missing dependencies...

REM Always try to upgrade pip first
python -m pip install upgrade pip --quiet 2>nul

REM Install llama-cpp-python only if missing
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'llama-cpp-python' in pkgs else 1)" 2>nul
if %errorlevel% neq 0 (
    echo   Installing llama-cpp-python (wheel: %WHEEL_INDEX%...

    pip install llama-cpp-python>=0.3.0 ^
        --extra-index-url %WHEEL_INDEX% ^
        --only-binary=llama-cpp-python ^
        --quiet 2>nul

    if %errorlevel% neq 0 (
        echo   Prebuilt wheel not available, attempting source compilation...

        if %HAS_COMPILER% equ 0 (
            echo   ERROR: No compiler found for source compilation.
            echo.
            echo   Options:
            echo   1. Install Visual Studio Build Tools
            echo   2. Use Python 3.10, 3.11, or 3.12 with prebuilt wheels
            echo.
            pause
            exit /b 1
        )

        pip install llama-cpp-python>=0.3.0 --quiet
        if %errorlevel% neq 0 (
            echo   ERROR: Failed to install llama-cpp-python.
            pause
            exit /b 1
        )
    )
    echo   llama-cpp-python installed.
) else (
    echo   llama-cpp-python: already installed, skipping.
)

REM Install other missing core dependencies
set "CORE_DEPS=Pillow>=10.0.0 numpy>=1.24.0 python-dotenv>=1.0.0 PyYAML>=6.0 tqdm>=4.65.0 rich>=13.0.0 pydantic>=2.0.0 typing-extensions>=4.5.0 diskcache>=5.6.0 Jinja2>=3.1.0 markdown-it-py>=3.0.0 packaging>=23.0 pathspec>=0.11.0"

REM Check each and install only missing ones
python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; missing=[p for p in sys.argv[1:] if p.split('>=')[0].split('==')[0].lower() not in pkgs]; print(' '.join(missing) if missing else 'NONE')" %CORE_DEPS% > "%TEMP%\rose_missing.txt" 2>nul

set /p MISSING_CORE=<"%TEMP%\rose_missing.txt"
if "%MISSING_CORE%"=="NONE" (
    echo   Core dependencies: all installed.
) else (
    echo   Installing: %MISSING_CORE%
    pip install %MISSING_CORE% --quiet
    if %errorlevel% neq 0 (
        echo   WARNING: Some core dependencies failed.
    ) else (
        echo   Core dependencies installed.
    )
)

echo.

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
    echo   Core only selected.
) else if "%INSTALL_TYPE%"=="2" (
    python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if 'pyside6' in pkgs else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   Installing PySide6...
        pip install "PySide6>=6.5.0" --quiet
    ) else (
        echo   PySide6: already installed.
    )
) else if "%INSTALL_TYPE%"=="3" (
    python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if all(p in pkgs for p in ['pyside6','playwright','opencv-python']) else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   Installing full features...
        pip install "PySide6>=6.5.0" --quiet
        pip install "playwright>=1.40.0" --quiet
        pip install "opencv-python>=4.8.0" --quiet
        echo   Installing Playwright browsers...
        python -m playwright install chromium
    ) else (
        echo   Full features: already installed.
    )
) else if "%INSTALL_TYPE%"=="4" (
    python -c "import json,sys; pkgs={p['name'].lower():p['version'] for p in json.load(open(r'%INSTALLED_FILE%'))}; sys.exit(0 if all(p in pkgs for p in ['pyside6','playwright','opencv-python','pytest']) else 1)" 2>nul
    if %errorlevel% neq 0 (
        echo   Installing development tools...
        pip install "PySide6>=6.5.0" --quiet
        pip install "playwright>=1.40.0" --quiet
        pip install "opencv-python>=4.8.0" --quiet
        pip install pytest pytest-cov pytest-asyncio --quiet
        echo   Installing Playwright browsers...
        python -m playwright install chromium
    ) else (
        echo   Development tools: already installed.
    )
) else (
    echo   Invalid choice, core only selected.
)

echo.

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
REM STEP 9: Check for Model Files
REM ============================================
echo [Step 9/9] Checking model files...

set "MODEL_FOUND=0"
for %%f in (models\*.gguf) do (
    set "MODEL_FOUND=1"
    echo   Found model: %%f
)

if %MODEL_FOUND% equ 0 (
    echo.
    echo   No .gguf model files found in models/ directory.
    echo.
    echo   Please download the Qwen2.5-VL-7B-Instruct GGUF files:
    echo     1. Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf (main model, ~4.7GB)
    echo     2. mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf (vision projector, ~1.3GB)
    echo.
    echo   Place both files in the models/ folder.
    echo.
    echo   Download from: https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF
)

echo.

REM ============================================
REM Clean up temp files
REM ============================================
del "%INSTALLED_FILE%" 2>nul
del "%TEMP%\rose_missing.txt" 2>nul

REM ============================================
REM DONE
REM ============================================
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
