@echo off
setlocal enabledelayedexpansion

REM ============================================
REM   ROSE - AUTONOMOUS AI AGENT INSTALLER
REM   Production-quality Windows installer
REM   with GPU detection, model download,
REM   and comprehensive verification
REM ============================================

set "ROSE_VERSION=1.0.0"
set "VENV_DIR=.venv"
set "MODELS_DIR=models"
set "LOGS_DIR=logs"
set "CONFIGS_DIR=configs"
set "SCRIPTS_DIR=scripts"

REM Parse command line arguments
set "SKIP_MODELS=0"
set "CPU_ONLY=0"
set "CHECK_ONLY=0"
set "QUIET=0"

:parse_args
if "%~1"=="" goto done_args
if /i "%~1"=="--skip-models" set "SKIP_MODELS=1"
if /i "%~1"=="--cpu-only" set "CPU_ONLY=1"
if /i "%~1"=="--check-only" set "CHECK_ONLY=1"
if /i "%~1"=="--quiet" set "QUIET=1"
if /i "%~1"=="/?" goto show_help
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="/help" goto show_help
shift
goto parse_args

:show_help
echo.
echo Rose Installer v%ROSE_VERSION%
echo.
echo Usage: install.bat [OPTIONS]
echo.
echo Options:
echo   --skip-models   Skip model download (use if models already present)
echo   --cpu-only      Force CPU-only mode (no GPU acceleration)
echo   --check-only    Only check if installation is complete
echo   --quiet         Minimal output
echo   /? or --help    Show this help
echo.
echo Examples:
echo   install.bat                    Full installation
echo   install.bat --skip-models      Install without downloading models
echo   install.bat --cpu-only         Install for CPU-only system
echo   install.bat --check-only       Verify installation status
echo.
exit /b 0

:done_args

REM ============================================
REM Banner
REM ============================================
if "%QUIET%"=="0" (
    echo.
    echo ============================================
    echo   ROSE - AUTONOMOUS AI AGENT INSTALLER
    echo   v%ROSE_VERSION%
    echo ============================================
    echo.
)

REM Track overall success
set "INSTALL_SUCCESS=1"
set "STEP_COUNT=0"
set "STEP_TOTAL=10"

REM ============================================
REM STEP 1: Check System
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Checking system...

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [FAIL] Python not found
    echo.
    echo   Please install Python 3.10, 3.11, or 3.12 from:
    echo   https://www.python.org/downloads/
    echo.
    echo   Note: Python 3.10-3.12 have prebuilt wheels available.
    echo         Python 3.13+ may require compilation tools.
    echo.
    set "INSTALL_SUCCESS=0"
    goto error_exit
)

REM Get Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PY_VERSION=%%v"
for /f "tokens=1,2,3 delims=." %%a in ("%PY_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
    set "PY_PATCH=%%c"
)

REM Check Python version (need 3.10-3.12 for prebuilt wheels)
if %PY_MAJOR% lss 3 (
    echo   [FAIL] Python %PY_VERSION% detected - requires Python 3.10+
    set "INSTALL_SUCCESS=0"
    goto error_exit
)
if %PY_MINOR% lss 10 (
    echo   [FAIL] Python %PY_VERSION% detected - requires Python 3.10+
    set "INSTALL_SUCCESS=0"
    goto error_exit
)

REM Check 64-bit architecture
python -c "import struct; exit(0 if struct.calcsize('P') == 8 else 1)" 2>nul
if %errorlevel% neq 0 (
    echo   [FAIL] Python %PY_VERSION% is 32-bit - requires 64-bit Python
    set "INSTALL_SUCCESS=0"
    goto error_exit
)

echo   [OK] Python %PY_VERSION% (64-bit)

REM Check Python 3.13+ warning
if %PY_MINOR% gtr 12 (
    echo   [WARN] Python 3.13+ detected - prebuilt wheels may not be available
)

REM Check for NVIDIA GPU (unless CPU-only mode)
set "HAS_NVIDIA=0"
set "CUDA_VERSION="
set "GPU_NAME="
if "%CPU_ONLY%"=="0" (
    nvidia-smi >nul 2>&1
    if %errorlevel% equ 0 (
        set "HAS_NVIDIA=1"

        REM Get GPU name
        for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do set "GPU_NAME=%%g"

        REM Get CUDA version from nvidia-smi
        for /f "tokens=3 delims= " %%c in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set "CUDA_VERSION=%%c"

        if defined CUDA_VERSION (
            echo   [OK] NVIDIA GPU: %GPU_NAME% (CUDA %CUDA_VERSION%)
        ) else (
            echo   [OK] NVIDIA GPU: %GPU_NAME%
        )
    ) else (
        echo   [INFO] No NVIDIA GPU detected - using CPU mode
        set "CPU_ONLY=1"
    )
) else (
    echo   [INFO] CPU-only mode requested
)

REM Check for compiler (for fallback source build)
set "HAS_COMPILER=0"
where cl >nul 2>&1
if %errorlevel% equ 0 (
    set "HAS_COMPILER=1"
    echo   [OK] MSVC compiler found
) else (
    echo   [INFO] No MSVC compiler found (not required for prebuilt wheels)
)

REM Check for CMake
set "HAS_CMAKE=0"
where cmake >nul 2>&1
if %errorlevel% equ 0 (
    set "HAS_CMAKE=1"
    echo   [OK] CMake found
)

if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 2: Create Directories
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Creating directories...

if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if not exist "%LOGS_DIR%" mkdir "%LOGS_DIR%"
if not exist "%CONFIGS_DIR%" mkdir "%CONFIGS_DIR%"
if not exist "workspace" mkdir "workspace"
if not exist "workspace\media" mkdir "workspace\media"

echo   [OK] Directories ready
if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 3: Create Virtual Environment
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Creating virtual environment...

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo   [OK] Virtual environment already exists
) else (
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo   [FAIL] Failed to create virtual environment
        echo   Try: python -m venv --clear %VENV_DIR%
        set "INSTALL_SUCCESS=0"
        goto error_exit
    )
    echo   [OK] Virtual environment created
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
python -m pip install --upgrade pip --quiet --disable-pip-version-check 2>nul
if %errorlevel% equ 0 (
    echo   [OK] pip upgraded
) else (
    echo   [WARN] pip upgrade failed (continuing)
)

if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 4: Select Installation Strategy
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Selecting installation strategy...

REM Default to CPU
set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu"
set "INSTALL_MODE=CPU"

if "%HAS_NVIDIA%"=="1" (
    if defined CUDA_VERSION (
        REM Parse CUDA version
        for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
            set "CUDA_MAJOR=%%a"
            set "CUDA_MINOR=%%b"
        )

        REM Select wheel based on CUDA version
        REM Official prebuilt wheels: cu118, cu121, cu122, cu123, cu124, cu125, cu130, cu132
        if !CUDA_MAJOR! geq 13 (
            if !CUDA_MINOR! geq 2 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu132"
                set "INSTALL_MODE=CUDA 13.2"
            ) else (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu130"
                set "INSTALL_MODE=CUDA 13.0"
            )
        ) else if !CUDA_MAJOR! equ 12 (
            if !CUDA_MINOR! geq 5 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu125"
                set "INSTALL_MODE=CUDA 12.5"
            ) else if !CUDA_MINOR! geq 4 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu124"
                set "INSTALL_MODE=CUDA 12.4"
            ) else if !CUDA_MINOR! geq 3 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu123"
                set "INSTALL_MODE=CUDA 12.3"
            ) else if !CUDA_MINOR! geq 2 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu122"
                set "INSTALL_MODE=CUDA 12.2"
            ) else (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu121"
                set "INSTALL_MODE=CUDA 12.1"
            )
        ) else if !CUDA_MAJOR! equ 11 (
            if !CUDA_MINOR! geq 8 (
                set "WHEEL_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu118"
                set "INSTALL_MODE=CUDA 11.8"
            ) else (
                echo   [WARN] CUDA !CUDA_VERSION! too old - falling back to CPU
                set "INSTALL_MODE=CPU (CUDA too old)"
            )
        ) else (
            echo   [WARN] CUDA !CUDA_VERSION! too old - falling back to CPU
            set "INSTALL_MODE=CPU (CUDA too old)"
        )
    ) else (
        echo   [WARN] NVIDIA GPU detected but CUDA version unknown - using CPU
        set "INSTALL_MODE=CPU (CUDA unknown)"
    )
)

echo   [OK] Strategy: %INSTALL_MODE%
if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 5: Install Runtime Dependencies
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Installing runtime dependencies...

REM Check if llama-cpp-python is already installed
python -c "import llama_cpp; print(llama_cpp.__version__)" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('python -c "import llama_cpp; print(llama_cpp.__version__)" 2^>nul') do set "LLAMA_VERSION=%%v"
    echo   [OK] llama-cpp-python !LLAMA_VERSION! already installed
) else (
    echo   Installing llama-cpp-python...

    REM Try prebuilt wheel first (NO source compilation)
    pip install llama-cpp-python>=0.3.0 ^
        --extra-index-url %WHEEL_INDEX% ^
        --only-binary=llama-cpp-python ^
        --quiet 2>nul

    if %errorlevel% neq 0 (
        echo   [FAIL] Prebuilt wheel not available for this configuration
        echo.
        echo   This can happen when:
        echo   - Python version is not supported by prebuilt wheels
        echo   - CUDA version is not supported
        echo   - No compatible wheel exists for your GPU
        echo.
        echo   Options:
        echo   1. Use Python 3.10, 3.11, or 3.12 (recommended)
        echo   2. Use --cpu-only flag
        echo   3. Install Visual Studio Build Tools for source compilation
        echo   4. Check https://github.com/abetlen/llama-cpp-python for updates
        echo.
        set "INSTALL_SUCCESS=0"
        goto error_exit
    )

    python -c "import llama_cpp; print(llama_cpp.__version__)" >nul 2>&1
    if %errorlevel% equ 0 (
        for /f "tokens=*" %%v in ('python -c "import llama_cpp; print(llama_cpp.__version__)" 2^>nul') do set "LLAMA_VERSION=%%v"
        echo   [OK] llama-cpp-python !LLAMA_VERSION! installed
    ) else (
        echo   [FAIL] llama-cpp-python installation failed
        set "INSTALL_SUCCESS=0"
        goto error_exit
    )
)

REM Install other runtime dependencies
echo   Installing other dependencies...
pip install -r requirements-runtime.txt --quiet 2>nul
if %errorlevel% equ 0 (
    echo   [OK] Runtime dependencies installed
) else (
    echo   [WARN] Some dependencies failed (continuing)
)

if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 6: Install Model Downloader Dependencies
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Preparing model downloader...

python -c "import huggingface_hub" >nul 2>&1
if %errorlevel% neq 0 (
    pip install "huggingface_hub>=0.20.0" --quiet 2>nul
)

python -c "import requests" >nul 2>&1
if %errorlevel% neq 0 (
    pip install requests --quiet 2>nul
)

python -c "import tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    pip install tqdm --quiet 2>nul
)

echo   [OK] Model downloader ready
if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 7: Download Models
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Checking models...

if "%SKIP_MODELS%"=="1" (
    echo   [SKIP] Model download skipped (--skip-models)
    REM Verify models exist
    if exist "%MODELS_DIR%\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" (
        echo   [OK] Main model found
    ) else (
        echo   [WARN] Main model not found - Rose will not work without it
    )
    if exist "%MODELS_DIR%\mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf" (
        echo   [OK] Vision projector found
    ) else (
        echo   [WARN] Vision projector not found - vision features disabled
    )
) else (
    REM Check if models already exist
    set "MODELS_READY=1"
    if not exist "%MODELS_DIR%\Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" set "MODELS_READY=0"
    if not exist "%MODELS_DIR%\mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf" set "MODELS_READY=0"

    if "%MODELS_READY%"=="1" (
        echo   [OK] Models already present
    ) else (
        echo   Downloading models (this may take several minutes)...
        echo.
        python "%SCRIPTS_DIR%\download_models.py" --model-dir "%MODELS_DIR%"
        if %errorlevel% neq 0 (
            echo.
            echo   [WARN] Model download had issues
            echo   You can manually download from:
            echo   https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF
            echo.
        )
    )
)

if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 8: Configure Rose
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Configuring Rose...

REM Create .env from .env.example if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo   [OK] Created .env from template
    ) else (
        echo   [WARN] No .env.example found - using defaults
    )
) else (
    echo   [OK] .env already exists
)

REM Ensure CUDA DLLs are in PATH for llama-cpp-python
if "%HAS_NVIDIA%"=="1" (
    set "CUDA_BIN="
    REM Try common CUDA installation paths
    for %%d in (
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    ) do (
        if exist "%%~d" (
            set "CUDA_BIN=%%~d"
        )
    )
    if defined CUDA_BIN (
        set "PATH=!CUDA_BIN!;!PATH!"
        echo   [OK] CUDA binaries added to PATH
    )
)

echo   [OK] Configuration complete
if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 9: Verify Installation
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Verifying installation...

REM Quick verification - import key modules
python -c "import llama_cpp; import pydantic; import yaml; import dotenv" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Core modules importable
) else (
    echo   [FAIL] Core module import failed
    set "INSTALL_SUCCESS=0"
    goto error_exit
)

REM Check CUDA backend
python -c "import llama_cpp; from pathlib import Path; lib_dir=Path(llama_cpp.__file__).parent / 'lib'; cuda_dll=lib_dir / 'ggml-cuda.dll'; exit(0 if cuda_dll.exists() else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] CUDA backend available
) else (
    if "%HAS_NVIDIA%"=="1" (
        echo   [WARN] CUDA backend not found - GPU acceleration may not work
    ) else (
        echo   [OK] CPU-only mode
    )
)

if "%QUIET%"=="0" echo.

REM ============================================
REM STEP 10: Final Health Check
REM ============================================
set /a STEP_COUNT+=1
if "%QUIET%"=="0" echo [%STEP_COUNT%/%STEP_TOTAL%] Running health check...

python "%SCRIPTS_DIR%\health_check.py" >nul 2>&1
if %errorlevel% equ 0 (
    echo   [OK] Health check passed
) else (
    echo   [WARN] Health check reported issues (Rose may still work)
)

if "%QUIET%"=="0" echo.

REM ============================================
REM DONE
REM ============================================
if "%INSTALL_SUCCESS%"=="1" (
    echo ============================================
    echo   ROSE INSTALLATION COMPLETE
    echo ============================================
    echo.
    echo   Python:    %PY_VERSION%
    echo   Mode:      %INSTALL_MODE%
    echo   Models:    %MODELS_DIR%/
    echo.
    echo   To run Rose:
    echo     %VENV_DIR%\Scripts\activate
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
) else (
    echo ============================================
    echo   ROSE INSTALLATION INCOMPLETE
    echo ============================================
    echo.
    echo   Some steps failed. Please review the errors above.
    echo.
)

exit /b %INSTALL_SUCCESS%

:error_exit
echo.
echo ============================================
echo   ROSE INSTALLATION FAILED
echo ============================================
echo.
echo   Please fix the issues above and try again.
echo.
exit /b 1
