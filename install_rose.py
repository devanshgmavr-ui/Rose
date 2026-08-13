#!/usr/bin/env python3
"""Rose Installer - System detection and dependency management.

Detects system capabilities and installs Rose with appropriate
configuration for the target machine.

Usage:
    python install_rose.py
    python install_rose.py --cpu-only
    python install_rose.py --check-only
"""

import os
import sys
import platform
import subprocess
import shutil
import ctypes
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class SystemInfo:
    """Detected system information."""
    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    python_version: str = ""
    python_implementation: str = ""
    cpu_count: int = 0
    ram_gb: float = 0.0
    has_nvidia_gpu: bool = False
    nvidia_driver_version: str = ""
    cuda_version: str = ""
    cuda_major: int = 0
    cuda_minor: int = 0
    has_compiler: bool = False
    has_cmake: bool = False
    has_git: bool = False
    disk_free_gb: float = 0.0
    recommended_wheel: str = "cpu"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def detect_system() -> SystemInfo:
    """Detect system capabilities."""
    info = SystemInfo()

    # OS detection
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.architecture = platform.machine()
    info.python_version = platform.python_version()
    info.python_implementation = platform.python_implementation()
    info.cpu_count = os.cpu_count() or 1

    # RAM detection (Windows)
    if info.os_name == "Windows":
        try:
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", c_ulong),
                    ("dwMemoryLoad", c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            info.ram_gb = mem.ullTotalPhys / (1024 ** 3)
        except Exception:
            info.ram_gb = 0.0

    # Disk space detection
    try:
        usage = shutil.disk_usage("/")
        info.disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        info.disk_free_gb = 0.0

    # NVIDIA GPU detection
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info.has_nvidia_gpu = True
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                if len(parts) >= 2:
                    info.nvidia_driver_version = parts[0].strip()
                    info.cuda_version = parts[1].strip()
                    # Parse CUDA version
                    cuda_parts = info.cuda_version.split(".")
                    if len(cuda_parts) >= 2:
                        try:
                            info.cuda_major = int(cuda_parts[0])
                            info.cuda_minor = int(cuda_parts[1])
                        except ValueError:
                            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info.has_nvidia_gpu = False

    # Compiler detection (Windows)
    if info.os_name == "Windows":
        try:
            result = subprocess.run(
                ["where", "cl"], capture_output=True, timeout=5
            )
            info.has_compiler = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info.has_compiler = False

    # CMake detection
    try:
        result = subprocess.run(
            ["cmake", "--version"], capture_output=True, timeout=5
        )
        info.has_cmake = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info.has_cmake = False

    # Git detection
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, timeout=5
        )
        info.has_git = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info.has_git = False

    # Determine recommended wheel
    if info.has_nvidia_gpu and info.cuda_major > 0:
        if info.cuda_major >= 13:
            if info.cuda_minor >= 2:
                info.recommended_wheel = "cu132"
            else:
                info.recommended_wheel = "cu130"
        elif info.cuda_major >= 12:
            if info.cuda_minor >= 5:
                info.recommended_wheel = "cu125"
            elif info.cuda_minor >= 4:
                info.recommended_wheel = "cu124"
            elif info.cuda_minor >= 3:
                info.recommended_wheel = "cu123"
            elif info.cuda_minor >= 2:
                info.recommended_wheel = "cu122"
            else:
                info.recommended_wheel = "cu121"
        elif info.cuda_major >= 11:
            info.recommended_wheel = "cu118"
        else:
            info.recommended_wheel = "cpu"
            info.warnings.append("CUDA version too old, falling back to CPU mode")
    else:
        info.recommended_wheel = "cpu"

    # Validate requirements
    py_parts = info.python_version.split(".")
    py_major = int(py_parts[0])
    py_minor = int(py_parts[1])

    if py_major < 3 or (py_major == 3 and py_minor < 10):
        info.errors.append("Python 3.10 or later required")

    if py_major == 3 and py_minor >= 13:
        info.warnings.append(
            "Python 3.13+ may not have prebuilt wheels. "
            "Consider using Python 3.10-3.12 for best compatibility."
        )

    if info.ram_gb > 0 and info.ram_gb < 8:
        info.warnings.append(f"Only {info.ram_gb:.1f} GB RAM detected. 16 GB recommended.")

    if info.disk_free_gb > 0 and info.disk_free_gb < 5:
        info.warnings.append(f"Only {info.disk_free_gb:.1f} GB free disk space. 10 GB recommended.")

    return info


def get_wheel_index_url(wheel_type: str) -> str:
    """Get the extra index URL for the given wheel type."""
    base = "https://abetlen.github.io/llama-cpp-python/whl"
    return f"{base}/{wheel_type}"


def install_dependencies(info: SystemInfo, install_type: str = "core") -> bool:
    """Install Rose dependencies."""
    print("\n[Installing Dependencies]")

    # Upgrade pip first
    print("  Upgrading pip...")
    subprocess.run(
        [sys.executable, "-m", "pip", "upgrade", "pip", "--quiet"],
        capture_output=True
    )

    # Install llama-cpp-python with prebuilt wheel
    wheel_url = get_wheel_index_url(info.recommended_wheel)
    print(f"  Installing llama-cpp-python (wheel: {info.recommended_wheel})...")

    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "llama-cpp-python>=0.3.0",
            "--extra-index-url", wheel_url,
            "--only-binary=llama-cpp-python",
            "--quiet"
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("  Prebuilt wheel not available, attempting source compilation...")
        if not info.has_compiler:
            print("  ERROR: No compiler found for source compilation.")
            print("  Install Visual Studio Build Tools or use Python 3.10-3.12.")
            return False

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.0", "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR: Failed to install llama-cpp-python: {result.stderr}")
            return False

    print("  llama-cpp-python installed.")

    # Install other core dependencies
    core_deps = [
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "diskcache>=5.6.0",
        "Jinja2>=3.1.0",
        "markdown-it-py>=3.0.0",
        "packaging>=23.0",
        "pathspec>=0.11.0",
    ]

    print("  Installing core dependencies...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + core_deps + ["--quiet"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: Some core dependencies failed: {result.stderr}")

    # Install optional dependencies based on install type
    if install_type in ("ui", "full", "dev"):
        print("  Installing UI dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "PySide6>=6.5.0", "--quiet"],
            capture_output=True
        )

    if install_type in ("full", "dev"):
        print("  Installing browser dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright>=1.40.0", "--quiet"],
            capture_output=True
        )
        print("  Installing Playwright Chromium...")
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True
        )

    if install_type in ("full", "dev"):
        print("  Installing vision dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "opencv-python>=4.8.0", "--quiet"],
            capture_output=True
        )

    if install_type == "dev":
        print("  Installing development dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "pytest>=7.0.0", "pytest-cov>=4.0.0", "pytest-asyncio>=0.21.0",
             "--quiet"],
            capture_output=True
        )

    return True


def create_directories():
    """Create required directories."""
    print("\n[Creating Directories]")
    dirs = ["config", "models", "workspace", "workspace/media", "logs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}/")


def verify_installation() -> bool:
    """Verify the installation works."""
    print("\n[Verifying Installation]")
    success = True

    checks = [
        ("llama_cpp", "llama-cpp-python"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("yaml", "PyYAML"),
        ("rich", "rich"),
        ("pydantic", "pydantic"),
    ]

    for module, name in checks:
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: FAILED")
            success = False

    return success


def print_system_info(info: SystemInfo):
    """Print detected system information."""
    print("\n[System Information]")
    print(f"  OS: {info.os_name} {info.os_version}")
    print(f"  Architecture: {info.architecture}")
    print(f"  Python: {info.python_version} ({info.python_implementation})")
    print(f"  CPU cores: {info.cpu_count}")
    if info.ram_gb > 0:
        print(f"  RAM: {info.ram_gb:.1f} GB")
    if info.disk_free_gb > 0:
        print(f"  Free disk: {info.disk_free_gb:.1f} GB")

    if info.has_nvidia_gpu:
        print(f"  NVIDIA GPU: Yes")
        if info.nvidia_driver_version:
            print(f"  Driver: {info.nvidia_driver_version}")
        if info.cuda_version:
            print(f"  CUDA: {info.cuda_version}")
    else:
        print(f"  NVIDIA GPU: No")

    print(f"  Compiler: {'Yes' if info.has_compiler else 'No'}")
    print(f"  CMake: {'Yes' if info.has_cmake else 'No'}")
    print(f"  Git: {'Yes' if info.has_git else 'No'}")

    print(f"\n  Recommended wheel: {info.recommended_wheel}")

    if info.warnings:
        print("\n  Warnings:")
        for w in info.warnings:
            print(f"    - {w}")

    if info.errors:
        print("\n  Errors:")
        for e in info.errors:
            print(f"    - {e}")


def main():
    """Main installer function."""
    import argparse
    parser = argparse.ArgumentParser(description="Rose Installer")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--check-only", action="store_true", help="Only check system, don't install")
    parser.add_argument("--install-type", choices=["core", "ui", "full", "dev"],
                       default="core", help="Installation type")
    args = parser.parse_args()

    print("=" * 50)
    print("  Rose - Autonomous AI Agent Installer")
    print("=" * 50)

    # Detect system
    info = detect_system()

    if args.cpu_only:
        info.recommended_wheel = "cpu"
        info.has_nvidia_gpu = False

    print_system_info(info)

    if args.check_only:
        return 0 if not info.errors else 1

    if info.errors:
        print("\nCannot proceed due to errors above.")
        return 1

    # Create virtual environment
    print("\n[Creating Virtual Environment]")
    venv_path = Path("venv")
    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("  Virtual environment created.")
    else:
        print("  Virtual environment already exists.")

    # Get the Python executable in the venv
    if sys.platform == "win32":
        venv_python = venv_path / "Scripts" / "python.exe"
    else:
        venv_python = venv_path / "bin" / "python"

    # Install dependencies
    success = install_dependencies(info, args.install_type)
    if not success:
        print("\nInstallation failed.")
        return 1

    # Create directories
    create_directories()

    # Verify
    if verify_installation():
        print("\n" + "=" * 50)
        print("  Installation Complete!")
        print("=" * 50)
        print(f"\n  Mode: {info.recommended_wheel}")
        print(f"  Type: {args.install_type}")
        print("\n  To run Rose:")
        print("    venv\\Scripts\\activate")
        print("    python run.py")
        print("\n  Models directory: models/")
        print("  Place .gguf model files there.")
        return 0
    else:
        print("\nInstallation completed with some warnings.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
