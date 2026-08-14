"""System detection for Rose installer."""
import os
import sys
import subprocess
import platform
import ctypes
import json
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class GPUInfo:
    name: str = "Not detected"
    vram_mb: int = 0
    driver_version: str = ""
    cuda_compatible: bool = False
    compute_capability: str = ""


@dataclass
class SystemInfo:
    windows_version: str = ""
    windows_build: int = 0
    architecture: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    free_disk_gb: float = 0.0
    total_disk_gb: float = 0.0
    gpu: GPUInfo = field(default_factory=GPUInfo)
    python_version: str = ""
    has_internet: bool = False
    has_nvidia_driver: bool = False
    has_cuda_toolkit: bool = False
    cuda_version: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_internet() -> bool:
    """Check if internet connection is available."""
    try:
        import urllib.request
        urllib.request.urlopen("https://www.google.com", timeout=5)
        return True
    except Exception:
        try:
            import urllib.request
            urllib.request.urlopen("https://1.1.1.1", timeout=5)
            return True
        except Exception:
            return False


def detect_gpu() -> GPUInfo:
    """Detect NVIDIA GPU using nvidia-smi."""
    gpu = GPUInfo()
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                gpu.name = parts[0].strip()
                gpu.vram_mb = int(parts[1].strip())
                gpu.driver_version = parts[2].strip()
                gpu.compute_capability = parts[3].strip()
                gpu.cuda_compatible = True
                gpu.vram_mb = gpu.vram_mb  # Already in MB from nvidia-smi
    except FileNotFoundError:
        gpu.warnings = ["nvidia-smi not found"]
    except Exception as e:
        gpu.warnings = [f"GPU detection error: {e}"]
    return gpu


def detect_cuda_toolkit() -> Tuple[bool, str]:
    """Detect CUDA toolkit installation."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    version = line.split("release")[-1].strip().rstrip(",").split(",")[0]
                    return True, version
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return False, ""


def get_system_info(install_dir: str = "C:\\Program Files\\Rose") -> SystemInfo:
    """Gather comprehensive system information."""
    info = SystemInfo()

    # Windows version
    info.windows_version = platform.platform()
    try:
        info.windows_build = int(platform.version().split(".")[-1])
    except (ValueError, IndexError):
        info.windows_build = 0

    # Architecture
    info.architecture = platform.machine()

    # CPU
    try:
        import multiprocessing
        info.cpu_cores = multiprocessing.cpu_count()
    except Exception:
        info.cpu_cores = 0

    info.cpu_name = platform.processor() or "Unknown CPU"

    # RAM
    try:
        kernel32 = ctypes.windll.kernel32
        c_ulonglong = ctypes.c_ulonglong
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", c_ulonglong),
                ("ullAvailPhys", c_ulonglong),
                ("ullTotalPageFile", c_ulonglong),
                ("ullAvailPageFile", c_ulonglong),
                ("ullTotalVirtual", c_ulonglong),
                ("ullAvailVirtual", c_ulonglong),
                ("ullAvailExtendedVirtual", c_ulonglong),
            ]
        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        info.ram_gb = round(mem.ullTotalPhys / (1024**3), 1)
    except Exception:
        info.ram_gb = 0.0

    # Disk space
    try:
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(install_dir[:3]),
            None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
        )
        info.free_disk_gb = round(free_bytes.value / (1024**3), 1)
        info.total_disk_gb = round(total_bytes.value / (1024**3), 1)
    except Exception:
        try:
            import shutil
            total, used, free = shutil.disk_usage(install_dir[:3])
            info.free_disk_gb = round(free / (1024**3), 1)
            info.total_disk_gb = round(total / (1024**3), 1)
        except Exception:
            info.free_disk_gb = 0.0
            info.total_disk_gb = 0.0

    # GPU
    info.gpu = detect_gpu()
    info.has_nvidia_driver = info.gpu.cuda_compatible

    # CUDA toolkit
    info.has_cuda_toolkit, info.cuda_version = detect_cuda_toolkit()

    # Internet
    info.has_internet = check_internet()

    # Python (internal use only)
    info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Validation
    if info.architecture not in ("AMD64", "x86_64", "Windows AMD64"):
        info.warnings.append(f"Architecture {info.architecture} may not be fully supported")

    if info.ram_gb < 8:
        info.warnings.append(f"Only {info.ram_gb} GB RAM detected. 16 GB recommended for optimal performance.")

    if info.free_disk_gb < 10:
        info.warnings.append(f"Only {info.free_disk_gb} GB free disk space. At least 10 GB recommended.")

    if not info.gpu.cuda_compatible:
        info.warnings.append("No NVIDIA GPU detected. Rose will run in CPU mode (slower).")

    if not info.has_internet:
        info.errors.append("No internet connection detected. Internet is required for initial installation.")

    return info


def get_check_results(info: SystemInfo) -> List[dict]:
    """Format system info as check results for UI display."""
    checks = []

    checks.append({
        "label": f"Windows",
        "detail": f"{info.windows_version}",
        "status": "pass" if info.windows_build >= 10000 else "warn",
    })

    checks.append({
        "label": f"Architecture",
        "detail": f"{info.architecture}",
        "status": "pass" if "64" in info.architecture or "AMD64" in info.architecture else "warn",
    })

    checks.append({
        "label": "CPU",
        "detail": f"{info.cpu_cores} cores",
        "status": "pass" if info.cpu_cores >= 4 else "warn",
    })

    checks.append({
        "label": "RAM",
        "detail": f"{info.ram_gb} GB",
        "status": "pass" if info.ram_gb >= 16 else ("warn" if info.ram_gb >= 8 else "fail"),
    })

    checks.append({
        "label": "Disk Space",
        "detail": f"{info.free_disk_gb} GB free",
        "status": "pass" if info.free_disk_gb >= 10 else "fail",
    })

    if info.gpu.cuda_compatible:
        checks.append({
            "label": "NVIDIA GPU",
            "detail": f"{info.gpu.name} ({info.gpu.vram_mb // 1024} GB VRAM)",
            "status": "pass",
        })
        checks.append({
            "label": "CUDA",
            "detail": f"Driver {info.gpu.driver_version}",
            "status": "pass",
        })
    else:
        checks.append({
            "label": "NVIDIA GPU",
            "detail": "Not detected (CPU mode)",
            "status": "warn",
        })

    checks.append({
        "label": "Internet",
        "detail": "Connected" if info.has_internet else "Not connected",
        "status": "pass" if info.has_internet else "fail",
    })

    return checks


if __name__ == "__main__":
    info = get_system_info()
    checks = get_check_results(info)
    for c in checks:
        icon = {"pass": "[OK]", "warn": "[!!]", "fail": "[X]"}.get(c["status"], "[?]")
        print(f"{icon} {c['label']}: {c['detail']}")
