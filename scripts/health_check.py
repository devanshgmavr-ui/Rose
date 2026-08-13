#!/usr/bin/env python3
"""Health check script for the local agent."""

import sys
import os
import ctypes
import importlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add CUDA bin directories to DLL search path (needed for llama.cpp CUDA backend)
_cuda_bin_dirs = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin",
]
for _cuda_dir in _cuda_bin_dirs:
    if os.path.isdir(_cuda_dir):
        try:
            ctypes.windll.kernel32.SetDllDirectoryW(_cuda_dir)
        except Exception:
            pass
        os.environ["PATH"] = _cuda_dir + ";" + os.environ.get("PATH", "")


class HealthChecker:
    """Comprehensive health checker for the local agent."""
    
    def __init__(self):
        self.results = {}
        self.all_passed = True
    
    def check(self, name: str, check_func):
        """Run a health check and store the result."""
        try:
            result = check_func()
            self.results[name] = {"status": "OK" if result else "FAIL", "details": result}
            if not result:
                self.all_passed = False
        except Exception as e:
            self.results[name] = {"status": "ERROR", "details": str(e)}
            self.all_passed = False
    
    def print_results(self):
        """Print formatted health check results."""
        print("\n" + "=" * 50)
        print("LOCAL AGENT HEALTH CHECK")
        print("=" * 50)
        
        for name, result in self.results.items():
            status = result["status"]
            details = result["details"]
            
            if status == "OK":
                icon = "[OK]"
                color = "\033[92m"  # Green
            elif status == "FAIL":
                icon = "[FAIL]"
                color = "\033[91m"  # Red
            else:
                icon = "[ERROR]"
                color = "\033[93m"  # Yellow
            
            reset = "\033[0m"
            print(f"{color}{icon}{reset} {name}: {status}")
            
            if details and details != True and details != False:
                print(f"       {details}")
        
        print("=" * 50)
        
        if self.all_passed:
            print("\n\033[92m[SUCCESS]\033[0m All health checks passed!")
        else:
            print("\n\033[91m[WARNING]\033[0m Some health checks failed.")
            print("Please review the issues above.")
        
        return self.all_passed


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        return f"Python {version.major}.{version.minor}.{version.micro}"
    return False


def check_virtual_env():
    """Check if running in virtual environment."""
    return hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )


def check_dependencies():
    """Check required dependencies."""
    required = ['pydantic', 'yaml', 'dotenv', 'llama_cpp']
    missing = []
    
    for dep in required:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
        except Exception:
            # Module imported but failed to load (e.g., missing CUDA DLLs)
            pass
    
    if missing:
        return f"Missing: {', '.join(missing)}"
    return True


def check_gpu():
    """Check GPU availability via nvidia-smi."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                name = parts[0].strip()
                vram_mb = parts[1].strip()
                driver = parts[2].strip()
                return f"{name} ({int(float(vram_mb))} MB VRAM, Driver {driver})"
            return result.stdout.strip()
        return "nvidia-smi failed"
    except FileNotFoundError:
        return "nvidia-smi not found"
    except Exception as e:
        return f"GPU check error: {e}"


def check_model():
    """Check if model file exists."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    
    model_path = os.getenv("MODEL_PATH", "./models/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
    full_path = PROJECT_ROOT / model_path
    
    if full_path.exists():
        size_gb = full_path.stat().st_size / (1024**3)
        result = f"Found ({size_gb:.2f} GB)"
    else:
        return f"Not found: {full_path}"
    
    # Check mmproj (vision projector) for VL models
    mmproj_path = os.getenv("MMPROJ_PATH", "./models/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")
    mmproj_full = PROJECT_ROOT / mmproj_path
    if mmproj_full.exists():
        size_gb = mmproj_full.stat().st_size / (1024**3)
        result += f" + mmproj ({size_gb:.2f} GB)"
    
    return result


def check_cuda_support():
    """Check llama-cpp-python CUDA support by checking for ggml-cuda.dll."""
    try:
        from llama_cpp import Llama
        # Check if ggml-cuda.dll exists in the llama_cpp lib directory
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).parent / "lib"
        cuda_dll = lib_dir / "ggml-cuda.dll"
        if cuda_dll.exists():
            size_mb = cuda_dll.stat().st_size / (1024**2)
            return f"CUDA backend available (ggml-cuda.dll {size_mb:.0f} MB)"
        return "CPU-only (no ggml-cuda.dll)"
    except ImportError:
        return "llama-cpp-python not installed"
    except Exception:
        # llama_cpp module exists but DLLs may not load without CUDA PATH
        # Check for ggml-cuda.dll directly
        try:
            import llama_cpp
            lib_dir = Path(llama_cpp.__file__).parent / "lib"
            cuda_dll = lib_dir / "ggml-cuda.dll"
            if cuda_dll.exists():
                size_mb = cuda_dll.stat().st_size / (1024**2)
                return f"CUDA backend available (ggml-cuda.dll {size_mb:.0f} MB)"
        except Exception:
            pass
        return "llama-cpp-python installed (CUDA status unknown)"


def main():
    """Run all health checks."""
    checker = HealthChecker()
    
    print("\nRunning health checks...")
    
    checker.check("Environment", check_virtual_env)
    checker.check("Python", check_python_version)
    checker.check("Dependencies", check_dependencies)
    checker.check("GPU", check_gpu)
    checker.check("Model Runtime", check_cuda_support)
    checker.check("Model", check_model)
    
    success = checker.print_results()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
