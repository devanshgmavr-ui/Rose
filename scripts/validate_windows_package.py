"""Packaging validator for Rose.exe - checks all required files are present.

Usage:
    python scripts/validate_windows_package.py [dist_path]
"""
import sys
import os
from pathlib import Path


def validate(dist_path: Path) -> list[str]:
    """Validate a Rose dist directory. Returns list of errors (empty = pass)."""
    errors = []
    rose_dir = dist_path / "Rose" if (dist_path / "Rose").exists() else dist_path
    internal = rose_dir / "_internal"

    if not rose_dir.exists():
        return [f"Dist directory not found: {rose_dir}"]

    # 1. Rose.exe exists
    exe = rose_dir / "Rose.exe"
    if not exe.exists():
        errors.append("Rose.exe not found")

    # 2. Critical DLLs in _internal
    if internal.exists():
        critical_dlls = [
            "python314.dll",
            "ucrtbase.dll",
            "VCRUNTIME140.dll",
            "VCRUNTIME140_1.dll",
            "MSVCP140.dll",
        ]
        for dll in critical_dlls:
            if not (internal / dll).exists():
                errors.append(f"Missing critical DLL: {dll}")

        # 3. CUDA DLLs
        llama_lib = internal / "llama_cpp" / "lib"
        if llama_lib.exists():
            cuda_dlls = ["ggml-cuda.dll", "llama.dll"]
            for dll in cuda_dlls:
                if not (llama_lib / dll).exists():
                    errors.append(f"Missing CUDA DLL: llama_cpp/lib/{dll}")
        else:
            errors.append("Missing llama_cpp/lib directory")

        nvidia_dir = internal / "nvidia"
        if nvidia_dir.exists():
            cublas = list(nvidia_dir.glob("cublas64_*.dll"))
            if not cublas:
                errors.append("Missing cublas64_*.dll in nvidia/")
        else:
            errors.append("Missing nvidia/ directory")

        # 4. Agent code
        agent_dir = internal / "agent"
        if not agent_dir.exists():
            errors.append("Missing agent/ directory")

        # 5. Config files
        configs_dir = internal / "configs"
        if not configs_dir.exists():
            errors.append("Missing configs/ directory")

    else:
        errors.append("Missing _internal/ directory")

    # 6. Install size check (should be > 500 MB for a working build)
    total_size = sum(f.stat().st_size for f in rose_dir.rglob("*") if f.is_file())
    if total_size < 200 * 1024 * 1024:  # < 200 MB
        errors.append(f"Build suspiciously small: {total_size / 1024 / 1024:.0f} MB")

    return errors


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    errs = validate(path)
    if errs:
        print("VALIDATION FAILED:")
        for e in errs:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print(f"VALIDATION PASSED - {path}")
        sys.exit(0)
