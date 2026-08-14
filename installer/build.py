"""Build script for Rose installer.

Produces:
  - dist/Rose/          (PyInstaller output - portable Rose)
  - dist/Rose-Setup.exe (Inno Setup installer)

Usage:
  python installer/build.py [--portable] [--installer] [--all]
"""
import os
import sys
import shutil
import subprocess
import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
INSTALLER_DIR = ROOT / "installer"


def log(msg: str):
    print(f"[BUILD] {msg}")


def _clean_duplicates(rose_dir: Path):
    """Remove duplicate/unnecessary DLLs that PyInstaller pulls in via ctypes detection."""
    internal = rose_dir / "_internal"
    if not internal.exists():
        return

    # Remove cublas13 from CUDA toolkit (ggml-cuda uses cublas12 from nvidia pip package)
    for name in ["cublas64_13.dll", "cublasLt64_13.dll"]:
        f = internal / name
        if f.exists():
            f.unlink()
            log(f"Removed duplicate {name}")

    # Remove duplicate nvidia/cublas/bin/ (keep nvidia/ only)
    dup_dir = internal / "nvidia" / "cublas"
    if dup_dir.exists():
        import shutil
        shutil.rmtree(dup_dir)
        log("Removed duplicate nvidia/cublas/bin/")


def clean():
    """Clean previous build artifacts."""
    log("Cleaning previous builds...")
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            shutil.rmtree(d)
    log("Clean complete.")


def build_portable():
    """Build portable Rose directory using PyInstaller."""
    log("Building portable Rose...")
    spec_file = INSTALLER_DIR / "rose_gui.spec"

    if not spec_file.exists():
        log(f"ERROR: Spec file not found: {spec_file}")
        return False

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        str(spec_file),
    ]

    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        log("ERROR: PyInstaller build failed")
        return False

    rose_dir = DIST_DIR / "Rose"
    if not rose_dir.exists():
        log("ERROR: Rose directory not found in dist/")
        return False

    # Verify key files exist
    rose_exe = rose_dir / "Rose.exe"
    if not rose_exe.exists():
        log("ERROR: Rose.exe not found")
        return False

    size_mb = rose_exe.stat().st_size / (1024 * 1024)
    log(f"Rose.exe built: {size_mb:.1f} MB")

    # Remove duplicate/unnecessary DLLs pulled in by PyInstaller's ctypes detection
    _clean_duplicates(rose_dir)

    # Calculate SHA256
    sha256 = hashlib.sha256(rose_exe.read_bytes()).hexdigest()
    log(f"Rose.exe SHA256: {sha256}")

    log("Portable build complete.")
    return True


def build_installer():
    """Build Windows installer using Inno Setup."""
    log("Building installer...")

    iscc_path = None
    # Check common Inno Setup locations
    possible_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Inno Setup 6", "ISCC.exe"),
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            iscc_path = p
            break

    if not iscc_path:
        # Try PATH
        try:
            result = subprocess.run(["where", "ISCC.exe"], capture_output=True, text=True)
            if result.returncode == 0:
                iscc_path = result.stdout.strip().split("\n")[0]
        except Exception:
            pass

    if not iscc_path:
        log("WARNING: Inno Setup not found. Skipping installer build.")
        log("Install Inno Setup from: https://jrsoftware.org/isdl.php")
        log("Then run: installer/build.py --installer")
        return False

    iss_file = INSTALLER_DIR / "rose_setup.iss"
    if not iss_file.exists():
        log(f"ERROR: Inno Setup script not found: {iss_file}")
        return False

    cmd = [iscc_path, str(iss_file)]
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode != 0:
        log("ERROR: Inno Setup build failed")
        return False

    setup_exe = DIST_DIR / "Rose-Setup.exe"
    if setup_exe.exists():
        size_mb = setup_exe.stat().st_size / (1024 * 1024)
        sha256 = hashlib.sha256(setup_exe.read_bytes()).hexdigest()
        log(f"Rose-Setup.exe built: {size_mb:.1f} MB")
        log(f"Rose-Setup.exe SHA256: {sha256}")
    else:
        log("WARNING: Rose-Setup.exe not found in dist/")
        return False

    log("Installer build complete.")
    return True


def print_usage():
    print("""
Rose Installer Build Script
===========================

Usage:
  python installer/build.py [option]

Options:
  (none)       Build portable Rose directory
  --portable   Build portable Rose directory
  --installer  Build Windows installer (requires Inno Setup)
  --all        Build both portable and installer
  --clean      Clean build artifacts only
  --help       Show this help

Requirements:
  - Python 3.10+
  - PyInstaller (pip install pyinstaller)
  - Inno Setup 6 (for installer, optional)
    Download: https://jrsoftware.org/isdl.php
""")


def main():
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print_usage()
        return

    start = time.time()

    if "--clean" in args:
        clean()
        return

    clean()

    if "--all" in args:
        if build_portable():
            build_installer()
    elif "--installer" in args:
        build_installer()
    elif "--portable" in args:
        build_portable()
    else:
        # Default: build portable
        build_portable()

    elapsed = time.time() - start
    log(f"Build completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
