#!/usr/bin/env python3
"""Rose Model Downloader.

Downloads Qwen2.5-VL-7B-Instruct GGUF model files from HuggingFace.
Supports resume, validation, and idempotent operation.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir ./models
    python scripts/download_models.py --check-only
"""

import os
import sys
import hashlib
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

# Model files expected by Rose (from agent/core/config.py)
MODEL_FILES = {
    "main": {
        "repo": "bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        "description": "Qwen2.5-VL-7B-Instruct Q4_K_M (main model)",
        "expected_size_gb": 4.68,
    },
    "mmproj": {
        "repo": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
        "filename": "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
        "description": "Vision projector (mmproj) f16",
        "expected_size_gb": 1.35,
    },
}

TOTAL_SIZE_GB = sum(f["expected_size_gb"] for f in MODEL_FILES.values())


def print_header():
    print("=" * 50)
    print("  ROSE MODEL DOWNLOADER")
    print("  Qwen2.5-VL-7B-Instruct")
    print("=" * 50)
    print()


def print_result(name: str, passed: bool, detail: str = ""):
    status = "[OK]" if passed else "[FAIL]"
    suffix = f" - {detail}" if detail else ""
    print(f"  {status} {name}{suffix}")


def check_disk_space(model_dir: Path, required_gb: float) -> Tuple[bool, float]:
    """Check if there's enough disk space."""
    try:
        usage = shutil.disk_usage(model_dir)
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= required_gb, free_gb
    except Exception:
        return True, 0.0


def get_file_size_gb(filepath: Path) -> float:
    """Get file size in GB."""
    if filepath.exists():
        return filepath.stat().st_size / (1024 ** 3)
    return 0.0


def download_with_huggingface_hub(
    repo_id: str,
    filename: str,
    local_dir: Path,
) -> Optional[Path]:
    """Download a file using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download

        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

        return Path(downloaded_path)
    except ImportError:
        print("    huggingface_hub not installed. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub>=0.20.0", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded_path)
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def download_with_requests(
    repo_id: str,
    filename: str,
    local_dir: Path,
) -> Optional[Path]:
    """Download a file using requests (fallback)."""
    try:
        import requests
        from tqdm import tqdm

        local_dir.mkdir(parents=True, exist_ok=True)
        target_path = local_dir / filename

        # Use HuggingFace API
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

        # Support resume
        headers = {}
        existing_size = 0
        if target_path.exists():
            existing_size = target_path.stat().st_size
            headers["Range"] = f"bytes={existing_size}-"

        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if response.status_code == 206:
            # Partial content - resume
            mode = "ab"
            total_size += existing_size
        else:
            # Full download
            mode = "wb"
            existing_size = 0

        with open(target_path, mode) as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=f"  Downloading {filename}",
            initial=existing_size,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

        return target_path
    except ImportError:
        print("    requests not installed. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "requests", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return download_with_requests(repo_id, filename, local_dir)
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def download_file(
    repo_id: str,
    filename: str,
    local_dir: Path,
) -> Optional[Path]:
    """Download a file, trying huggingface_hub first, then requests."""
    # Check if already downloaded
    target_path = local_dir / filename
    if target_path.exists():
        size_gb = get_file_size_gb(target_path)
        if size_gb > 0.1:  # Sanity check - file should be > 100MB
            print(f"  [OK] {filename} already exists ({size_gb:.2f} GB)")
            return target_path

    print(f"  Downloading {filename}...")

    # Try huggingface_hub first
    result = download_with_huggingface_hub(repo_id, filename, local_dir)
    if result and result.exists():
        return result

    # Fallback to requests
    print("  Retrying with requests...")
    result = download_with_requests(repo_id, filename, local_dir)
    if result and result.exists():
        return result

    return None


def validate_file(filepath: Path, expected_size_gb: float) -> bool:
    """Validate a downloaded file."""
    if not filepath.exists():
        return False

    actual_size_gb = get_file_size_gb(filepath)

    # Check size is within 10% of expected
    if expected_size_gb > 0:
        tolerance = 0.1
        if abs(actual_size_gb - expected_size_gb) / expected_size_gb > tolerance:
            print(f"    Warning: Size mismatch - expected ~{expected_size_gb:.2f} GB, got {actual_size_gb:.2f} GB")
            return False

    return True


def check_models(model_dir: Path) -> bool:
    """Check if all required model files exist."""
    all_ok = True
    for key, info in MODEL_FILES.items():
        filepath = model_dir / info["filename"]
        if filepath.exists():
            size_gb = get_file_size_gb(filepath)
            if validate_file(filepath, info["expected_size_gb"]):
                print_result(info["description"], True, f"{size_gb:.2f} GB")
            else:
                print_result(info["description"], False, f"Invalid file ({size_gb:.2f} GB)")
                all_ok = False
        else:
            print_result(info["description"], False, "Not found")
            all_ok = False
    return all_ok


def download_models(model_dir: Path, check_only: bool = False) -> bool:
    """Download all required model files."""
    print_header()

    model_dir.mkdir(parents=True, exist_ok=True)

    if check_only:
        print("Checking model files...\n")
        return check_models(model_dir)

    # Check disk space
    has_space, free_gb = check_disk_space(model_dir, TOTAL_SIZE_GB + 1.0)
    if not has_space:
        print(f"  [FAIL] Insufficient disk space")
        print(f"    Required: ~{TOTAL_SIZE_GB:.1f} GB + 1 GB buffer")
        print(f"    Available: {free_gb:.1f} GB")
        return False

    print(f"  Disk space: {free_gb:.1f} GB available ({TOTAL_SIZE_GB:.1f} GB required)")
    print()

    # Download each file
    success = True
    for key, info in MODEL_FILES.items():
        print(f"[{info['description']}]")
        result = download_file(
            repo_id=info["repo"],
            filename=info["filename"],
            local_dir=model_dir,
        )
        if result:
            if validate_file(result, info["expected_size_gb"]):
                size_gb = get_file_size_gb(result)
                print_result(info["description"], True, f"{size_gb:.2f} GB")
            else:
                print_result(info["description"], False, "Validation failed")
                success = False
        else:
            print_result(info["description"], False, "Download failed")
            success = False
        print()

    return success


def main():
    parser = argparse.ArgumentParser(description="Rose Model Downloader")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to store model files (default: ./models)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if models exist, don't download",
    )
    args = parser.parse_args()

    success = download_models(args.model_dir, args.check_only)

    if success:
        print("=" * 50)
        print("  Models ready!")
        print("=" * 50)
    else:
        print("=" * 50)
        print("  Some models failed to download")
        print("=" * 50)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
