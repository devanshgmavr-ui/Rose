"""Model downloader with progress tracking and resume support."""
import os
import time
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ModelFile:
    name: str
    url: str
    expected_size: int  # bytes
    description: str
    sha256: Optional[str] = None  # None = no hash verification


@dataclass
class DownloadProgress:
    bytes_downloaded: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    current_file: str = ""
    file_index: int = 0
    total_files: int = 0
    status: str = "idle"  # idle, downloading, verifying, complete, error
    error_message: str = ""
    filename: str = ""


# Qwen2.5-VL model files
QWEN_MODEL_FILES = [
    ModelFile(
        name="Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        url="https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        expected_size=4_360_000_000,  # ~4.36 GB
        description="Qwen2.5-VL-7B-Instruct (main model)",
    ),
    ModelFile(
        name="mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
        url="https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
        expected_size=1_260_000_000,  # ~1.26 GB
        description="Vision projector",
    ),
]


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"


def format_speed(bytes_per_sec: float) -> str:
    """Format download speed."""
    if bytes_per_sec < 1024:
        return f"{bytes_per_sec:.0f} B/s"
    elif bytes_per_sec < 1024**2:
        return f"{bytes_per_sec / 1024:.1f} KB/s"
    else:
        return f"{bytes_per_sec / 1024**2:.1f} MB/s"


def format_eta(seconds: float) -> str:
    """Format ETA."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


class ModelDownloader:
    """Downloads model files with progress, resume, and error handling."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cancelled = False
        self._progress_callback: Optional[Callable[[DownloadProgress], None]] = None
        self._retry_count = 3
        self._timeout = 30

    def set_progress_callback(self, callback: Callable[[DownloadProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def cancel(self):
        """Cancel download."""
        self._cancelled = True

    def _report_progress(self, progress: DownloadProgress):
        """Report progress to callback."""
        if self._progress_callback:
            self._progress_callback(progress)

    def _download_file(self, model: ModelFile) -> Tuple[bool, str]:
        """Download a single model file with resume support."""
        filepath = self.models_dir / model.name
        temp_filepath = self.models_dir / f"{model.name}.tmp"

        # Check if file already exists and is complete
        if filepath.exists():
            existing_size = filepath.stat().st_size
            if existing_size >= model.expected_size * 0.99:  # Allow 1% tolerance
                progress = DownloadProgress(
                    bytes_downloaded=existing_size,
                    total_bytes=model.expected_size,
                    status="complete",
                    current_file=model.description,
                    filename=model.name,
                )
                self._report_progress(progress)
                return True, "File already exists and is complete"

        # Resume from partial download
        resume_from = 0
        if temp_filepath.exists():
            resume_from = temp_filepath.stat().st_size

        for attempt in range(self._retry_count):
            if self._cancelled:
                return False, "Download cancelled"

            try:
                req = urllib.request.Request(model.url)
                if resume_from > 0:
                    req.add_header("Range", f"bytes={resume_from}-")

                with urllib.request.urlopen(req, timeout=self._timeout) as response:
                    # Check if server supports resume
                    if resume_from > 0 and response.status == 200:
                        # Server doesn't support resume, start fresh
                        resume_from = 0
                        if temp_filepath.exists():
                            temp_filepath.unlink()

                    total_size = int(response.headers.get("Content-Length", model.expected_size))
                    if resume_from > 0 and response.status == 206:
                        total_size = resume_from + total_size

                    downloaded = resume_from
                    start_time = time.time()
                    last_report = start_time

                    with open(temp_filepath, "ab" if resume_from > 0 else "wb") as f:
                        while True:
                            if self._cancelled:
                                f.close()
                                return False, "Download cancelled"

                            chunk = response.read(1024 * 1024)  # 1MB chunks
                            if not chunk:
                                break

                            f.write(chunk)
                            downloaded += len(chunk)

                            current_time = time.time()
                            elapsed = current_time - start_time

                            if elapsed > 0 and current_time - last_report >= 0.25:
                                speed = downloaded / elapsed if elapsed > 0 else 0
                                remaining = total_size - downloaded
                                eta = remaining / speed if speed > 0 else 0

                                progress = DownloadProgress(
                                    bytes_downloaded=downloaded,
                                    total_bytes=total_size,
                                    speed_bps=speed,
                                    eta_seconds=eta,
                                    current_file=model.description,
                                    filename=model.name,
                                    status="downloading",
                                )
                                self._report_progress(progress)
                                last_report = current_time

                    # Download complete, rename temp to final
                    if filepath.exists():
                        filepath.unlink()
                    temp_filepath.rename(filepath)

                    # Verify file size
                    if filepath.stat().st_size < model.expected_size * 0.95:
                        return False, f"Downloaded file too small: {filepath.stat().st_size} bytes"

                    progress = DownloadProgress(
                        bytes_downloaded=downloaded,
                        total_bytes=total_size,
                        status="verifying",
                        current_file=model.description,
                        filename=model.name,
                    )
                    self._report_progress(progress)

                    return True, "Download complete"

            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
                resume_from = temp_filepath.stat().st_size if temp_filepath.exists() else 0
                if attempt < self._retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False, f"Download failed after {self._retry_count} attempts: {e}"

            except Exception as e:
                return False, f"Unexpected error: {e}"

        return False, "Download failed"

    def download_all(self, models: Optional[list] = None) -> bool:
        """Download all model files."""
        if models is None:
            models = QWEN_MODEL_FILES

        self._cancelled = False
        total_files = len(models)
        overall_downloaded = 0
        overall_total = sum(m.expected_size for m in models)

        for i, model in enumerate(models):
            if self._cancelled:
                return False

            progress = DownloadProgress(
                file_index=i + 1,
                total_files=total_files,
                current_file=model.description,
                filename=model.name,
                status="downloading",
            )
            self._report_progress(progress)

            success, message = self._download_file(model)

            if not success and not self._cancelled:
                progress = DownloadProgress(
                    file_index=i + 1,
                    total_files=total_files,
                    current_file=model.description,
                    filename=model.name,
                    status="error",
                    error_message=message,
                )
                self._report_progress(progress)
                return False

        if not self._cancelled:
            progress = DownloadProgress(
                status="complete",
                total_files=total_files,
            )
            self._report_progress(progress)

        return True

    def verify_models(self, models: Optional[list] = None) -> Tuple[bool, List[str]]:
        """Verify all model files exist and are valid."""
        if models is None:
            models = QWEN_MODEL_FILES

        errors = []
        for model in models:
            filepath = self.models_dir / model.name
            if not filepath.exists():
                errors.append(f"Missing: {model.name}")
            elif filepath.stat().st_size < model.expected_size * 0.95:
                errors.append(f"Incomplete: {model.name} ({filepath.stat().st_size} / {model.expected_size} bytes)")

        return len(errors) == 0, errors

    def get_total_size(self, models: Optional[list] = None) -> int:
        """Get total download size in bytes."""
        if models is None:
            models = QWEN_MODEL_FILES
        return sum(m.expected_size for m in models)

    def get_downloaded_size(self, models: Optional[list] = None) -> int:
        """Get total downloaded size in bytes."""
        if models is None:
            models = QWEN_MODEL_FILES

        total = 0
        for model in models:
            filepath = self.models_dir / model.name
            if filepath.exists():
                total += filepath.stat().st_size
        return total


if __name__ == "__main__":
    import sys
    downloader = ModelDownloader("models")
    total = downloader.get_total_size()
    downloaded = downloader.get_downloaded_size()
    print(f"Total: {format_bytes(total)}")
    print(f"Downloaded: {format_bytes(downloaded)}")
    print(f"Remaining: {format_bytes(total - downloaded)}")
