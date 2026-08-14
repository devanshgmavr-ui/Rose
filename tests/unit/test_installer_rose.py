"""Installer-specific tests for Rose."""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "installer"))

from installer.system_check import (
    get_system_info, get_check_results, detect_gpu, check_internet, SystemInfo, GPUInfo
)
from installer.model_downloader import (
    ModelDownloader, ModelFile, QWEN_MODEL_FILES,
    format_bytes, format_speed, format_eta
)


class TestSystemCheck:
    """Test system detection functions."""

    def test_get_system_info_returns_system_info(self):
        info = get_system_info()
        assert isinstance(info, SystemInfo)

    def test_system_info_has_windows_version(self):
        info = get_system_info()
        assert info.windows_version != ""

    def test_system_info_has_architecture(self):
        info = get_system_info()
        assert info.architecture in ("AMD64", "x86_64", "ARM64", "Windows AMD64")

    def test_system_info_has_ram(self):
        info = get_system_info()
        assert info.ram_gb > 0

    def test_system_info_has_cpu_cores(self):
        info = get_system_info()
        assert info.cpu_cores > 0

    def test_system_info_has_disk_space(self):
        info = get_system_info()
        assert info.free_disk_gb >= 0
        assert info.total_disk_gb >= 0

    def test_get_check_results(self):
        info = get_system_info()
        checks = get_check_results(info)
        assert isinstance(checks, list)
        assert len(checks) >= 5  # At least Windows, Arch, CPU, RAM, Disk, Internet

    def test_check_result_format(self):
        info = get_system_info()
        checks = get_check_results(info)
        for check in checks:
            assert "label" in check
            assert "detail" in check
            assert "status" in check
            assert check["status"] in ("pass", "warn", "fail")


class TestGPU:
    """Test GPU detection."""

    def test_detect_gpu_returns_gpu_info(self):
        gpu = detect_gpu()
        assert isinstance(gpu, GPUInfo)

    def test_gpu_has_name(self):
        gpu = detect_gpu()
        assert gpu.name != ""


class TestModelDownloader:
    """Test model download utilities."""

    def test_format_bytes(self):
        assert format_bytes(0) == "0 B"
        assert format_bytes(1023) == "1023 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024**2) == "1.0 MB"
        assert format_bytes(1024**3) == "1.00 GB"

    def test_format_speed(self):
        assert format_speed(0) == "0 B/s"
        assert format_speed(500) == "500 B/s"
        assert format_speed(1024) == "1.0 KB/s"
        assert format_speed(1024**2) == "1.0 MB/s"

    def test_format_eta(self):
        assert format_eta(30) == "30s"
        assert format_eta(90) == "1m 30s"
        assert format_eta(3661) == "1h 1m"

    def test_qwen_model_files_defined(self):
        assert len(QWEN_MODEL_FILES) == 2
        names = [m.name for m in QWEN_MODEL_FILES]
        assert "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" in names
        assert "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf" in names

    def test_model_file_sizes(self):
        for model in QWEN_MODEL_FILES:
            assert model.expected_size > 0
            assert model.expected_size > 1_000_000_000  # At least 1GB each

    def test_downloader_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = ModelDownloader(tmpdir)
            assert downloader.models_dir.exists()

    def test_downloader_get_total_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = ModelDownloader(tmpdir)
            total = downloader.get_total_size()
            assert total > 5_000_000_000  # ~5.6 GB total

    def test_downloader_verify_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            downloader = ModelDownloader(tmpdir)
            valid, errors = downloader.verify_models()
            assert not valid
            assert len(errors) == 2


class TestInstallerFiles:
    """Test that installer files exist and are valid."""

    def test_installer_dir_exists(self):
        assert (ROOT / "installer").is_dir()

    def test_rose_installer_exists(self):
        assert (ROOT / "installer" / "rose_installer.py").is_file()

    def test_rose_gui_exists(self):
        assert (ROOT / "installer" / "rose_gui.py").is_file()

    def test_system_check_exists(self):
        assert (ROOT / "installer" / "system_check.py").is_file()

    def test_model_downloader_exists(self):
        assert (ROOT / "installer" / "model_downloader.py").is_file()

    def test_build_script_exists(self):
        assert (ROOT / "installer" / "build.py").is_file()

    def test_pyinstaller_spec_exists(self):
        assert (ROOT / "installer" / "rose_gui.spec").is_file()

    def test_inno_setup_script_exists(self):
        assert (ROOT / "installer" / "rose_setup.iss").is_file()

    def test_installer_syntax_valid(self):
        import py_compile
        py_compile.compile(str(ROOT / "installer" / "rose_installer.py"), doraise=True)

    def test_gui_syntax_valid(self):
        import py_compile
        py_compile.compile(str(ROOT / "installer" / "rose_gui.py"), doraise=True)

    def test_system_check_syntax_valid(self):
        import py_compile
        py_compile.compile(str(ROOT / "installer" / "system_check.py"), doraise=True)

    def test_model_downloader_syntax_valid(self):
        import py_compile
        py_compile.compile(str(ROOT / "installer" / "model_downloader.py"), doraise=True)

    def test_build_script_syntax_valid(self):
        import py_compile
        py_compile.compile(str(ROOT / "installer" / "build.py"), doraise=True)


class TestInstallerModules:
    """Test that installer modules can be imported."""

    def test_import_system_check(self):
        from installer import system_check
        assert hasattr(system_check, "get_system_info")

    def test_import_model_downloader(self):
        from installer import model_downloader
        assert hasattr(model_downloader, "ModelDownloader")


class TestLicense:
    """Test license file exists and is valid."""

    def test_license_exists(self):
        assert (ROOT / "LICENSE").is_file()

    def test_license_is_mit(self):
        content = (ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "MIT License" in content
        assert "Devansh Gupta" in content


class TestRequirements:
    """Test requirements files are complete."""

    def test_runtime_requirements_exist(self):
        assert (ROOT / "requirements-runtime.txt").is_file()

    def test_runtime_has_llama_cpp(self):
        content = (ROOT / "requirements-runtime.txt").read_text()
        assert "llama-cpp-python" in content

    def test_runtime_has_pillow(self):
        content = (ROOT / "requirements-runtime.txt").read_text()
        assert "Pillow" in content

    def test_runtime_has_numpy(self):
        content = (ROOT / "requirements-runtime.txt").read_text()
        assert "numpy" in content

    def test_dev_requirements_exist(self):
        assert (ROOT / "requirements-dev.txt").is_file()

    def test_dev_has_pytest(self):
        content = (ROOT / "requirements-dev.txt").read_text()
        assert "pytest" in content
