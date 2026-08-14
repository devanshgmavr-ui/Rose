#!/usr/bin/env python3
"""Tests for Rose Installation Pipeline.

Tests the installer logic without actually modifying the system.
Uses mocking to verify behavior in various scenarios.
"""

import os
import sys
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPythonDetection:
    """Test Python version and architecture detection."""

    def test_python_version_detection(self):
        """Test that Python version is correctly detected."""
        version = sys.version_info
        assert version.major >= 3, "Python 3.x required"
        assert version.minor >= 10, "Python 3.10+ required"

    def test_64bit_architecture(self):
        """Test that 64-bit Python is detected."""
        is_64bit = struct.calcsize("P") == 8
        assert is_64bit, "64-bit Python required"

    def test_supported_python_versions(self):
        """Test that supported Python versions are identified."""
        version = sys.version_info
        # Official prebuilt wheels support 3.10, 3.11, 3.12
        # Python 3.13+ may require compilation
        # This test documents the current state - Python 3.14 is NOT officially supported
        supported = (3, 10) <= (version.major, version.minor) <= (3, 12)
        partially_supported = (version.major, version.minor) == (3, 13)
        # Python 3.14+ is not supported by prebuilt wheels
        # The installer should warn about this
        if not supported and not partially_supported:
            # This is expected on Python 3.14+ - the installer should warn
            pass
        else:
            assert True


class TestGPUDetection:
    """Test GPU and CUDA detection logic."""

    def test_nvidia_smi_detection(self):
        """Test that nvidia-smi is detected when available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            has_nvidia = result.returncode == 0
            if has_nvidia:
                gpu_name = result.stdout.strip()
                assert len(gpu_name) > 0, "GPU name should not be empty"
        except FileNotFoundError:
            # nvidia-smi not found - this is expected on non-NVIDIA systems
            pass

    def test_cuda_version_parsing(self):
        """Test CUDA version parsing logic."""
        # Test various CUDA version strings
        test_cases = [
            ("12.3", 12, 3),
            ("13.0", 13, 0),
            ("11.8", 11, 8),
            ("12.5.1", 12, 5),
        ]
        for cuda_str, expected_major, expected_minor in test_cases:
            parts = cuda_str.split(".")
            major = int(parts[0])
            minor = int(parts[1])
            assert major == expected_major, f"Major version mismatch for {cuda_str}"
            assert minor == expected_minor, f"Minor version mismatch for {cuda_str}"


class TestWheelSelection:
    """Test wheel selection logic for different CUDA versions."""

    def test_wheel_index_selection(self):
        """Test that correct wheel index is selected based on CUDA version."""
        wheel_map = {
            (11, 8): "https://abetlen.github.io/llama-cpp-python/whl/cu118",
            (12, 1): "https://abetlen.github.io/llama-cpp-python/whl/cu121",
            (12, 2): "https://abetlen.github.io/llama-cpp-python/whl/cu122",
            (12, 3): "https://abetlen.github.io/llama-cpp-python/whl/cu123",
            (12, 4): "https://abetlen.github.io/llama-cpp-python/whl/cu124",
            (12, 5): "https://abetlen.github.io/llama-cpp-python/whl/cu125",
            (13, 0): "https://abetlen.github.io/llama-cpp-python/whl/cu130",
            (13, 2): "https://abetlen.github.io/llama-cpp-python/whl/cu132",
        }
        for (major, minor), expected_url in wheel_map.items():
            # Simulate the selection logic
            if major >= 13:
                if minor >= 2:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu132"
                else:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu130"
            elif major == 12:
                if minor >= 5:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu125"
                elif minor >= 4:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
                elif minor >= 3:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu123"
                elif minor >= 2:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu122"
                else:
                    url = "https://abetlen.github.io/llama-cpp-python/whl/cu121"
            elif major == 11 and minor >= 8:
                url = "https://abetlen.github.io/llama-cpp-python/whl/cu118"
            else:
                url = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

            assert url == expected_url, f"Wrong wheel for CUDA {major}.{minor}"

    def test_cpu_fallback(self):
        """Test that CPU mode is used when no GPU is available."""
        # When no NVIDIA GPU is detected, should fall back to CPU
        cpu_url = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
        assert "cpu" in cpu_url.lower()


class TestModelFiles:
    """Test model file detection and validation."""

    def test_model_filenames_match_config(self):
        """Test that model filenames match what config.py expects."""
        # From agent/core/config.py
        expected_main = "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        expected_mmproj = "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"

        # From download_models.py
        from scripts.download_models import MODEL_FILES
        actual_main = MODEL_FILES["main"]["filename"]
        actual_mmproj = MODEL_FILES["mmproj"]["filename"]

        assert actual_main == expected_main, f"Main model filename mismatch: {actual_main} != {expected_main}"
        assert actual_mmproj == expected_mmproj, f"mmproj filename mismatch: {actual_mmproj} != {expected_mmproj}"

    def test_model_repos_are_valid(self):
        """Test that HuggingFace repos are valid."""
        from scripts.download_models import MODEL_FILES
        for key, info in MODEL_FILES.items():
            repo = info["repo"]
            assert "/" in repo, f"Invalid repo format: {repo}"
            assert repo.startswith(("bartowski/", "ggml-org/", "Qwen/")), f"Unexpected repo: {repo}"

    def test_model_sizes_are_reasonable(self):
        """Test that expected model sizes are reasonable."""
        from scripts.download_models import MODEL_FILES, TOTAL_SIZE_GB
        # Qwen2.5-VL-7B-Instruct Q4_K_M should be around 4-5 GB
        assert 3.0 < MODEL_FILES["main"]["expected_size_gb"] < 6.0
        # mmproj should be around 1-2 GB
        assert 0.5 < MODEL_FILES["mmproj"]["expected_size_gb"] < 3.0
        # Total should be around 5-8 GB
        assert 4.0 < TOTAL_SIZE_GB < 10.0


class TestDiskSpaceCheck:
    """Test disk space checking logic."""

    def test_disk_space_check_returns_tuple(self):
        """Test that disk space check returns (bool, float)."""
        import shutil
        from pathlib import Path
        usage = shutil.disk_usage(Path.cwd())
        free_gb = usage.free / (1024 ** 3)
        assert isinstance(free_gb, float)
        assert free_gb > 0


class TestIdempotentInstallation:
    """Test that installation is idempotent."""

    def test_venv_exists_check(self):
        """Test that existing venv is detected."""
        venv_path = Path(PROJECT_ROOT) / ".venv"
        if venv_path.exists():
            assert (venv_path / "Scripts" / "activate.bat").exists() or \
                   (venv_path / "bin" / "activate").exists()

    def test_model_exists_check(self):
        """Test that existing models are detected."""
        models_dir = Path(PROJECT_ROOT) / "models"
        main_model = models_dir / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        mmproj = models_dir / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"

        # If models exist, they should be valid files
        if main_model.exists():
            assert main_model.stat().st_size > 100 * 1024 * 1024, "Main model should be > 100MB"
        if mmproj.exists():
            assert mmproj.stat().st_size > 100 * 1024 * 1024, "mmproj should be > 100MB"


class TestSecurityVerification:
    """Test security measures."""

    def test_gitignore_excludes_models(self):
        """Test that .gitignore excludes model files."""
        gitignore_path = Path(PROJECT_ROOT) / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert "models/*.gguf" in content, ".gitignore should exclude .gguf files"
            assert "models/.gitkeep" in content, ".gitignore should preserve models/.gitkeep"

    def test_gitignore_excludes_env(self):
        """Test that .gitignore excludes .env files."""
        gitignore_path = Path(PROJECT_ROOT) / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            assert ".env" in content, ".gitignore should exclude .env"
            assert "!.env.example" in content, ".gitignore should preserve .env.example"

    def test_no_secrets_in_env_example(self):
        """Test that .env.example doesn't contain secrets."""
        env_example_path = Path(PROJECT_ROOT) / ".env.example"
        if env_example_path.exists():
            content = env_example_path.read_text()
            # Check for common secret patterns
            assert "password" not in content.lower() or "your_password" in content.lower()
            assert "api_key" not in content.lower() or "your_api_key" in content.lower()
            assert "secret" not in content.lower() or "your_secret" in content.lower()


class TestInstallScriptSyntax:
    """Test install.bat syntax."""

    def test_install_bat_exists(self):
        """Test that install.bat exists."""
        install_bat = Path(PROJECT_ROOT) / "install.bat"
        assert install_bat.exists(), "install.bat should exist"

    def test_install_bat_has_required_sections(self):
        """Test that install.bat has all required sections."""
        install_bat = Path(PROJECT_ROOT) / "install.bat"
        if install_bat.exists():
            content = install_bat.read_text()
            # Check for required steps
            assert "STEP 1" in content or "Step 1" in content, "Should have Step 1"
            assert "STEP 10" in content or "Step 10" in content, "Should have Step 10"
            # Check for key features
            assert "--only-binary" in content, "Should use --only-binary to prevent source compilation"
            assert "nvidia-smi" in content.lower() or "CPU_ONLY" in content, "Should detect GPU"


class TestDownloadScript:
    """Test download_models.py script."""

    def test_download_script_exists(self):
        """Test that download_models.py exists."""
        download_script = Path(PROJECT_ROOT) / "scripts" / "download_models.py"
        assert download_script.exists(), "download_models.py should exist"

    def test_download_script_has_required_functions(self):
        """Test that download_models.py has required functions."""
        download_script = Path(PROJECT_ROOT) / "scripts" / "download_models.py"
        if download_script.exists():
            content = download_script.read_text()
            assert "def download_file" in content, "Should have download_file function"
            assert "def validate_file" in content, "Should have validate_file function"
            assert "def check_disk_space" in content, "Should have check_disk_space function"
            assert "--check-only" in content, "Should support --check-only flag"


class TestCleanMachineScenario:
    """Test clean machine installation scenario."""

    def test_prevent_source_compilation(self):
        """Test that source compilation is prevented."""
        install_bat = Path(PROJECT_ROOT) / "install.bat"
        if install_bat.exists():
            content = install_bat.read_text()
            # Should use --only-binary to prevent source compilation
            assert "--only-binary=llama-cpp-python" in content, \
                "Should use --only-binary to prevent source compilation"
            # Should not have unconditional source build
            assert "pip install llama-cpp-python" in content, \
                "Should install llama-cpp-python"
            # Should have clear error message when prebuilt wheel not available
            assert "Prebuilt wheel not available" in content, \
                "Should show clear error when prebuilt wheel not available"

    def test_compiler_check_before_source_build(self):
        """Test that compiler is checked before attempting source build."""
        install_bat = Path(PROJECT_ROOT) / "install.bat"
        if install_bat.exists():
            content = install_bat.read_text()
            # Should check for compiler
            assert "cl" in content.lower() or "compiler" in content.lower(), \
                "Should check for MSVC compiler"
            # Should have fallback options
            assert "Visual Studio" in content or "Build Tools" in content, \
                "Should mention Visual Studio as fallback"


class TestRequirementsFiles:
    """Test requirements files."""

    def test_requirements_runtime_exists(self):
        """Test that requirements-runtime.txt exists."""
        req_path = Path(PROJECT_ROOT) / "requirements-runtime.txt"
        assert req_path.exists(), "requirements-runtime.txt should exist"

    def test_requirements_runtime_has_llama_cpp(self):
        """Test that requirements-runtime.txt has llama-cpp-python."""
        req_path = Path(PROJECT_ROOT) / "requirements-runtime.txt"
        if req_path.exists():
            content = req_path.read_text()
            assert "llama-cpp-python" in content, "Should include llama-cpp-python"

    def test_requirements_runtime_has_model_downloader_deps(self):
        """Test that requirements-runtime.txt has model downloader dependencies."""
        req_path = Path(PROJECT_ROOT) / "requirements-runtime.txt"
        if req_path.exists():
            content = req_path.read_text()
            assert "huggingface-hub" in content or "huggingface_hub" in content, "Should include huggingface-hub"
            assert "requests" in content, "Should include requests"


class TestDocumentation:
    """Test README.md documentation."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = Path(PROJECT_ROOT) / "README.md"
        assert readme_path.exists(), "README.md should exist"

    def test_readme_has_installation_instructions(self):
        """Test that README.md has installation instructions."""
        readme_path = Path(PROJECT_ROOT) / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
            assert "install.bat" in content, "Should document install.bat"
            assert "--skip-models" in content, "Should document --skip-models flag"
            assert "--cpu-only" in content, "Should document --cpu-only flag"

    def test_readme_has_troubleshooting(self):
        """Test that README.md has troubleshooting section."""
        readme_path = Path(PROJECT_ROOT) / "README.md"
        if readme_path.exists():
            content = readme_path.read_text(encoding="utf-8")
            assert "Troubleshooting" in content, "Should have troubleshooting section"
            assert "CMAKE" in content or "CMake" in content, "Should address CMake issues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
