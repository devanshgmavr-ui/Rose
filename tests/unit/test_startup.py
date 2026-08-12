"""Tests for Phase 7 - Startup and End-to-End."""

import os
import json
import time
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent.startup import (
    StartupConfig, StartupReport, FirstRunDetector, DependencyValidator,
    ModelDiscovery, ConfigGenerator, StartupManager, DependencyCheck,
)


class TestStartupConfig:
    def test_defaults(self):
        cfg = StartupConfig()
        assert cfg.log_level == "INFO"
        assert cfg.headless is False
        assert cfg.web_mode is False
        assert cfg.web_port == 8080
        assert cfg.skip_model_check is False

    def test_to_dict(self):
        cfg = StartupConfig(web_port=9090, headless=True)
        d = cfg.to_dict()
        assert d["web_port"] == 9090
        assert d["headless"] is True


class TestDependencyCheck:
    def test_creation(self):
        dep = DependencyCheck(name="test", installed=True, version="1.0")
        assert dep.name == "test"
        assert dep.installed is True

    def test_to_dict(self):
        dep = DependencyCheck(name="test", installed=False, error="not found")
        d = dep.to_dict()
        assert d["name"] == "test"
        assert d["installed"] is False
        assert d["error"] == "not found"


class TestStartupReport:
    def test_defaults(self):
        report = StartupReport()
        assert report.success is False
        assert report.first_run is False
        assert report.model_found is False

    def test_to_dict(self):
        report = StartupReport(success=True, model_path="/path/to/model.gguf")
        d = report.to_dict()
        assert d["success"] is True
        assert d["model_path"] == "/path/to/model.gguf"


class TestFirstRunDetector:
    def test_is_first_run(self, tmp_path):
        detector = FirstRunDetector(str(tmp_path / "rose"))
        assert detector.is_first_run() is True

    def test_mark_complete(self, tmp_path):
        detector = FirstRunDetector(str(tmp_path / "rose"))
        detector.mark_complete()
        assert detector.is_first_run() is False

    def test_get_state(self, tmp_path):
        detector = FirstRunDetector(str(tmp_path / "rose"))
        state = detector.get_state()
        assert state["first_run"] is True
        assert "workspace" in state

    def test_persistence(self, tmp_path):
        detector1 = FirstRunDetector(str(tmp_path / "rose"))
        detector1.mark_complete()
        detector2 = FirstRunDetector(str(tmp_path / "rose"))
        assert detector2.is_first_run() is False


class TestDependencyValidator:
    def test_check_all(self):
        validator = DependencyValidator()
        results = validator.check_all()
        assert len(results) > 0

    def test_has_pil(self):
        validator = DependencyValidator()
        results = validator.check_all()
        pil_dep = next((d for d in results if d.name == "PIL"), None)
        assert pil_dep is not None
        assert pil_dep.installed is True

    def test_has_numpy(self):
        validator = DependencyValidator()
        results = validator.check_all()
        numpy_dep = next((d for d in results if d.name == "numpy"), None)
        assert numpy_dep is not None
        assert numpy_dep.installed is True


class TestModelDiscovery:
    def test_find_models_empty(self, tmp_path):
        discovery = ModelDiscovery(str(tmp_path / "models"))
        models = discovery.find_models()
        assert len(models) == 0

    def test_find_models_with_gguf(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "test_model.gguf").write_bytes(b"0" * (1024 * 1024))
        discovery = ModelDiscovery(str(models_dir))
        models = discovery.find_models()
        assert len(models) == 1
        assert models[0]["name"] == "test_model"

    def test_find_best_model(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "qwen_coder.gguf").write_bytes(b"0" * (1024 * 1024))
        (models_dir / "other_model.gguf").write_bytes(b"0" * (1024 * 1024))
        discovery = ModelDiscovery(str(models_dir))
        best = discovery.find_best_model()
        assert best is not None
        assert "qwen" in best["name"]

    def test_validate_model(self, tmp_path):
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"0" * (20 * 1024 * 1024))  # 20MB
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model(str(model_file))
        assert valid is True
        assert "Valid model" in msg

    def test_validate_model_not_found(self, tmp_path):
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model(str(tmp_path / "nonexistent.gguf"))
        assert valid is False

    def test_validate_model_wrong_extension(self, tmp_path):
        model_file = tmp_path / "model.txt"
        model_file.write_bytes(b"0" * (1024 * 1024))
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model(str(model_file))
        assert valid is False

    def test_validate_model_too_small(self, tmp_path):
        model_file = tmp_path / "tiny.gguf"
        model_file.write_bytes(b"0" * (5 * 1024 * 1024))  # 5MB
        discovery = ModelDiscovery()
        valid, msg = discovery.validate_model(str(model_file))
        assert valid is False


class TestConfigGenerator:
    def test_generate_default(self, tmp_path):
        gen = ConfigGenerator(str(tmp_path / "rose"))
        config = gen.generate_default()
        assert "agent" in config
        assert "llm" in config
        assert "security" in config

    def test_save_load_config(self, tmp_path):
        gen = ConfigGenerator(str(tmp_path / "rose"))
        config = gen.generate_default()
        path = gen.save_config(config)
        assert Path(path).exists()
        loaded = gen.load_config()
        assert loaded is not None
        assert loaded["agent"]["name"] == "Rose"

    def test_ensure_config_creates(self, tmp_path):
        gen = ConfigGenerator(str(tmp_path / "rose"))
        config = gen.ensure_config()
        assert config is not None
        assert gen.load_config() is not None

    def test_ensure_config_loads_existing(self, tmp_path):
        gen = ConfigGenerator(str(tmp_path / "rose"))
        config1 = gen.ensure_config()
        config1["agent"]["name"] = "Custom"
        gen.save_config(config1)
        config2 = gen.ensure_config()
        assert config2["agent"]["name"] == "Custom"

    def test_load_nonexistent(self, tmp_path):
        gen = ConfigGenerator(str(tmp_path / "rose"))
        assert gen.load_config("nonexistent.json") is None


class TestStartupManager:
    def test_run_startup(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            models_dir=str(tmp_path / "models"),
            skip_model_check=True,
            skip_dependency_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert report.success is True
        assert report.config_generated is True

    def test_run_startup_first_run(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            skip_model_check=True,
            skip_dependency_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert report.first_run is True
        assert report.success is True

    def test_run_startup_with_model(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "qwen.gguf").write_bytes(b"0" * (1024 * 1024))
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            models_dir=str(models_dir),
            skip_dependency_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert report.model_found is True
        assert "qwen" in report.model_path.lower()

    def test_get_quick_status(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            skip_model_check=True,
        )
        manager = StartupManager(config)
        status = manager.get_quick_status()
        assert "first_run" in status
        assert "deps_ok" in status

    def test_platform_info(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            skip_model_check=True,
            skip_dependency_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert "system" in report.platform_info
        assert "python_version" in report.platform_info

    def test_dependencies_in_report(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            skip_model_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert len(report.dependencies) > 0

    def test_run_startup_dependency_check(self, tmp_path):
        config = StartupConfig(
            workspace_dir=str(tmp_path / "rose"),
            skip_model_check=True,
        )
        manager = StartupManager(config)
        report = manager.run_startup()
        assert len(report.dependencies) > 0
        assert any(d.name == "PIL" for d in report.dependencies)


class TestRunPy:
    """Test run.py module can be imported."""

    def test_import(self):
        import run
        assert hasattr(run, "main")
        assert hasattr(run, "cmd_run")
        assert hasattr(run, "cmd_status")

    def test_print_banner(self, capsys):
        from run import print_banner
        print_banner()
        captured = capsys.readouterr()
        assert "Rose" in captured.out

    def test_print_help(self, capsys):
        from run import _print_help
        _print_help()
        captured = capsys.readouterr()
        assert "help" in captured.out.lower()
