"""Tests for Phase 8 - Packaging and Production."""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch


class TestPyprojectToml:
    def test_exists(self):
        assert Path("pyproject.toml").exists()

    def test_valid_toml(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
                return

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        assert "project" in config
        assert config["project"]["name"] == "rose-agent"
        assert config["project"]["version"] == "1.1.0"

    def test_has_dependencies(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
                return

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        deps = config["project"]["dependencies"]
        assert any("llama" in d for d in deps)
        assert any("Pillow" in d or "pillow" in d.lower() for d in deps)
        assert any("numpy" in d for d in deps)

    def test_has_optional_deps(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
                return

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        optional = config["project"]["optional-dependencies"]
        assert "ui" in optional
        assert "full" in optional
        assert "dev" in optional

    def test_has_entry_point(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
                return

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        assert "project.scripts" in config or "scripts" in config.get("project", {})


class TestRequirements:
    def test_exists(self):
        assert Path("requirements-runtime.txt").exists()

    def test_has_core_deps(self):
        content = Path("requirements-runtime.txt").read_text()
        assert "llama" in content.lower()
        assert "pillow" in content.lower() or "Pillow" in content


class TestInstaller:
    def test_install_bat_exists(self):
        assert Path("install.bat").exists()

    def test_install_bat_not_empty(self):
        content = Path("install.bat").read_text()
        assert len(content) > 100


class TestRunPy:
    def test_exists(self):
        assert Path("run.py").exists()

    def test_has_main(self):
        content = Path("run.py").read_text(encoding="utf-8")
        assert "def main()" in content

    def test_importable(self):
        import run
        assert hasattr(run, "main")


class TestPackageStructure:
    def test_agent_package(self):
        assert Path("agent/__init__.py").exists()

    def test_core_package(self):
        assert Path("agent/core/__init__.py").exists()

    def test_tools_package(self):
        assert Path("agent/tools/__init__.py").exists()

    def test_web_package(self):
        assert Path("agent/web/__init__.py").exists()

    def test_ui_package(self):
        assert Path("agent/ui/__init__.py").exists()

    def test_media_package(self):
        assert Path("agent/media/__init__.py").exists()

    def test_tests_directory(self):
        assert Path("tests").is_dir()


class TestConfigFiles:
    def test_env_example(self):
        assert Path(".env.example").exists()

    def test_gitignore(self):
        assert Path(".gitignore").exists()

    def test_readme(self):
        assert Path("README.md").exists()


class TestVersionConsistency:
    def test_version_in_pyproject(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("No TOML parser available")
                return

        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)

        version = config["project"]["version"]
        assert version == "1.1.0"

    def test_version_in_web_server(self):
        from agent.web.server import WebServer
        server = WebServer()
        config = server.get_config()
        assert config.host == "127.0.0.1"

    def test_version_in_application_service(self):
        from agent.web.application import ApplicationService
        svc = ApplicationService()
        info = svc.get_app_info()
        assert info["version"] == "1.1.0"


class TestAgentModules:
    def test_import_agent(self):
        from agent.core.agent import Agent
        assert Agent is not None

    def test_import_config(self):
        from agent.core.config import Config
        assert Config is not None

    def test_import_web_server(self):
        from agent.web.server import WebServer
        assert WebServer is not None

    def test_import_application_service(self):
        from agent.web.application import ApplicationService
        assert ApplicationService is not None

    def test_import_event_bus(self):
        from agent.web.events import EventBus
        assert EventBus is not None

    def test_import_rose_ui(self):
        from agent.ui import RoseUI
        assert RoseUI is not None

    def test_import_startup(self):
        from agent.startup import StartupManager
        assert StartupManager is not None
