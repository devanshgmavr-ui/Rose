"""Tests for Stage 5.1 - Controlled Application Launching."""

import pytest
import subprocess
from unittest.mock import MagicMock, patch

from agent.tools.app_launcher import AppLauncher, LaunchAppTool, LaunchMode, LaunchResult
from agent.tools.base import Permission


class TestLaunchResult:
    def test_success(self):
        r = LaunchResult(success=True, app_name="test.exe", pid=1234)
        assert r.success is True
        assert r.pid == 1234

    def test_failure(self):
        r = LaunchResult(success=False, app_name="test.exe", error="not found")
        assert r.success is False
        assert r.error == "not found"

    def test_to_dict(self):
        r = LaunchResult(success=True, app_name="test.exe", pid=100, launch_time=1.5)
        d = r.to_dict()
        assert d["success"] is True
        assert d["pid"] == 100
        assert d["launch_time"] == 1.5

    def test_to_text_success(self):
        r = LaunchResult(success=True, app_name="test.exe", pid=100)
        assert "Launched test.exe" in r.to_text()
        assert "100" in r.to_text()

    def test_to_text_failure(self):
        r = LaunchResult(success=False, app_name="test.exe", error="fail")
        assert "Failed" in r.to_text()
        assert "fail" in r.to_text()


class TestAppLauncher:
    def test_init(self):
        launcher = AppLauncher()
        assert launcher._default_timeout == 30.0

    def test_validate_path_empty(self):
        launcher = AppLauncher()
        assert launcher._validate_path("") is False
        assert launcher._validate_path(None) is False

    def test_validate_path_bad_ext(self):
        launcher = AppLauncher()
        assert launcher._validate_path("test.exe") is True
        assert launcher._validate_path("test.dll") is False
        assert launcher._validate_path("test.py") is True

    def test_validate_path_blocked(self):
        launcher = AppLauncher(blocked_apps=["malware.exe"])
        assert launcher._validate_path("malware.exe") is False
        assert launcher._validate_path("good.exe") is True

    def test_validate_path_allowed(self):
        launcher = AppLauncher(allowed_apps=["notepad.exe", "calc.exe"])
        assert launcher._validate_path("notepad.exe") is True
        assert launcher._validate_path("unknown.exe") is False

    def test_launch_not_found(self):
        launcher = AppLauncher()
        r = launcher.launch("C:\\nonexistent\\fake.exe")
        assert r.success is False
        assert "not found" in r.error

    def test_launch_bad_path(self):
        launcher = AppLauncher()
        r = launcher.launch("")
        assert r.success is False

    def test_launch_bad_ext(self):
        launcher = AppLauncher()
        r = launcher.launch("C:\\test\\malware.dll")
        assert r.success is False

    def test_launch_by_name_not_found(self):
        launcher = AppLauncher()
        r = launcher.launch_by_name("nonexistent_app_xyz_123")
        assert r.success is False
        assert "not found" in r.error

    @patch("agent.tools.app_launcher.shutil.which", return_value="C:\\Python\\python.exe")
    @patch("agent.tools.app_launcher.os.path.exists", return_value=True)
    def test_launch_by_name_found(self, mock_exists, mock_which):
        launcher = AppLauncher()
        with patch("agent.tools.app_launcher.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            r = launcher.launch_by_name("python", args=["--version"], mode=LaunchMode.WAIT)
            assert r.success is True
            assert r.exit_code == 0

    def test_is_running_unknown(self):
        launcher = AppLauncher()
        assert launcher.is_running(99999) is False

    def test_get_exit_code_unknown(self):
        launcher = AppLauncher()
        assert launcher.get_exit_code(99999) is None

    def test_terminate_unknown(self):
        launcher = AppLauncher()
        assert launcher.terminate(99999) is False

    def test_list_running_empty(self):
        launcher = AppLauncher()
        assert launcher.list_running() == []

    @patch("agent.tools.app_launcher.os.path.exists", return_value=True)
    def test_launch_wait_mode(self, mock_exists):
        launcher = AppLauncher()
        with patch("agent.tools.app_launcher.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            r = launcher.launch("python.exe", args=["--version"], mode=LaunchMode.WAIT)
            assert r.success is True
            assert r.exit_code == 0

    @patch("agent.tools.app_launcher.os.path.exists", return_value=True)
    def test_launch_detached(self, mock_exists):
        launcher = AppLauncher()
        with patch("agent.tools.app_launcher.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1234
            mock_popen.return_value = mock_proc
            r = launcher.launch("python.exe", args=["--version"])
            assert r.success is True

    @patch("agent.tools.app_launcher.os.path.exists", return_value=True)
    def test_launch_custom_timeout(self, mock_exists):
        launcher = AppLauncher()
        with patch("agent.tools.app_launcher.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.pid = 1234
            mock_popen.return_value = mock_proc
            r = launcher.launch("python.exe", args=["--version"], timeout=10.0)
            assert r.success is True


class TestLaunchAppTool:
    def test_tool_properties(self):
        tool = LaunchAppTool()
        assert tool.name == "launch_app"
        assert tool.category == "os_automation"
        assert "app_path" in tool.parameters["properties"]

    def test_tool_permissions(self):
        tool = LaunchAppTool()
        perms = tool.required_permissions
        assert Permission.APP_LAUNCH in perms

    def test_tool_execute_no_path(self):
        tool = LaunchAppTool()
        r = tool.execute()
        assert r.success is False

    def test_tool_execute_not_found(self):
        tool = LaunchAppTool()
        r = tool.execute(app_path="C:\\fake\\app.exe")
        assert r.success is False

    @patch("agent.tools.app_launcher.os.path.exists", return_value=True)
    def test_tool_execute_python(self, mock_exists):
        tool = LaunchAppTool()
        with patch("agent.tools.app_launcher.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            r = tool.execute(app_path="python.exe", args=["--version"], mode="wait")
            assert r.success is True
