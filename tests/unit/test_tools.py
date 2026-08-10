"""Unit tests for tool system (Stage 1.3)."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from agent.tools.base import (
    Permission,
    ConfirmationLevel,
    ToolRequest,
    ToolResult,
)
from agent.tools.registry import ToolRegistry
from agent.tools.permissions import PermissionManager
from agent.tools.router import ToolRouter
from agent.tools.audit import AuditLogger, AuditRecord
from agent.tools.filesystem_tool import FilesystemTool
from agent.tools.python_sandbox import PythonSandboxTool
from agent.tools.cli_tool import CLITool


class TestBaseDataClasses:
    """Test tool base data classes."""

    def test_tool_request_creation(self):
        req = ToolRequest(tool="test_tool", arguments={"key": "value"})
        assert req.tool == "test_tool"
        assert req.arguments == {"key": "value"}
        assert req.request_id is not None

    def test_tool_request_to_dict(self):
        req = ToolRequest(tool="test", arguments={"a": 1})
        d = req.to_dict()
        assert d["tool"] == "test"
        assert d["arguments"] == {"a": 1}

    def test_tool_request_from_dict(self):
        original = ToolRequest(tool="x", arguments={"b": 2})
        d = original.to_dict()
        restored = ToolRequest.from_dict(d)
        assert restored.tool == original.tool
        assert restored.arguments == original.arguments

    def test_tool_result_creation(self):
        result = ToolResult(success=True, tool_name="test", output="hello")
        assert result.success is True
        assert result.output == "hello"
        assert result.error == ""

    def test_tool_result_to_dict(self):
        result = ToolResult(success=False, tool_name="t", error="fail")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "fail"

    def test_tool_result_from_dict(self):
        original = ToolResult(success=True, tool_name="ok", output="data")
        d = original.to_dict()
        restored = ToolResult.from_dict(d)
        assert restored.success == original.success
        assert restored.output == original.output


class TestToolRegistry:
    """Test tool registry."""

    def test_register_tool(self):
        registry = ToolRegistry()
        tool = FilesystemTool()
        result = registry.register(tool)
        assert result is True
        assert registry.has("filesystem")

    def test_register_duplicate(self):
        registry = ToolRegistry()
        tool = FilesystemTool()
        registry.register(tool)
        result = registry.register(tool)
        assert result is False

    def test_unregister_tool(self):
        registry = ToolRegistry()
        tool = FilesystemTool()
        registry.register(tool)
        result = registry.unregister("filesystem")
        assert result is True
        assert not registry.has("filesystem")

    def test_get_tool(self):
        registry = ToolRegistry()
        tool = FilesystemTool()
        registry.register(tool)
        retrieved = registry.get("filesystem")
        assert retrieved is tool

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(FilesystemTool())
        registry.register(PythonSandboxTool())
        tools = registry.list_tools()
        assert len(tools) == 2

    def test_list_names(self):
        registry = ToolRegistry()
        registry.register(FilesystemTool())
        names = registry.list_names()
        assert "filesystem" in names

    def test_count(self):
        registry = ToolRegistry()
        assert registry.count() == 0
        registry.register(FilesystemTool())
        assert registry.count() == 1

    def test_clear(self):
        registry = ToolRegistry()
        registry.register(FilesystemTool())
        registry.clear()
        assert registry.count() == 0


class TestPermissionManager:
    """Test permission system."""

    def test_default_permissions(self):
        pm = PermissionManager()
        assert pm.has_permission("filesystem.read", "workspace")
        assert pm.has_permission("filesystem.write", "workspace")
        assert pm.has_permission("code.execute", "sandbox")

    def test_no_command_execute_by_default(self):
        pm = PermissionManager()
        assert not pm.has_permission("command.execute")

    def test_grant_permission(self):
        pm = PermissionManager()
        pm.grant_permission("command.execute")
        assert pm.has_permission("command.execute")

    def test_revoke_permission(self):
        pm = PermissionManager()
        pm.revoke_permission("filesystem.read")
        assert not pm.has_permission("filesystem.read", "workspace")

    def test_confirmation_level(self):
        pm = PermissionManager()
        level = pm.get_confirmation_level("filesystem.write")
        assert level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_check_tool_permissions(self):
        pm = PermissionManager()
        has, denied, needs_confirm = pm.check_tool_permissions(
            [Permission.FILESYSTEM_READ], "workspace"
        )
        assert has is True
        assert denied == []

    def test_check_tool_permissions_denied(self):
        pm = PermissionManager()
        has, denied, needs_confirm = pm.check_tool_permissions(
            [Permission.COMMAND_EXECUTE]
        )
        assert has is False
        assert "command.execute" in denied


class TestFilesystemTool:
    """Test filesystem tool with workspace boundary."""

    def test_name_and_description(self):
        tool = FilesystemTool()
        assert tool.name == "filesystem"
        assert "workspace" in tool.description.lower()

    def test_validate_list(self):
        tool = FilesystemTool()
        valid, errors = tool.validate({"action": "list"})
        assert valid is True

    def test_validate_read(self):
        tool = FilesystemTool()
        valid, errors = tool.validate({"action": "read", "path": "test.txt"})
        assert valid is True

    def test_validate_read_missing_path(self):
        tool = FilesystemTool()
        valid, errors = tool.validate({"action": "read"})
        assert valid is False

    def test_validate_traversal_blocked(self):
        tool = FilesystemTool()
        valid, errors = tool.validate({"action": "read", "path": "../../etc/passwd"})
        assert valid is False

    def test_validate_absolute_path_blocked(self):
        tool = FilesystemTool()
        valid, errors = tool.validate({"action": "read", "path": "C:/Windows/System32/config"})
        assert valid is False

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello")
            result = tool.execute({"action": "list"})
            assert result.success is True
            assert "test.txt" in result.output

    def test_read_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("hello world")
            result = tool.execute({"action": "read", "path": "test.txt"})
            assert result.success is True
            assert "hello world" in result.output

    def test_write_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            result = tool.execute({
                "action": "write", "path": "output.txt", "content": "test data"
            })
            assert result.success is True
            assert (Path(tmpdir) / "output.txt").read_text() == "test data"

    def test_read_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            result = tool.execute({"action": "read", "path": "nope.txt"})
            assert result.success is False

    def test_path_traversal_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            result = tool.execute({"action": "read", "path": "../../etc/passwd"})
            assert result.success is False

    def test_absolute_path_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = FilesystemTool(workspace_dir=tmpdir)
            result = tool.execute({"action": "read", "path": "C:/Windows/System32/config"})
            assert result.success is False


class TestPythonSandbox:
    """Test Python execution sandbox."""

    def test_name_and_description(self):
        tool = PythonSandboxTool()
        assert tool.name == "python_sandbox"
        assert "sandbox" in tool.description.lower()

    def test_validate_code(self):
        tool = PythonSandboxTool()
        valid, errors = tool.validate({"code": "print('hello')"})
        assert valid is True

    def test_validate_missing_code(self):
        tool = PythonSandboxTool()
        valid, errors = tool.validate({})
        assert valid is False

    def test_execute_simple_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PythonSandboxTool(workspace_dir=tmpdir)
            result = tool.execute({"code": "print(2 + 2)"})
            assert result.success is True
            assert "4" in result.output

    def test_execute_code_with_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PythonSandboxTool(workspace_dir=tmpdir)
            result = tool.execute({"code": "1/0"})
            assert result.success is False

    def test_execute_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PythonSandboxTool(workspace_dir=tmpdir)
            tool._timeout = 0.1
            result = tool.execute({"code": "import time; time.sleep(5)"})
            assert result.success is False
            assert "timed out" in result.error.lower()

    def test_execute_output_capture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = PythonSandboxTool(workspace_dir=tmpdir)
            result = tool.execute({"code": "print('hello'); print('world')"})
            assert "hello" in result.output
            assert "world" in result.output


class TestCLITool:
    """Test CLI tool with allowlist."""

    def test_name_and_description(self):
        tool = CLITool()
        assert tool.name == "cli"
        assert "disabled" in tool.description.lower()

    def test_disabled_by_default(self):
        tool = CLITool()
        assert tool.is_enabled() is False

    def test_validate_disabled(self):
        tool = CLITool()
        valid, errors = tool.validate({"command": "echo"})
        assert valid is False
        assert "disabled" in errors[0].lower()

    def test_validate_allowed_command(self):
        tool = CLITool()
        tool.enable()
        valid, errors = tool.validate({"command": "echo", "args": ["hello"]})
        assert valid is True

    def test_validate_blocked_command(self):
        tool = CLITool()
        tool.enable()
        valid, errors = tool.validate({"command": "rm"})
        assert valid is False

    def test_validate_dangerous_pattern(self):
        tool = CLITool()
        tool.enable()
        valid, errors = tool.validate({"command": "echo", "args": ["a; rm -rf /"]})
        assert valid is False

    def test_validate_path_traversal(self):
        tool = CLITool()
        tool.enable()
        valid, errors = tool.validate({"command": "dir", "args": ["../../"]})
        assert valid is False

    def test_execute_disabled(self):
        tool = CLITool()
        result = tool.execute({"command": "echo"})
        assert result.success is False

    def test_enable_disable(self):
        tool = CLITool()
        tool.enable()
        assert tool.is_enabled() is True
        tool.disable()
        assert tool.is_enabled() is False

    def test_get_allowed_commands(self):
        tool = CLITool()
        allowed = tool.get_allowed_commands()
        assert "echo" in allowed
        assert "dir" in allowed


class TestToolRouter:
    """Test tool router."""

    def _make_router(self, tmpdir):
        registry = ToolRegistry()
        fs_tool = FilesystemTool(workspace_dir=tmpdir)
        py_tool = PythonSandboxTool(workspace_dir=tmpdir)
        cli_tool = CLITool(workspace_dir=tmpdir)
        registry.register(fs_tool)
        registry.register(py_tool)
        registry.register(cli_tool)
        pm = PermissionManager()
        al = AuditLogger(log_dir=tmpdir)
        return ToolRouter(registry=registry, permission_manager=pm, audit_logger=al)

    def test_route_unknown_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = self._make_router(tmpdir)
            request = ToolRequest(tool="unknown", arguments={})
            result = router.route(request)
            assert result.success is False
            assert "Unknown tool" in result.error

    def test_route_invalid_arguments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = self._make_router(tmpdir)
            request = ToolRequest(tool="filesystem", arguments={"action": "read"})
            result = router.route(request)
            assert result.success is False
            assert "Invalid arguments" in result.error

    def test_route_permission_denied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = self._make_router(tmpdir)
            cli_tool = router.registry.get("cli")
            cli_tool.enable()
            request = ToolRequest(tool="cli", arguments={"command": "echo"})
            result = router.route(request)
            assert result.success is False
            assert "Permission denied" in result.error

    def test_execute_tool_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = self._make_router(tmpdir)
            result = router.execute_tool(
                "filesystem", {"action": "list"}
            )
            assert result.success is True

    def test_max_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            router = self._make_router(tmpdir)
            router.max_tool_calls_per_request = 2
            router.reset_call_count()
            r1 = router.execute_tool("filesystem", {"action": "list"})
            r2 = router.execute_tool("filesystem", {"action": "list"})
            r3 = router.execute_tool("filesystem", {"action": "list"})
            assert r1.success is True
            assert r2.success is True
            assert r3.success is False
            assert "exceeded" in r3.error.lower()


class TestAuditLogger:
    """Test audit logging."""

    def test_log_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            al = AuditLogger(log_dir=tmpdir)
            record = al.log_request("test_tool", {"arg": "val"}, session_id="s1")
            al.finalize_record(record, True, output="done", execution_time=0.5)
            assert al.get_record_count() == 1

    def test_get_recent_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            al = AuditLogger(log_dir=tmpdir)
            for i in range(5):
                record = al.log_request(f"tool_{i}", {})
                al.finalize_record(record, True)
            records = al.get_recent_records(limit=3)
            assert len(records) == 3

    def test_clear_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            al = AuditLogger(log_dir=tmpdir)
            record = al.log_request("test", {})
            al.finalize_record(record, True)
            al.clear_logs()
            assert al.get_record_count() == 0

    def test_sanitize_args(self):
        record = AuditRecord(arguments={"long": "x" * 300})
        d = record.to_dict()
        assert len(d["arguments"]["long"]) < 300
