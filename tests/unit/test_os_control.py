"""Unit tests for OS control and screen perception system."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.tools.base import Tool, ToolResult, Permission, ConfirmationLevel
from agent.tools.registry import ToolRegistry
from agent.tools.router import ToolRouter
from agent.tools.permissions import PermissionManager
from agent.tools.audit import AuditLogger
from agent.os_control.screen import ScreenCaptureTool
from agent.os_control.system import SystemInfoTool
from agent.os_control.mouse import MouseTool
from agent.os_control.keyboard import KeyboardTool
from agent.os_control.windows import WindowTool
from agent.os_control.permissions import (
    register_os_permissions,
    OS_PERMISSIONS,
    OS_PERMISSION_SCOPES,
)


class TestOSPermissions:
    def test_os_permissions_defined(self):
        assert "os.screen_capture" in OS_PERMISSIONS
        assert "os.system_info" in OS_PERMISSIONS
        assert "os.mouse" in OS_PERMISSIONS
        assert "os.keyboard" in OS_PERMISSIONS
        assert "os.window" in OS_PERMISSIONS

    def test_os_permission_scopes(self):
        assert "*" in OS_PERMISSION_SCOPES["os.screen_capture"]
        assert "*" in OS_PERMISSION_SCOPES["os.system_info"]
        assert "*" in OS_PERMISSION_SCOPES["os.mouse"]
        assert "*" in OS_PERMISSION_SCOPES["os.keyboard"]
        assert OS_PERMISSION_SCOPES["os.window"] == set()

    def test_register_os_permissions(self):
        pm = PermissionManager()
        register_os_permissions(pm)

        assert pm.has_permission("os.screen_capture", "os_control")
        assert pm.has_permission("os.system_info", "os_control")
        assert pm.has_permission("os.screen_capture", "workspace")
        assert pm.has_permission("os.system_info", "workspace")
        assert not pm.has_permission("os.mouse", "os_control")
        assert not pm.has_permission("os.keyboard", "os_control")
        assert not pm.has_permission("os.window", "os_control")

    def test_register_os_permissions_mouse_enabled(self):
        pm = PermissionManager()
        register_os_permissions(pm, mouse_enabled=True)

        assert pm.has_permission("os.mouse", "os_control")
        assert pm.has_permission("os.mouse", "workspace")
        assert not pm.has_permission("os.keyboard", "os_control")

    def test_register_os_permissions_keyboard_enabled(self):
        pm = PermissionManager()
        register_os_permissions(pm, keyboard_enabled=True)

        assert pm.has_permission("os.keyboard", "os_control")
        assert pm.has_permission("os.keyboard", "workspace")
        assert not pm.has_permission("os.mouse", "os_control")

    def test_register_os_permissions_both_enabled(self):
        pm = PermissionManager()
        register_os_permissions(pm, mouse_enabled=True, keyboard_enabled=True)

        assert pm.has_permission("os.mouse", "os_control")
        assert pm.has_permission("os.keyboard", "os_control")
        assert pm.get_confirmation_level("os.mouse") == ConfirmationLevel.REQUIRE_CONFIRMATION
        assert pm.get_confirmation_level("os.keyboard") == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_os_confirmation_levels(self):
        pm = PermissionManager()
        register_os_permissions(pm)

        assert pm.get_confirmation_level("os.screen_capture") == ConfirmationLevel.ALLOW
        assert pm.get_confirmation_level("os.system_info") == ConfirmationLevel.ALLOW
        assert pm.get_confirmation_level("os.mouse") == ConfirmationLevel.REQUIRE_CONFIRMATION
        assert pm.get_confirmation_level("os.keyboard") == ConfirmationLevel.REQUIRE_CONFIRMATION
        assert pm.get_confirmation_level("os.window") == ConfirmationLevel.DENY


class TestScreenCaptureTool:
    def test_name_and_description(self):
        tool = ScreenCaptureTool()
        assert tool.name == "screen_capture"
        assert "screenshot" in tool.description.lower()

    def test_required_permissions(self):
        tool = ScreenCaptureTool()
        assert "os.screen_capture" in tool.required_permissions

    def test_confirmation_level(self):
        tool = ScreenCaptureTool()
        assert tool.confirmation_level == ConfirmationLevel.ALLOW

    def test_timeout(self):
        tool = ScreenCaptureTool()
        assert tool.timeout == 15.0

    def test_validate_no_args(self):
        tool = ScreenCaptureTool()
        ok, errors = tool.validate({})
        assert ok is True

    def test_validate_region(self):
        tool = ScreenCaptureTool()
        ok, errors = tool.validate({"region": {"x": 0, "y": 0, "width": 100, "height": 100}})
        assert ok is True

    def test_validate_invalid_region(self):
        tool = ScreenCaptureTool()
        ok, errors = tool.validate({"region": "invalid"})
        assert ok is False

    def test_validate_negative_dimensions(self):
        tool = ScreenCaptureTool()
        ok, errors = tool.validate({"region": {"width": -1, "height": 100}})
        assert ok is False

    def test_validate_oversized_region(self):
        tool = ScreenCaptureTool()
        ok, errors = tool.validate({"region": {"width": 9999, "height": 9999}})
        assert ok is False

    def test_execute_captures_screenshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ScreenCaptureTool(workspace_dir=tmpdir)
            result = tool.execute({})
            assert result.success is True
            assert result.metadata.get("width", 0) > 0
            assert result.metadata.get("height", 0) > 0

    def test_screenshot_stored_in_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ScreenCaptureTool(workspace_dir=tmpdir)
            result = tool.execute({})
            assert result.success is True
            path = Path(result.metadata["path"]).resolve()
            workspace = Path(tmpdir).resolve()
            assert str(workspace) in str(path)

    def test_screenshot_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ScreenCaptureTool(workspace_dir=tmpdir)
            result = tool.execute({})
            assert result.success is True
            assert os.path.exists(result.metadata["path"])

    def test_screenshot_with_custom_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ScreenCaptureTool(workspace_dir=tmpdir)
            result = tool.execute({"filename": "custom_test.png"})
            assert result.success is True
            assert "custom_test" in result.metadata["path"]

    def test_screenshot_with_region(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = ScreenCaptureTool(workspace_dir=tmpdir)
            result = tool.execute({"region": {"x": 0, "y": 0, "width": 100, "height": 100}})
            assert result.success is True
            assert result.metadata["width"] == 100
            assert result.metadata["height"] == 100

    def test_sanitize_filename(self):
        tool = ScreenCaptureTool()
        safe = tool._sanitize_filename("../../../etc/passwd.png")
        assert ".." not in safe
        assert "/" not in safe

    def test_sanitize_empty_filename(self):
        tool = ScreenCaptureTool()
        safe = tool._sanitize_filename("")
        assert safe.startswith("screenshot_")


class TestSystemInfoTool:
    def test_name_and_description(self):
        tool = SystemInfoTool()
        assert tool.name == "system_info"
        assert "system" in tool.description.lower()

    def test_required_permissions(self):
        tool = SystemInfoTool()
        assert "os.system_info" in tool.required_permissions

    def test_confirmation_level(self):
        tool = SystemInfoTool()
        assert tool.confirmation_level == ConfirmationLevel.ALLOW

    def test_validate_all(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "all"})
        assert ok is True

    def test_validate_os(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "os"})
        assert ok is True

    def test_validate_screen(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "screen"})
        assert ok is True

    def test_validate_cursor(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "cursor"})
        assert ok is True

    def test_validate_active_window(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "active_window"})
        assert ok is True

    def test_validate_invalid_type(self):
        tool = SystemInfoTool()
        ok, errors = tool.validate({"info_type": "invalid"})
        assert ok is False

    def test_execute_all_info(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "all"})
        assert result.success is True
        assert "os" in result.metadata
        assert "screen" in result.metadata
        assert "cursor" in result.metadata

    def test_execute_os_info(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "os"})
        assert result.success is True
        assert "system" in result.metadata["os"]

    def test_execute_screen_info(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "screen"})
        assert result.success is True
        assert "width" in result.metadata["screen"]

    def test_execute_cursor_info(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "cursor"})
        assert result.success is True
        assert "x" in result.metadata["cursor"]

    def test_execute_active_window(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "active_window"})
        assert result.success is True
        assert "title" in result.metadata["active_window"]

    def test_no_secrets_in_output(self):
        tool = SystemInfoTool()
        result = tool.execute({"info_type": "all"})
        output = result.output.lower()
        forbidden = ["password", "secret", "token", "api_key", "credential"]
        for word in forbidden:
            assert word not in output


class TestMouseTool:
    def test_name(self):
        tool = MouseTool()
        assert tool.name == "mouse"

    def test_disabled(self):
        tool = MouseTool()
        ok, errors = tool.validate({})
        assert ok is False
        assert "not enabled" in errors[0].lower()

    def test_execute_returns_disabled(self):
        tool = MouseTool()
        result = tool.execute({})
        assert result.success is False
        assert "not enabled" in result.error.lower()

    def test_confirmation_level_deny(self):
        tool = MouseTool()
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_permissions_required(self):
        tool = MouseTool()
        assert "os.mouse" in tool.required_permissions

    def test_enabled_confirmation_level(self):
        tool = MouseTool(enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_enabled_validate_position(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "position"})
        assert ok is True

    def test_enabled_validate_move_requires_coords(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "move"})
        assert ok is False
        assert "requires x and y" in errors[0].lower()

    def test_enabled_validate_move_valid_coords(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "move", "x": 100, "y": 100})
        assert ok is True

    def test_enabled_validate_move_negative_coords(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "move", "x": -1, "y": 100})
        assert ok is False

    def test_enabled_validate_click_requires_coords(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click"})
        assert ok is False

    def test_enabled_validate_click_valid(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click", "x": 100, "y": 100})
        assert ok is True

    def test_enabled_validate_scroll(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "scroll", "scroll_amount": 3})
        assert ok is True

    def test_enabled_validate_scroll_exceeds_max(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "scroll", "scroll_amount": 100})
        assert ok is False
        assert "exceeds maximum" in errors[0].lower()

    def test_enabled_validate_invalid_action(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "invalid"})
        assert ok is False

    def test_enabled_execute_position(self):
        tool = MouseTool(enabled=True)
        result = tool.execute({"action": "position"})
        assert result.success is True
        assert "position" in result.metadata

    def test_description(self):
        tool = MouseTool()
        assert "mouse" in tool.description.lower()

    def test_timeout(self):
        tool = MouseTool()
        assert tool.timeout == 5.0


class TestKeyboardTool:
    def test_name(self):
        tool = KeyboardTool()
        assert tool.name == "keyboard"

    def test_disabled(self):
        tool = KeyboardTool()
        ok, errors = tool.validate({})
        assert ok is False
        assert "not enabled" in errors[0].lower()

    def test_execute_returns_disabled(self):
        tool = KeyboardTool()
        result = tool.execute({})
        assert result.success is False
        assert "not enabled" in result.error.lower()

    def test_confirmation_level_deny(self):
        tool = KeyboardTool()
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_permissions_required(self):
        tool = KeyboardTool()
        assert "os.keyboard" in tool.required_permissions

    def test_enabled_confirmation_level(self):
        tool = KeyboardTool(enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_enabled_validate_type(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "type", "text": "hello"})
        assert ok is True

    def test_enabled_validate_type_empty(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "type", "text": ""})
        assert ok is False

    def test_enabled_validate_type_too_long(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "type", "text": "x" * 2000})
        assert ok is False
        assert "exceeds maximum" in errors[0].lower()

    def test_enabled_validate_press(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "press", "key": "ENTER"})
        assert ok is True

    def test_enabled_validate_press_requires_key(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "press"})
        assert ok is False

    def test_enabled_validate_press_unknown_key(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "press", "key": "NONEXISTENT"})
        assert ok is False

    def test_enabled_validate_hotkey(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["CTRL", "C"]})
        assert ok is True

    def test_enabled_validate_hotkey_empty(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": []})
        assert ok is False

    def test_enabled_validate_hotkey_too_many_keys(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["A", "B", "C", "D", "E"]})
        assert ok is False
        assert "maximum" in errors[0].lower()

    def test_enabled_validate_hotkey_restricted(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["CTRL", "ALT", "DELETE"]})
        assert ok is False
        assert "restricted" in errors[0].lower()

    def test_enabled_validate_hotkey_alt_f4_restricted(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["ALT", "F4"]})
        assert ok is False
        assert "restricted" in errors[0].lower()

    def test_enabled_validate_invalid_action(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "invalid"})
        assert ok is False

    def test_description(self):
        tool = KeyboardTool()
        assert "keyboard" in tool.description.lower()

    def test_timeout(self):
        tool = KeyboardTool()
        assert tool.timeout == 5.0

    def test_normalize_key_enter(self):
        tool = KeyboardTool(enabled=True)
        assert tool._normalize_key("enter") == "ENTER"
        assert tool._normalize_key("ENTER") == "ENTER"
        assert tool._normalize_key("Return") == "RETURN"

    def test_normalize_key_single_char(self):
        tool = KeyboardTool(enabled=True)
        assert tool._normalize_key("a") == "a"
        assert tool._normalize_key("A") == "A"

    def test_get_vk_code_enter(self):
        tool = KeyboardTool(enabled=True)
        assert tool._get_vk_code("ENTER") == 0x0D

    def test_get_vk_code_single_char(self):
        tool = KeyboardTool(enabled=True)
        assert tool._get_vk_code("A") == ord("A")

    def test_is_restricted_combination(self):
        tool = KeyboardTool(enabled=True)
        restricted, combo = tool._is_restricted_combination(["CTRL", "ALT", "DELETE"])
        assert restricted is True

    def test_is_not_restricted_combination(self):
        tool = KeyboardTool(enabled=True)
        restricted, combo = tool._is_restricted_combination(["CTRL", "C"])
        assert restricted is False


class TestWindowTool:
    def test_name(self):
        tool = WindowTool()
        assert tool.name == "window"

    def test_disabled(self):
        tool = WindowTool()
        ok, errors = tool.validate({})
        assert ok is False
        assert "not enabled" in errors[0].lower()

    def test_execute_returns_disabled(self):
        tool = WindowTool()
        result = tool.execute({})
        assert result.success is False
        assert "not enabled" in result.error.lower()

    def test_confirmation_level_deny(self):
        tool = WindowTool()
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_permissions_required(self):
        tool = WindowTool()
        assert "os.window" in tool.required_permissions


class TestOSToolRegistry:
    def test_register_all_os_tools(self):
        registry = ToolRegistry()

        tools = [
            ScreenCaptureTool(),
            SystemInfoTool(),
            MouseTool(),
            KeyboardTool(),
            WindowTool(),
        ]

        for tool in tools:
            ok = registry.register(tool)
            assert ok is True

        assert registry.count() == 5

    def test_os_tool_names(self):
        registry = ToolRegistry()
        registry.register(ScreenCaptureTool())
        registry.register(SystemInfoTool())
        registry.register(MouseTool())
        registry.register(KeyboardTool())
        registry.register(WindowTool())

        names = registry.list_names()
        assert "screen_capture" in names
        assert "system_info" in names
        assert "mouse" in names
        assert "keyboard" in names
        assert "window" in names


class TestOSToolRouter:
    def test_screen_capture_through_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(ScreenCaptureTool(workspace_dir=tmpdir))
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("screen_capture", {})
            assert result.success is True

    def test_system_info_through_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(SystemInfoTool())
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("system_info", {"info_type": "os"})
            assert result.success is True

    def test_mouse_blocked_through_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(MouseTool())
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("mouse", {})
            assert result.success is False

    def test_keyboard_blocked_through_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(KeyboardTool())
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("keyboard", {})
            assert result.success is False

    def test_window_blocked_through_router(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(WindowTool())
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("window", {})
            assert result.success is False


class TestOSConfig:
    def test_os_control_settings(self):
        from agent.core.config import Config
        config = Config()
        assert hasattr(config, "os_control_enabled")
        assert hasattr(config, "screen_capture_enabled")
        assert hasattr(config, "mouse_control_enabled")
        assert hasattr(config, "keyboard_control_enabled")
        assert hasattr(config, "window_control_enabled")

    def test_os_control_defaults(self):
        from agent.core.config import Config
        config = Config()
        assert config.os_control_enabled is True
        assert config.screen_capture_enabled is True
        assert config.mouse_control_enabled is False
        assert config.keyboard_control_enabled is False
        assert config.window_control_enabled is False


class TestOSControlIntegration:
    def test_all_os_tools_are_tool_subclasses(self):
        tools = [
            ScreenCaptureTool(),
            SystemInfoTool(),
            MouseTool(),
            KeyboardTool(),
            WindowTool(),
        ]
        for tool in tools:
            assert isinstance(tool, Tool)

    def test_all_os_tools_have_schemas(self):
        tools = [
            ScreenCaptureTool(),
            SystemInfoTool(),
            MouseTool(),
            KeyboardTool(),
            WindowTool(),
        ]
        for tool in tools:
            assert "type" in tool.input_schema
            assert "type" in tool.output_schema

    def test_all_os_tools_serializable(self):
        tools = [
            ScreenCaptureTool(),
            SystemInfoTool(),
            MouseTool(),
            KeyboardTool(),
            WindowTool(),
        ]
        for tool in tools:
            d = tool.to_dict()
            assert "name" in d
            assert "description" in d
            assert "required_permissions" in d


class TestMouseToolEnabled:
    def test_move_through_router_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm, mouse_enabled=True)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(MouseTool(enabled=True))
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("mouse", {"action": "position"})
            assert result.success is True

    def test_move_blocked_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm, mouse_enabled=False)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(MouseTool(enabled=False))
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("mouse", {"action": "position"})
            assert result.success is False


class TestKeyboardToolEnabled:
    def test_type_through_router_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm, keyboard_enabled=True)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(KeyboardTool(enabled=True))
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("keyboard", {"action": "type", "text": "test"})
            assert result.success is True

    def test_type_blocked_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry()
            pm = PermissionManager()
            register_os_permissions(pm, keyboard_enabled=False)
            al = AuditLogger(log_dir=tmpdir)
            registry.register(KeyboardTool(enabled=False))
            router = ToolRouter(
                registry=registry,
                permission_manager=pm,
                audit_logger=al,
            )
            result = router.execute_tool("keyboard", {"action": "type", "text": "test"})
            assert result.success is False


class TestOSControlSecurity:
    def test_mouse_validate_out_of_bounds_x(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click", "x": 99999, "y": 100})
        assert ok is False
        assert "outside screen bounds" in errors[0].lower()

    def test_mouse_validate_out_of_bounds_y(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click", "x": 100, "y": 99999})
        assert ok is False
        assert "outside screen bounds" in errors[0].lower()

    def test_mouse_validate_negative_x(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click", "x": -10, "y": 100})
        assert ok is False

    def test_mouse_validate_negative_y(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "click", "x": 100, "y": -10})
        assert ok is False

    def test_mouse_scroll_exceeds_max(self):
        tool = MouseTool(enabled=True)
        ok, errors = tool.validate({"action": "scroll", "scroll_amount": 100})
        assert ok is False

    def test_keyboard_type_exceeds_max_length(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "type", "text": "x" * 5000})
        assert ok is False

    def test_keyboard_hotkey_restricted_ctrl_alt_del(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["CTRL", "ALT", "DELETE"]})
        assert ok is False

    def test_keyboard_hotkey_restricted_alt_f4(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["ALT", "F4"]})
        assert ok is False

    def test_keyboard_hotkey_restricted_ctrl_shift_esc(self):
        tool = KeyboardTool(enabled=True)
        ok, errors = tool.validate({"action": "hotkey", "keys": ["CTRL", "SHIFT", "ESC"]})
        assert ok is False

    def test_mouse_execute_returns_metadata(self):
        tool = MouseTool(enabled=True)
        result = tool.execute({"action": "position"})
        assert result.success is True
        assert "position" in result.metadata
        assert "x" in result.metadata["position"]
        assert "y" in result.metadata["position"]

    def test_keyboard_press_returns_metadata(self):
        tool = KeyboardTool(enabled=True)
        result = tool.execute({"action": "press", "key": "SPACE"})
        assert result.success is True
        assert "key" in result.metadata
        assert result.metadata["key"] == "SPACE"

    def test_keyboard_type_returns_metadata(self):
        tool = KeyboardTool(enabled=True)
        result = tool.execute({"action": "type", "text": "hello"})
        assert result.success is True
        assert "characters_typed" in result.metadata
        assert result.metadata["characters_typed"] == 5

    def test_hotkey_returns_metadata(self):
        tool = KeyboardTool(enabled=True)
        result = tool.execute({"action": "hotkey", "keys": ["CTRL", "C"]})
        assert result.success is True
        assert "keys" in result.metadata

    def test_mouse_metadata_includes_active_window(self):
        tool = MouseTool(enabled=True)
        result = tool.execute({"action": "position"})
        assert result.success is True
        assert "screen" in result.metadata

    def test_keyboard_metadata_includes_active_window(self):
        tool = KeyboardTool(enabled=True)
        result = tool.execute({"action": "press", "key": "SPACE"})
        assert result.success is True
        assert "active_window" in result.metadata


class TestOSControlConfig:
    def test_config_has_mouse_settings(self):
        from agent.core.config import Config
        config = Config()
        assert hasattr(config, "max_mouse_actions_per_request")
        assert hasattr(config, "max_keyboard_actions_per_request")
        assert hasattr(config, "max_typed_text_length")
        assert hasattr(config, "mouse_action_timeout")
        assert hasattr(config, "keyboard_action_timeout")
        assert hasattr(config, "max_scroll_amount")

    def test_config_mouse_defaults(self):
        from agent.core.config import Config
        config = Config()
        assert config.max_mouse_actions_per_request == 20
        assert config.max_keyboard_actions_per_request == 20
        assert config.max_typed_text_length == 1000
        assert config.mouse_action_timeout == 5.0
        assert config.keyboard_action_timeout == 5.0
        assert config.max_scroll_amount == 10
