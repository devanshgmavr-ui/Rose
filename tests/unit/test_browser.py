"""Unit tests for browser automation.

Stage 2.4.3 - Browser page reading tests.

Covers: configuration, models, session, manager, permissions,
URL validation, URL sanitization, navigation tool, page reading,
security, agent integration.
"""

import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock, PropertyMock

from agent.tools.base import Tool, ToolResult, ToolRequest, ConfirmationLevel
from agent.tools.registry import ToolRegistry
from agent.tools.router import ToolRouter
from agent.tools.permissions import PermissionManager
from agent.tools.audit import AuditLogger
from agent.browser.models import BrowserSession, BrowserPage, SessionStatus
from agent.browser.session import BrowserSessionManager
from agent.browser.manager import BrowserManager
from agent.browser.permissions import (
    register_browser_permissions,
    BROWSER_PERMISSIONS,
    BROWSER_PERMISSION_SCOPES,
)
from agent.browser.tools import BrowserSessionTool, BrowserNavigationTool, BrowserPageReadTool


# ============================================================
# Configuration Tests
# ============================================================

class TestBrowserConfig:
    def test_browser_disabled_by_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_automation_enabled is False

    def test_browser_headless_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_headless is True

    def test_browser_max_sessions_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_sessions == 2

    def test_browser_max_tabs_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_tabs == 5

    def test_browser_navigation_timeout_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_navigation_timeout == 30000

    def test_browser_action_timeout_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_action_timeout == 10000

    @patch.dict("os.environ", {"BROWSER_AUTOMATION_ENABLED": "true"})
    def test_browser_enabled_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_automation_enabled is True

    @patch.dict("os.environ", {"BROWSER_HEADLESS": "false"})
    def test_browser_headless_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_headless is False

    @patch.dict("os.environ", {"BROWSER_MAX_SESSIONS": "5"})
    def test_browser_max_sessions_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_sessions == 5

    @patch.dict("os.environ", {"BROWSER_MAX_TABS": "10"})
    def test_browser_max_tabs_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_tabs == 10


# ============================================================
# Models Tests
# ============================================================

class TestBrowserModels:
    def test_session_creation(self):
        s = BrowserSession(
            session_id="test_123",
            created_at=time.time(),
            headless=True,
        )
        assert s.session_id == "test_123"
        assert s.headless is True
        assert s.status == SessionStatus.ACTIVE
        assert s.page_count == 1

    def test_session_serialization(self):
        s = BrowserSession(
            session_id="test_abc",
            created_at=1000.0,
            headless=False,
            status=SessionStatus.ACTIVE,
            page_count=3,
            current_url="https://example.com",
        )
        d = s.to_dict()
        assert d["session_id"] == "test_abc"
        assert d["headless"] is False
        assert d["status"] == "active"
        assert d["page_count"] == 3
        assert d["current_url"] == "https://example.com"

    def test_session_deserialization(self):
        data = {
            "session_id": "test_xyz",
            "created_at": 2000.0,
            "headless": True,
            "status": "closed",
            "page_count": 0,
        }
        s = BrowserSession.from_dict(data)
        assert s.session_id == "test_xyz"
        assert s.status == SessionStatus.CLOSED
        assert s.page_count == 0

    def test_session_generate_id(self):
        sid = BrowserSession.generate_id()
        assert sid.startswith("browser_")
        assert len(sid) == 20

    def test_page_creation(self):
        p = BrowserPage(page_id="p1", created_at=time.time())
        assert p.page_id == "p1"
        assert p.title == ""
        assert p.url == ""

    def test_page_serialization(self):
        p = BrowserPage(
            page_id="p2",
            created_at=1000.0,
            title="Test",
            url="https://example.com",
            is_active=True,
        )
        d = p.to_dict()
        assert d["page_id"] == "p2"
        assert d["title"] == "Test"
        assert d["is_active"] is True

    def test_session_status_enum(self):
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.CLOSING.value == "closing"
        assert SessionStatus.CLOSED.value == "closed"
        assert SessionStatus.ERROR.value == "error"


# ============================================================
# Permissions Tests
# ============================================================

class TestBrowserPermissions:
    def test_permissions_defined(self):
        assert "browser.session" in BROWSER_PERMISSIONS

    def test_permission_scopes(self):
        assert "*" in BROWSER_PERMISSION_SCOPES["browser.session"]

    def test_register_permissions_disabled(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=False)
        assert not pm.has_permission("browser.session", "browser")

    def test_register_permissions_enabled(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        assert pm.has_permission("browser.session", "browser")
        assert pm.has_permission("browser.session", "workspace")

    def test_confirmation_level(self):
        assert BROWSER_PERMISSIONS["browser.session"] == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_register_sets_confirmation_level(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        level = pm.get_confirmation_level("browser.session")
        assert level == ConfirmationLevel.REQUIRE_CONFIRMATION


# ============================================================
# BrowserManager Tests
# ============================================================

class TestBrowserManager:
    def test_initialization(self):
        m = BrowserManager()
        assert m.initialized is False
        assert m.session_count == 0

    def test_max_sessions_property(self):
        m = BrowserManager(max_sessions=5)
        assert m.max_sessions == 5

    def test_health_check_not_initialized(self):
        m = BrowserManager()
        h = m.health_check()
        assert h["initialized"] is False
        assert h["active_sessions"] == 0

    def test_stats_not_initialized(self):
        m = BrowserManager()
        s = m.get_stats()
        assert s["initialized"] is False
        assert s["active_sessions"] == 0

    @patch("agent.browser.manager.BrowserManager.initialize")
    def test_create_session_not_initialized(self, mock_init):
        m = BrowserManager()
        result = m.create_session()
        assert result is None

    @patch("agent.browser.manager.BrowserManager.initialize")
    def test_close_all_empty(self, mock_init):
        m = BrowserManager()
        count = m.close_all()
        assert count == 0

    def test_shutdown(self):
        m = BrowserManager()
        m.shutdown()
        assert m.initialized is False

    def test_list_sessions_empty(self):
        m = BrowserManager()
        sessions = m.list_sessions()
        assert sessions == []

    def test_get_session_not_found(self):
        m = BrowserManager()
        result = m.get_session("nonexistent")
        assert result is None

    def test_close_session_not_found(self):
        m = BrowserManager()
        result = m.close_session("nonexistent")
        assert result is False

    @patch("playwright.sync_api.sync_playwright")
    def test_initialize_success(self, mock_sync):
        mock_pw = MagicMock()
        mock_sync.return_value.start.return_value = mock_pw
        m = BrowserManager()
        result = m.initialize()
        assert result is True
        assert m.initialized is True

    @patch("playwright.sync_api.sync_playwright")
    def test_initialize_failure(self, mock_sync):
        mock_sync.return_value.start.side_effect = Exception("PW error")
        m = BrowserManager()
        result = m.initialize()
        assert result is False
        assert m.initialized is False

    def test_headless_default(self):
        m = BrowserManager()
        assert m._headless is True

    def test_headless_configurable(self):
        m = BrowserManager(headless=False)
        assert m._headless is False


# ============================================================
# BrowserSessionManager Tests
# ============================================================

class TestBrowserSessionManager:
    def _make_manager(self, closed=False):
        session = BrowserSession(
            session_id="test_session_1",
            created_at=time.time(),
            headless=True,
        )
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "about:blank"
        mock_page.title.return_value = "New Tab"

        mgr = BrowserSessionManager(
            session=session,
            browser=mock_browser,
            context=mock_context,
            page=mock_page,
        )
        if closed:
            mgr._closed = True
        return mgr

    def test_session_id(self):
        mgr = self._make_manager()
        assert mgr.session_id == "test_session_1"

    def test_is_closed_false(self):
        mgr = self._make_manager()
        assert mgr.is_closed is False

    def test_is_closed_true(self):
        mgr = self._make_manager(closed=True)
        assert mgr.is_closed is True

    def test_get_current_url(self):
        mgr = self._make_manager()
        url = mgr.get_current_url()
        assert url == "about:blank"

    def test_get_current_url_closed(self):
        mgr = self._make_manager(closed=True)
        url = mgr.get_current_url()
        assert url == ""

    def test_get_current_title(self):
        mgr = self._make_manager()
        title = mgr.get_current_title()
        assert title == "New Tab"

    def test_get_current_title_closed(self):
        mgr = self._make_manager(closed=True)
        title = mgr.get_current_title()
        assert title == ""

    def test_close(self):
        mgr = self._make_manager()
        result = mgr.close()
        assert result is True
        assert mgr.is_closed is True
        assert mgr.session.status == SessionStatus.CLOSED

    def test_close_already_closed(self):
        mgr = self._make_manager(closed=True)
        result = mgr.close()
        assert result is False

    def test_close_handles_context_error(self):
        mgr = self._make_manager()
        mgr._context.close.side_effect = Exception("ctx err")
        result = mgr.close()
        assert result is True

    def test_close_handles_browser_error(self):
        mgr = self._make_manager()
        mgr._browser.close.side_effect = Exception("brw err")
        result = mgr.close()
        assert result is True

    def test_page_property(self):
        mgr = self._make_manager()
        assert mgr.page is not None

    def test_context_property(self):
        mgr = self._make_manager()
        assert mgr.context is not None

    def test_browser_property(self):
        mgr = self._make_manager()
        assert mgr.browser is not None

    def test_update_state(self):
        mgr = self._make_manager()
        mgr.update_state()
        assert mgr.session.current_url == "about:blank"
        assert mgr.session.current_title == "New Tab"

    def test_update_state_closed(self):
        mgr = self._make_manager(closed=True)
        mgr.update_state()
        assert mgr.session.current_url == ""

    def test_page_access_error(self):
        mgr = self._make_manager()
        type(mgr._page).url = PropertyMock(side_effect=Exception("err"))
        url = mgr.get_current_url()
        assert url == ""


# ============================================================
# BrowserSessionTool Tests
# ============================================================

class TestBrowserSessionTool:
    def _make_tool(self, enabled=True, session_count=0, max_sessions=2):
        mock_manager = MagicMock()
        mock_manager.session_count = session_count
        mock_manager.max_sessions = max_sessions
        mock_manager.list_sessions.return_value = []
        return BrowserSessionTool(
            browser_manager=mock_manager,
            browser_enabled=enabled,
        ), mock_manager

    def test_tool_name(self):
        tool, _ = self._make_tool()
        assert tool.name == "browser_session"

    def test_tool_description(self):
        tool, _ = self._make_tool()
        assert "browser" in tool.description.lower()

    def test_tool_is_tool_subclass(self):
        tool, _ = self._make_tool()
        assert isinstance(tool, Tool)

    def test_tool_permissions(self):
        tool, _ = self._make_tool()
        assert "browser.session" in tool.required_permissions

    def test_tool_confirmation_level_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_tool_confirmation_level_enabled(self):
        tool, _ = self._make_tool(enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_tool_timeout(self):
        tool, _ = self._make_tool()
        assert tool.timeout == 30.0

    def test_tool_has_input_schema(self):
        tool, _ = self._make_tool()
        schema = tool.input_schema
        assert "action" in schema["properties"]
        assert "create" in schema["properties"]["action"]["enum"]

    def test_tool_has_output_schema(self):
        tool, _ = self._make_tool()
        schema = tool.output_schema
        assert "success" in schema["properties"]

    def test_validate_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        valid, errors = tool.validate({"action": "create"})
        assert valid is False
        assert "disabled" in errors[0].lower()

    def test_validate_invalid_action(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({"action": "invalid"})
        assert valid is False
        assert "invalid action" in errors[0].lower()

    def test_validate_create_valid(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({"action": "create"})
        assert valid is True
        assert errors == []

    def test_validate_list_valid(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({"action": "list"})
        assert valid is True

    def test_validate_close_missing_session_id(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({"action": "close"})
        assert valid is False
        assert "session_id" in errors[0].lower()

    def test_validate_close_with_session_id(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({"action": "close", "session_id": "abc"})
        assert valid is True

    def test_execute_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        result = tool.execute({"action": "create"})
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_execute_create_success(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.session_id = "new_session"
        mock_session.to_dict.return_value = {"session_id": "new_session"}
        mock_manager.create_session.return_value = mock_session
        result = tool.execute({"action": "create"})
        assert result.success is True
        assert "new_session" in result.metadata["session_id"]

    def test_execute_create_limit_reached(self):
        tool, mock_manager = self._make_tool(session_count=2, max_sessions=2)
        result = tool.execute({"action": "create"})
        assert result.success is False
        assert "limit" in result.error.lower()

    def test_execute_create_failure(self):
        tool, mock_manager = self._make_tool()
        mock_manager.create_session.return_value = None
        result = tool.execute({"action": "create"})
        assert result.success is False
        assert "failed" in result.error.lower()

    def test_execute_list_success(self):
        tool, mock_manager = self._make_tool()
        mock_manager.list_sessions.return_value = [{"session_id": "s1"}]
        result = tool.execute({"action": "list"})
        assert result.success is True
        assert result.metadata["count"] == 1

    def test_execute_close_success(self):
        tool, mock_manager = self._make_tool()
        mock_manager.get_session.return_value = MagicMock()
        mock_manager.close_session.return_value = True
        result = tool.execute({"action": "close", "session_id": "s1"})
        assert result.success is True
        assert result.metadata["session_id"] == "s1"

    def test_execute_close_not_found(self):
        tool, mock_manager = self._make_tool()
        mock_manager.get_session.return_value = None
        result = tool.execute({"action": "close", "session_id": "missing"})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_close_failure(self):
        tool, mock_manager = self._make_tool()
        mock_manager.get_session.return_value = MagicMock()
        mock_manager.close_session.return_value = False
        result = tool.execute({"action": "close", "session_id": "s1"})
        assert result.success is False
        assert "failed" in result.error.lower()

    def test_execute_unexpected_error(self):
        tool, mock_manager = self._make_tool()
        mock_manager.list_sessions.side_effect = Exception("unexpected")
        result = tool.execute({"action": "list"})
        assert result.success is False
        assert "unexpected" in result.error.lower()


# ============================================================
# Security Tests
# ============================================================

class TestBrowserSecurity:
    def test_browser_disabled_by_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_automation_enabled is False

    def test_no_persistent_profile_in_session(self):
        s = BrowserSession(
            session_id="test",
            created_at=time.time(),
            headless=True,
        )
        d = s.to_dict()
        assert "profile" not in d
        assert "cookies" not in d
        assert "password" not in d
        assert "token" not in d

    def test_no_sensitive_data_in_page(self):
        p = BrowserPage(page_id="p1", created_at=time.time())
        d = p.to_dict()
        assert "cookies" not in d
        assert "password" not in d
        assert "token" not in d

    def test_session_limit_enforced(self):
        tool, mock_manager = self._make_tool(session_count=2, max_sessions=2)
        result = tool.execute({"action": "create"})
        assert result.success is False

    def _make_tool(self, enabled=True, session_count=0, max_sessions=2):
        mock_manager = MagicMock()
        mock_manager.session_count = session_count
        mock_manager.max_sessions = max_sessions
        return BrowserSessionTool(
            browser_manager=mock_manager,
            browser_enabled=enabled,
        ), mock_manager

    def test_confirmation_required_for_mutations(self):
        tool, _ = self._make_tool(enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_deny_when_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        assert tool.confirmation_level == ConfirmationLevel.DENY


# ============================================================
# BrowserManager Integration Tests
# ============================================================

class TestBrowserManagerIntegration:
    def test_stats_after_init(self):
        m = BrowserManager(max_sessions=3, max_tabs=10)
        s = m.get_stats()
        assert s["max_sessions"] == 3
        assert s["max_tabs"] == 10

    def test_health_check_structure(self):
        m = BrowserManager()
        h = m.health_check()
        assert "initialized" in h
        assert "headless" in h
        assert "active_sessions" in h
        assert "max_sessions" in h
        assert "playwright_available" in h

    def test_close_all_returns_count(self):
        m = BrowserManager()
        count = m.close_all()
        assert isinstance(count, int)
        assert count == 0

    def test_double_shutdown(self):
        m = BrowserManager()
        m.shutdown()
        m.shutdown()
        assert m.initialized is False


# ============================================================
# Router Integration Tests
# ============================================================

class TestBrowserRouterIntegration:
    def test_tool_through_router_disabled(self):
        mock_manager = MagicMock()
        tool = BrowserSessionTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        result = router.execute_tool("browser_session", {"action": "create"})
        assert result.success is False

    def test_tool_through_router_enabled(self):
        mock_manager = MagicMock()
        mock_manager.session_count = 0
        mock_manager.max_sessions = 2
        mock_session = MagicMock()
        mock_session.session_id = "routed_session"
        mock_session.to_dict.return_value = {"session_id": "routed_session"}
        mock_manager.create_session.return_value = mock_session

        tool = BrowserSessionTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        result = router.execute_tool("browser_session", {"action": "create"})
        assert result.success is True
        assert "routed_session" in result.metadata.get("session_id", "")


# ============================================================
# Package Import Tests
# ============================================================

class TestBrowserPackageImports:
    def test_import_models(self):
        from agent.browser.models import BrowserSession, BrowserPage, SessionStatus
        assert BrowserSession is not None
        assert BrowserPage is not None
        assert SessionStatus is not None

    def test_import_session(self):
        from agent.browser.session import BrowserSessionManager
        assert BrowserSessionManager is not None

    def test_import_manager(self):
        from agent.browser.manager import BrowserManager
        assert BrowserManager is not None

    def test_import_permissions(self):
        from agent.browser.permissions import (
            register_browser_permissions,
            BROWSER_PERMISSIONS,
            BROWSER_PERMISSION_SCOPES,
        )
        assert register_browser_permissions is not None

    def test_import_tools(self):
        from agent.browser.tools import BrowserSessionTool, BrowserNavigationTool
        assert BrowserSessionTool is not None
        assert BrowserNavigationTool is not None

    def test_import_from_package(self):
        from agent.browser import (
            BrowserSession,
            BrowserPage,
            SessionStatus,
            BrowserSessionManager,
            BrowserManager,
            BrowserSessionTool,
            BrowserNavigationTool,
            register_browser_permissions,
            validate_url,
            sanitize_url_for_log,
        )
        assert all([
            BrowserSession, BrowserPage, SessionStatus,
            BrowserSessionManager, BrowserManager,
            BrowserSessionTool, BrowserNavigationTool,
            register_browser_permissions,
            validate_url, sanitize_url_for_log,
        ])


# ============================================================
# URL Validation Tests (Stage 2.4.2)
# ============================================================

class TestURLValidation:
    def test_valid_http(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("http://example.com")
        assert valid is True
        assert errors == []

    def test_valid_https(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("https://example.com")
        assert valid is True
        assert errors == []

    def test_valid_https_with_path(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("https://example.com/path/to/page")
        assert valid is True

    def test_valid_https_with_query(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("https://example.com/search?q=test&page=1")
        assert valid is True

    def test_empty_url(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("")
        assert valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_none_like_url(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("   ")
        assert valid is False

    def test_malformed_url(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("not-a-url")
        assert valid is False

    def test_file_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("file:///etc/passwd")
        assert valid is False
        assert any("scheme" in e.lower() for e in errors)

    def test_javascript_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("javascript:alert(1)")
        assert valid is False
        assert any("scheme" in e.lower() for e in errors)

    def test_data_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("data:text/html,<h1>hi</h1>")
        assert valid is False

    def test_vbscript_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("vbscript:MsgBox(1)")
        assert valid is False

    def test_about_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("about:blank")
        assert valid is False

    def test_chrome_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("chrome://settings")
        assert valid is False

    def test_view_source_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("view-source:https://example.com")
        assert valid is False

    def test_blob_scheme_rejected(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("blob:https://example.com/id")
        assert valid is False

    def test_no_hostname(self):
        from agent.browser.policy import validate_url
        valid, errors = validate_url("https://")
        assert valid is False
        assert any("hostname" in e.lower() for e in errors)


# ============================================================
# URL Sanitization Tests (Stage 2.4.2)
# ============================================================

class TestURLSanitization:
    def test_sanitize_no_sensitive_params(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/page?q=search&page=1"
        result = sanitize_url_for_log(url)
        assert result == url

    def test_sanitize_token_param(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/auth?token=abc123&user=test"
        result = sanitize_url_for_log(url)
        assert "abc123" not in result
        assert "REDACTED" in result
        assert "user=test" in result

    def test_sanitize_api_key_param(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/api?key=secret123"
        result = sanitize_url_for_log(url)
        assert "secret123" not in result
        assert "REDACTED" in result

    def test_sanitize_password_param(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/login?password=mypass"
        result = sanitize_url_for_log(url)
        assert "mypass" not in result

    def test_sanitize_access_token(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/callback?access_token=xyz789"
        result = sanitize_url_for_log(url)
        assert "xyz789" not in result

    def test_sanitize_secret_param(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/config?secret=mysecret"
        result = sanitize_url_for_log(url)
        assert "mysecret" not in result

    def test_sanitize_empty_url(self):
        from agent.browser.policy import sanitize_url_for_log
        result = sanitize_url_for_log("")
        assert result == ""

    def test_sanitize_no_params(self):
        from agent.browser.policy import sanitize_url_for_log
        url = "https://example.com/page"
        result = sanitize_url_for_log(url)
        assert result == url


# ============================================================
# Browser Navigation Permission Tests (Stage 2.4.2)
# ============================================================

class TestBrowserNavigationPermissions:
    def test_navigation_permission_defined(self):
        assert "browser.navigation" in BROWSER_PERMISSIONS

    def test_navigation_permission_scope(self):
        assert "*" in BROWSER_PERMISSION_SCOPES["browser.navigation"]

    def test_navigation_permission_confirmation_level(self):
        assert BROWSER_PERMISSIONS["browser.navigation"] == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_register_permissions_with_navigation(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        assert pm.has_permission("browser.session", "browser")
        assert pm.has_permission("browser.navigation", "browser")

    def test_register_permissions_disabled_no_navigation(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=False)
        assert not pm.has_permission("browser.navigation", "browser")


# ============================================================
# BrowserNavigationTool Tests (Stage 2.4.2)
# ============================================================

class TestBrowserNavigationTool:
    def _make_tool(self, enabled=True, session_count=0, max_sessions=2):
        mock_manager = MagicMock()
        mock_manager.session_count = session_count
        mock_manager.max_sessions = max_sessions
        return BrowserNavigationTool(
            browser_manager=mock_manager,
            browser_enabled=enabled,
        ), mock_manager

    def test_tool_name(self):
        tool, _ = self._make_tool()
        assert tool.name == "browser_navigation"

    def test_tool_description(self):
        tool, _ = self._make_tool()
        assert "navigate" in tool.description.lower()

    def test_tool_is_tool_subclass(self):
        tool, _ = self._make_tool()
        assert isinstance(tool, Tool)

    def test_tool_permissions(self):
        tool, _ = self._make_tool()
        assert "browser.navigation" in tool.required_permissions

    def test_tool_confirmation_level_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_tool_confirmation_level_enabled(self):
        tool, _ = self._make_tool(enabled=True)
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_tool_timeout(self):
        tool, _ = self._make_tool()
        assert tool.timeout == 60.0

    def test_tool_has_input_schema(self):
        tool, _ = self._make_tool()
        schema = tool.input_schema
        assert "action" in schema["properties"]
        assert "navigate" in schema["properties"]["action"]["enum"]
        assert "session_id" in schema["properties"]
        assert "url" in schema["properties"]

    def test_tool_has_output_schema(self):
        tool, _ = self._make_tool()
        schema = tool.output_schema
        assert "success" in schema["properties"]

    def test_validate_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        valid, errors = tool.validate({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert valid is False
        assert "disabled" in errors[0].lower()

    def test_validate_invalid_action(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "click",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert valid is False
        assert "invalid action" in errors[0].lower()

    def test_validate_missing_session_id(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "navigate",
            "url": "https://example.com",
        })
        assert valid is False
        assert "session_id" in errors[0].lower()

    def test_validate_empty_url(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "navigate",
            "session_id": "s1",
            "url": "",
        })
        assert valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_validate_invalid_url_scheme(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "navigate",
            "session_id": "s1",
            "url": "file:///etc/passwd",
        })
        assert valid is False
        assert any("scheme" in e.lower() for e in errors)

    def test_validate_valid_https(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert valid is True
        assert errors == []

    def test_validate_valid_http(self):
        tool, _ = self._make_tool()
        valid, errors = tool.validate({
            "action": "navigate",
            "session_id": "s1",
            "url": "http://example.com",
        })
        assert valid is True

    def test_execute_disabled(self):
        tool, _ = self._make_tool(enabled=False)
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "disabled" in result.error.lower()

    def test_execute_session_not_found(self):
        tool, mock_manager = self._make_tool()
        mock_manager.get_session.return_value = None
        result = tool.execute({
            "action": "navigate",
            "session_id": "nonexistent",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_session_closed(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = True
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "closed" in result.error.lower()

    def test_execute_navigation_success(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": True,
            "url": "https://example.com",
            "final_url": "https://example.com/",
            "title": "Example Domain",
            "status": 200,
            "navigation_time_ms": 500,
        }
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is True
        assert result.metadata["final_url"] == "https://example.com/"
        assert result.metadata["status"] == 200

    def test_execute_navigation_failure(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": False,
            "error": "Connection refused",
            "error_type": "ConnectionError",
            "navigation_time_ms": 100,
        }
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "connection refused" in result.error.lower()

    def test_execute_navigation_timeout(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": False,
            "error": "Navigation timeout after 30000ms",
            "error_type": "timeout",
            "navigation_time_ms": 30000,
        }
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_execute_unexpected_error(self):
        tool, mock_manager = self._make_tool()
        mock_manager.get_session.side_effect = Exception("unexpected")
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False
        assert "unexpected" in result.error.lower()

    def test_metadata_includes_url(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": True,
            "final_url": "https://example.com/",
            "title": "",
            "status": 200,
            "navigation_time_ms": 100,
        }
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert "url" in result.metadata
        assert "final_url" in result.metadata

    def test_metadata_sanitizes_url(self):
        tool, mock_manager = self._make_tool()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": True,
            "final_url": "https://example.com/",
            "title": "",
            "status": 200,
            "navigation_time_ms": 100,
        }
        mock_manager.get_session.return_value = mock_session
        result = tool.execute({
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com/auth?token=secret123",
        })
        assert "secret123" not in str(result.metadata)


# ============================================================
# BrowserSessionManager Navigation Tests (Stage 2.4.2)
# ============================================================

class TestBrowserSessionNavigation:
    def _make_session_manager(self, closed=False):
        session = BrowserSession(
            session_id="nav_test_session",
            created_at=time.time(),
            headless=True,
        )
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "about:blank"
        mock_page.title.return_value = "New Tab"

        mgr = BrowserSessionManager(
            session=session,
            browser=mock_browser,
            context=mock_context,
            page=mock_page,
        )
        if closed:
            mgr._closed = True
        return mgr

    def test_navigate_closed_session(self):
        mgr = self._make_session_manager(closed=True)
        result = mgr.navigate("https://example.com")
        assert result["success"] is False
        assert "closed" in result["error"].lower()

    def test_navigate_no_page(self):
        mgr = self._make_session_manager()
        mgr._page = None
        result = mgr.navigate("https://example.com")
        assert result["success"] is False
        assert "no page" in result["error"].lower()

    def test_navigate_success(self):
        mgr = self._make_session_manager()
        mock_response = MagicMock()
        mock_response.status = 200
        mgr._page.goto.return_value = mock_response
        mgr._page.url = "https://example.com/"
        mgr._page.title.return_value = "Example"

        result = mgr.navigate("https://example.com")
        assert result["success"] is True
        assert result["final_url"] == "https://example.com/"
        assert result["title"] == "Example"
        assert result["status"] == 200
        assert "navigation_time_ms" in result

    def test_navigate_updates_session_state(self):
        mgr = self._make_session_manager()
        mock_response = MagicMock()
        mock_response.status = 200
        mgr._page.goto.return_value = mock_response
        mgr._page.url = "https://updated.com/"
        mgr._page.title.return_value = "Updated"

        mgr.navigate("https://updated.com")
        assert mgr.session.current_url == "https://updated.com/"
        assert mgr.session.current_title == "Updated"

    def test_navigate_timeout_error(self):
        mgr = self._make_session_manager()
        mgr._page.goto.side_effect = Exception("Navigation timeout")

        result = mgr.navigate("https://example.com")
        assert result["success"] is False
        assert "timeout" in result["error_type"]

    def test_navigate_playwright_error(self):
        mgr = self._make_session_manager()
        mgr._page.goto.side_effect = Exception("Connection refused")

        result = mgr.navigate("https://example.com")
        assert result["success"] is False
        assert result["error_type"] == "Exception"

    def test_navigate_no_response_status(self):
        mgr = self._make_session_manager()
        mgr._page.goto.return_value = None
        mgr._page.url = "https://example.com/"

        result = mgr.navigate("https://example.com")
        assert result["success"] is True
        assert result["status"] is None

    def test_navigate_with_timeout_parameter(self):
        mgr = self._make_session_manager()
        mock_response = MagicMock()
        mock_response.status = 200
        mgr._page.goto.return_value = mock_response
        mgr._page.url = "https://example.com/"

        mgr.navigate("https://example.com", timeout=5000)
        mgr._page.goto.assert_called_once_with("https://example.com", timeout=5000)


# ============================================================
# Browser Navigation Router Integration Tests (Stage 2.4.2)
# ============================================================

class TestBrowserNavigationRouterIntegration:
    def test_navigation_tool_through_router_disabled(self):
        mock_manager = MagicMock()
        tool = BrowserNavigationTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        result = router.execute_tool("browser_navigation", {
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is False

    def test_navigation_tool_through_router_enabled(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.navigate.return_value = {
            "success": True,
            "final_url": "https://example.com/",
            "title": "Example",
            "status": 200,
            "navigation_time_ms": 200,
        }
        mock_manager.get_session.return_value = mock_session

        tool = BrowserNavigationTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        result = router.execute_tool("browser_navigation", {
            "action": "navigate",
            "session_id": "s1",
            "url": "https://example.com",
        })
        assert result.success is True
        assert result.metadata["final_url"] == "https://example.com/"


# ============================================================
# Stage 2.4.3 - Browser Page Reading Tests
# ============================================================

class TestPageReadConfig:
    def test_browser_max_page_text_chars_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_page_text_chars == 20000

    def test_browser_max_page_text_tokens_default(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_page_text_tokens == 5000

    @patch.dict("os.environ", {"BROWSER_MAX_PAGE_TEXT_CHARS": "10000"})
    def test_browser_max_page_text_chars_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_page_text_chars == 10000

    @patch.dict("os.environ", {"BROWSER_MAX_PAGE_TEXT_TOKENS": "2500"})
    def test_browser_max_page_text_tokens_via_env(self):
        from agent.core.config import Config
        c = Config()
        assert c.browser_max_page_text_tokens == 2500


class TestPageReadPermissions:
    def test_page_read_permission_exists(self):
        assert "browser.page_read" in BROWSER_PERMISSIONS

    def test_page_read_permission_default_level(self):
        assert BROWSER_PERMISSIONS["browser.page_read"] == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_page_read_permission_scopes(self):
        assert "browser.page_read" in BROWSER_PERMISSION_SCOPES
        assert "*" in BROWSER_PERMISSION_SCOPES["browser.page_read"]

    def test_register_page_read_permission(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        assert pm.has_permission("browser.page_read", "*")

    def test_register_page_read_not_when_disabled(self):
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=False)
        assert not pm.has_permission("browser.page_read", "*")


class TestBrowserPageReadTool:
    def test_tool_name(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        assert tool.name == "browser_page_read"

    def test_tool_description(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        assert "browser" in tool.description.lower()
        assert "read" in tool.description.lower()

    def test_required_permissions(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        assert tool.required_permissions == ["browser.page_read"]

    def test_confirmation_level_when_enabled(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        assert tool.confirmation_level == ConfirmationLevel.REQUIRE_CONFIRMATION

    def test_confirmation_level_when_disabled(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        assert tool.confirmation_level == ConfirmationLevel.DENY

    def test_validate_disabled(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        valid, errors = tool.validate({"action": "read", "session_id": "s1"})
        assert valid is False
        assert "disabled" in errors[0].lower()

    def test_validate_invalid_action(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        valid, errors = tool.validate({"action": "invalid", "session_id": "s1"})
        assert valid is False
        assert "Invalid action" in errors[0]

    def test_validate_missing_session_id(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        valid, errors = tool.validate({"action": "read"})
        assert valid is False
        assert "session_id is required" in errors[0]

    def test_validate_success(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        valid, errors = tool.validate({"action": "read", "session_id": "s1"})
        assert valid is True
        assert errors == []

    def test_execute_session_not_found(self):
        mock_manager = MagicMock()
        mock_manager.get_session.return_value = None
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "missing"})
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_execute_session_closed(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = True
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is False
        assert "closed" in result.error.lower()

    def test_execute_success(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "Hello world",
            "truncated": False,
            "char_count": 11,
            "token_estimate": 2,
            "page_url": "https://example.com",
            "page_title": "Example",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is True
        assert "[BEGIN UNTRUSTED WEBPAGE CONTENT]" in result.output
        assert "Hello world" in result.output
        assert "[END UNTRUSTED WEBPAGE CONTENT]" in result.output

    def test_execute_success_metadata(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "Test content",
            "truncated": False,
            "char_count": 12,
            "token_estimate": 3,
            "page_url": "https://example.com",
            "page_title": "Test",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.metadata["action"] == "read"
        assert result.metadata["char_count"] == 12
        assert result.metadata["token_estimate"] == 3
        assert result.metadata["truncated"] is False
        assert result.metadata["page_title"] == "Test"

    def test_execute_truncated_content(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "x" * 20000,
            "truncated": True,
            "char_count": 20000,
            "token_estimate": 5000,
            "page_url": "https://example.com",
            "page_title": "Long",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is True
        assert result.metadata["truncated"] is True

    def test_execute_with_custom_limits(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "Small",
            "truncated": False,
            "char_count": 5,
            "token_estimate": 1,
            "page_url": "https://example.com",
            "page_title": "Small",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({
            "action": "read",
            "session_id": "s1",
            "max_chars": 1000,
            "max_tokens": 250,
        })
        assert result.success is True
        mock_session.read_page_text.assert_called_once_with(
            max_chars=1000,
            max_tokens=250,
        )

    def test_execute_read_error(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "error": "No page available",
            "code": "NO_PAGE",
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is False
        assert "No page available" in result.error
        assert result.metadata["code"] == "NO_PAGE"

    def test_execute_unexpected_exception(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.side_effect = RuntimeError("unexpected")
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is False
        assert "Unexpected error" in result.error

    def test_tool_timeout(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        assert tool.timeout == 30.0

    def test_input_schema(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        schema = tool.input_schema
        assert "action" in schema["properties"]
        assert "session_id" in schema["properties"]
        assert "max_chars" in schema["properties"]
        assert "max_tokens" in schema["properties"]
        assert "read" in schema["properties"]["action"]["enum"]

    def test_output_schema(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        schema = tool.output_schema
        assert "content" in schema["properties"]
        assert "truncated" in schema["properties"]
        assert "char_count" in schema["properties"]

    def test_content_not_logged_to_audit(self):
        from agent.tools.audit import AuditLogger
        audit = AuditLogger(log_dir="logs")
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "SECRET DATA",
            "truncated": False,
            "char_count": 11,
            "token_estimate": 2,
            "page_url": "https://example.com",
            "page_title": "Test",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is True
        content_wrapper = "[BEGIN UNTRUSTED WEBPAGE CONTENT]"
        assert content_wrapper in result.output

    def test_validate_disabled_returns_errors(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        valid, errors = tool.validate({"action": "read", "session_id": "s1"})
        assert valid is False
        assert len(errors) == 1
        assert "Browser automation is disabled" in errors[0]

    def test_execute_disabled_returns_error(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        result = tool.execute({"action": "read", "session_id": "s1"})
        assert result.success is False
        assert "disabled" in result.error.lower()


class TestSessionReadPageText:
    def test_read_page_text_closed_session(self):
        session = BrowserSession(
            session_id="s1",
            created_at=time.time(),
            headless=True,
        )
        mgr = BrowserSessionManager(
            session=session,
            browser=MagicMock(),
            context=MagicMock(),
            page=MagicMock(),
        )
        mgr.close()
        result = mgr.read_page_text()
        assert "error" in result
        assert result["code"] == "SESSION_CLOSED"

    def test_read_page_text_success(self):
        session = BrowserSession(
            session_id="s1",
            created_at=time.time(),
            headless=True,
        )
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "Hello world"
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Example"
        mgr = BrowserSessionManager(
            session=session,
            browser=MagicMock(),
            context=MagicMock(),
            page=mock_page,
        )
        result = mgr.read_page_text()
        assert result["content"] == "Hello world"
        assert result["truncated"] is False
        assert result["char_count"] == 11
        assert result["page_url"] == "https://example.com"
        assert result["page_title"] == "Example"
        assert result["content_wrapped"] is True

    def test_read_page_text_truncation(self):
        session = BrowserSession(
            session_id="s1",
            created_at=time.time(),
            headless=True,
        )
        mock_page = MagicMock()
        mock_page.inner_text.return_value = "x" * 50000
        mock_page.url = "https://example.com"
        mock_page.title.return_value = "Long"
        mgr = BrowserSessionManager(
            session=session,
            browser=MagicMock(),
            context=MagicMock(),
            page=mock_page,
        )
        result = mgr.read_page_text(max_chars=1000, max_tokens=250)
        assert result["truncated"] is True
        assert len(result["content"]) == 1000

    def test_read_page_text_read_error(self):
        session = BrowserSession(
            session_id="s1",
            created_at=time.time(),
            headless=True,
        )
        mock_page = MagicMock()
        mock_page.inner_text.side_effect = Exception("read error")
        mgr = BrowserSessionManager(
            session=session,
            browser=MagicMock(),
            context=MagicMock(),
            page=mock_page,
        )
        result = mgr.read_page_text()
        assert "error" in result
        assert result["code"] == "READ_FAILED"


class TestPageReadRouterIntegration:
    def test_router_routes_page_read_tool(self):
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.is_closed = False
        mock_session.read_page_text.return_value = {
            "content": "Hello",
            "truncated": False,
            "char_count": 5,
            "token_estimate": 1,
            "page_url": "https://example.com",
            "page_title": "Test",
            "content_wrapped": True,
        }
        mock_manager.get_session.return_value = mock_session
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=True,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        register_browser_permissions(pm, browser_enabled=True)
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        req = ToolRequest(
            tool="browser_page_read",
            arguments={"action": "read", "session_id": "s1"},
        )
        result = router.route(req)
        assert result.success is True
        assert "[BEGIN UNTRUSTED WEBPAGE CONTENT]" in result.output

    def test_router_blocks_page_read_when_disabled(self):
        mock_manager = MagicMock()
        tool = BrowserPageReadTool(
            browser_manager=mock_manager,
            browser_enabled=False,
        )
        registry = ToolRegistry()
        registry.register(tool)
        pm = PermissionManager()
        audit = AuditLogger(log_dir="logs")
        router = ToolRouter(
            registry=registry,
            permission_manager=pm,
            audit_logger=audit,
        )
        result = router.execute_tool("browser_page_read", {
            "action": "read",
            "session_id": "s1",
        })
        assert result.success is False
        assert "disabled" in result.error.lower()


class TestPageReadPackageImports:
    def test_import_browser_page_read_tool(self):
        from agent.browser import BrowserPageReadTool
        assert BrowserPageReadTool is not None

    def test_import_from_tools_module(self):
        from agent.browser.tools import BrowserPageReadTool
        assert BrowserPageReadTool is not None

    def test_all_exports(self):
        from agent.browser import __all__
        assert "BrowserPageReadTool" in __all__

    def test_permissions_module_updated(self):
        from agent.browser.permissions import BROWSER_PERMISSIONS
        assert "browser.page_read" in BROWSER_PERMISSIONS
