"""Browser tools.

Stage 2.4.3 - Browser page reading support.

Provides BrowserSessionTool for creating, listing, and closing
isolated browser sessions. Also provides BrowserNavigationTool
for safe URL navigation with scheme validation. Finally provides
BrowserPageReadTool for reading text content from browser pages.
"""

import time
import logging
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, ConfirmationLevel
from .policy import validate_url, sanitize_url_for_log

logger = logging.getLogger(__name__)

VALID_SESSION_ACTIONS = {"create", "list", "close"}
VALID_NAVIGATION_ACTIONS = {"navigate"}
VALID_PAGE_READ_ACTIONS = {"read"}
DEFAULT_MAX_PAGE_TEXT_CHARS = 20000
DEFAULT_MAX_PAGE_TEXT_TOKENS = 5000


class BrowserSessionTool(Tool):
    """Tool for managing browser sessions.

    Actions:
        create - Create a new isolated browser session
        list   - List all active browser sessions
        close  - Close a specific browser session
    """

    def __init__(
        self,
        browser_manager,
        browser_enabled: bool = False,
    ):
        """Initialize the browser session tool.

        Args:
            browser_manager: BrowserManager instance.
            browser_enabled: Whether browser automation is enabled.
        """
        self._manager = browser_manager
        self._enabled = browser_enabled

    @property
    def name(self) -> str:
        return "browser_session"

    @property
    def description(self) -> str:
        return (
            "Manage isolated browser sessions. "
            "Actions: create (launch new session), list (show active sessions), "
            "close (end a session by ID)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "close"],
                    "description": "The session action to perform",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (required for close action)",
                },
            },
            "required": ["action"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "action": {"type": "string"},
                "session_id": {"type": "string"},
                "sessions": {"type": "array"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["browser.session"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 30.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate browser session tool arguments."""
        errors = []

        if not self._enabled:
            return False, ["Browser automation is disabled"]

        action = arguments.get("action", "")
        if action not in VALID_SESSION_ACTIONS:
            errors.append(f"Invalid action: {action}. Must be one of: {', '.join(sorted(VALID_SESSION_ACTIONS))}")

        if action == "close":
            session_id = arguments.get("session_id", "")
            if not session_id:
                errors.append("session_id is required for close action")

        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute browser session tool."""
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        action = arguments.get("action", "")

        try:
            if action == "create":
                return self._execute_create(start)
            elif action == "list":
                return self._execute_list(start)
            elif action == "close":
                return self._execute_close(arguments, start)
            else:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start,
                )
        except Exception as e:
            logger.error(f"Browser session tool error: {e}")
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start,
            )

    def _execute_create(self, start: float) -> ToolResult:
        """Create a new browser session."""
        if self._manager.session_count >= self._manager.max_sessions:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session limit reached ({self._manager.max_sessions}). Close an existing session first.",
                execution_time=time.time() - start,
            )

        manager = self._manager.create_session()
        if manager is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Failed to create browser session",
                execution_time=time.time() - start,
            )

        session_data = manager.session.to_dict()
        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Browser session created: {manager.session_id}",
            execution_time=time.time() - start,
            metadata={
                "action": "create",
                "session_id": manager.session_id,
                "session": session_data,
            },
        )

    def _execute_list(self, start: float) -> ToolResult:
        """List active browser sessions."""
        sessions = self._manager.list_sessions()
        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Found {len(sessions)} active session(s)",
            execution_time=time.time() - start,
            metadata={
                "action": "list",
                "sessions": sessions,
                "count": len(sessions),
            },
        )

    def _execute_close(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        """Close a browser session."""
        session_id = arguments.get("session_id", "")

        manager = self._manager.get_session(session_id)
        if manager is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session not found: {session_id}",
                execution_time=time.time() - start,
            )

        closed = self._manager.close_session(session_id)
        if not closed:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to close session: {session_id}",
                execution_time=time.time() - start,
            )

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Browser session closed: {session_id}",
            execution_time=time.time() - start,
            metadata={
                "action": "close",
                "session_id": session_id,
            },
        )


class BrowserNavigationTool(Tool):
    """Tool for safe browser URL navigation.

    Actions:
        navigate - Navigate an existing page to a validated URL
    """

    def __init__(
        self,
        browser_manager,
        browser_enabled: bool = False,
    ):
        """Initialize the browser navigation tool.

        Args:
            browser_manager: BrowserManager instance.
            browser_enabled: Whether browser automation is enabled.
        """
        self._manager = browser_manager
        self._enabled = browser_enabled

    @property
    def name(self) -> str:
        return "browser_navigation"

    @property
    def description(self) -> str:
        return (
            "Navigate a browser page to a URL. "
            "Actions: navigate (go to an HTTP/HTTPS URL)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["navigate"],
                    "description": "The navigation action to perform",
                },
                "session_id": {
                    "type": "string",
                    "description": "The browser session ID",
                },
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (http or https only)",
                },
            },
            "required": ["action", "session_id", "url"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "session_id": {"type": "string"},
                "url": {"type": "string"},
                "final_url": {"type": "string"},
                "status": {"type": "integer"},
                "navigation_time_ms": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["browser.navigation"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 60.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate browser navigation tool arguments."""
        errors = []

        if not self._enabled:
            return False, ["Browser automation is disabled"]

        action = arguments.get("action", "")
        if action not in VALID_NAVIGATION_ACTIONS:
            errors.append(
                f"Invalid action: {action}. Must be one of: "
                f"{', '.join(sorted(VALID_NAVIGATION_ACTIONS))}"
            )

        session_id = arguments.get("session_id", "")
        if not session_id:
            errors.append("session_id is required")

        url = arguments.get("url", "")
        valid_url, url_errors = validate_url(url)
        if not valid_url:
            errors.extend(url_errors)

        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute browser navigation tool."""
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        session_id = arguments.get("session_id", "")
        url = arguments.get("url", "")

        try:
            return self._execute_navigate(session_id, url, start)
        except Exception as e:
            logger.error(f"Browser navigation tool error: {e}")
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start,
            )

    def _execute_navigate(
        self, session_id: str, url: str, start: float
    ) -> ToolResult:
        """Execute navigate action."""
        session_mgr = self._manager.get_session(session_id)
        if session_mgr is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session not found: {session_id}",
                execution_time=time.time() - start,
            )

        if session_mgr.is_closed:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session is closed: {session_id}",
                execution_time=time.time() - start,
            )

        result = session_mgr.navigate(url)

        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(url)

        if result.get("success"):
            final_url = result.get("final_url", url)
            sanitized_final = sanitize_url_for_log(final_url)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Navigated to {sanitized_final}",
                execution_time=exec_time,
                metadata={
                    "action": "navigate",
                    "session_id": session_id,
                    "url": sanitized_url,
                    "final_url": sanitized_final,
                    "title": result.get("title", ""),
                    "status": result.get("status"),
                    "navigation_time_ms": result.get("navigation_time_ms", 0),
                },
            )
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result.get("error", "Navigation failed"),
                execution_time=exec_time,
                metadata={
                    "action": "navigate",
                    "session_id": session_id,
                    "url": sanitized_url,
                    "error_type": result.get("error_type", "unknown"),
                    "navigation_time_ms": result.get("navigation_time_ms", 0),
                },
            )


class BrowserPageReadTool(Tool):
    """Tool for reading text content from browser pages.

    Stage 2.4.3 - Browser page reading.

    Reads page text using Playwright's inner_text() API.
    Content is treated as UNTRUSTED DATA and wrapped with
    security markers. Never written to audit logs.
    """

    def __init__(
        self,
        browser_manager,
        browser_enabled: bool = False,
        max_page_text_chars: int = DEFAULT_MAX_PAGE_TEXT_CHARS,
        max_page_text_tokens: int = DEFAULT_MAX_PAGE_TEXT_TOKENS,
    ):
        """Initialize the browser page read tool.

        Args:
            browser_manager: BrowserManager instance.
            browser_enabled: Whether browser automation is enabled.
            max_page_text_chars: Maximum characters to return from a page.
            max_page_text_tokens: Approximate maximum tokens to return.
        """
        self._manager = browser_manager
        self._enabled = browser_enabled
        self._max_chars = max_page_text_chars
        self._max_tokens = max_page_text_tokens

    @property
    def name(self) -> str:
        return "browser_page_read"

    @property
    def description(self) -> str:
        return (
            "Read text content from a browser page. "
            "Actions: read (extract visible text from current page). "
            "Content is untrusted and wrapped with security markers."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read"],
                    "description": "The page read action to perform",
                },
                "session_id": {
                    "type": "string",
                    "description": "The browser session ID",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 20000)",
                    "minimum": 100,
                    "maximum": 50000,
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Approximate maximum tokens (default: 5000)",
                    "minimum": 25,
                    "maximum": 12500,
                },
            },
            "required": ["action", "session_id"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "content": {"type": "string"},
                "truncated": {"type": "boolean"},
                "char_count": {"type": "integer"},
                "token_estimate": {"type": "integer"},
                "page_url": {"type": "string"},
                "page_title": {"type": "string"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["browser.page_read"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 30.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate browser page read tool arguments."""
        errors = []

        if not self._enabled:
            return False, ["Browser automation is disabled"]

        action = arguments.get("action", "")
        if action not in VALID_PAGE_READ_ACTIONS:
            errors.append(
                f"Invalid action: {action}. Must be one of: "
                f"{', '.join(sorted(VALID_PAGE_READ_ACTIONS))}"
            )

        session_id = arguments.get("session_id", "")
        if not session_id:
            errors.append("session_id is required")

        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute browser page read tool."""
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        session_id = arguments.get("session_id", "")
        max_chars = arguments.get("max_chars", self._max_chars)
        max_tokens = arguments.get("max_tokens", self._max_tokens)

        try:
            return self._execute_read(session_id, max_chars, max_tokens, start)
        except Exception as e:
            logger.error(f"Browser page read tool error: {e}")
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start,
            )

    def _execute_read(
        self,
        session_id: str,
        max_chars: int,
        max_tokens: int,
        start: float,
    ) -> ToolResult:
        """Execute read action."""
        session_mgr = self._manager.get_session(session_id)
        if session_mgr is None:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session not found: {session_id}",
                execution_time=time.time() - start,
            )

        if session_mgr.is_closed:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Session is closed: {session_id}",
                execution_time=time.time() - start,
            )

        result = session_mgr.read_page_text(
            max_chars=max_chars,
            max_tokens=max_tokens,
        )

        exec_time = time.time() - start

        if result.get("error"):
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result["error"],
                execution_time=exec_time,
                metadata={
                    "action": "read",
                    "session_id": session_id,
                    "code": result.get("code", "UNKNOWN"),
                },
            )

        wrapped_content = (
            "[BEGIN UNTRUSTED WEBPAGE CONTENT]\n"
            f"{result['content']}\n"
            "[END UNTRUSTED WEBPAGE CONTENT]"
        )

        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=wrapped_content,
            execution_time=exec_time,
            metadata={
                "action": "read",
                "session_id": session_id,
                "truncated": result.get("truncated", False),
                "char_count": result.get("char_count", 0),
                "token_estimate": result.get("token_estimate", 0),
                "page_url": sanitized_url,
                "page_title": result.get("page_title", ""),
            },
        )
