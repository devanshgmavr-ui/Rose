"""Browser tools.

Stage 2.4.4 - Browser interaction support.

Provides BrowserSessionTool for creating, listing, and closing
isolated browser sessions. BrowserNavigationTool for safe URL
navigation. BrowserPageReadTool for reading page text.
BrowserInteractionTool for inspecting and interacting with page
elements.
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
VALID_INTERACTION_ACTIONS = {"inspect", "click", "fill", "select", "press", "wait"}
VALID_WAIT_CONDITIONS = {"selector", "text", "load_state", "url_pattern"}

DEFAULT_MAX_PAGE_TEXT_CHARS = 20000
DEFAULT_MAX_PAGE_TEXT_TOKENS = 5000
DEFAULT_MAX_ELEMENTS_RETURNED = 100
DEFAULT_MAX_INPUT_TEXT_LENGTH = 2000
DEFAULT_INTERACTION_TIMEOUT = 10000
DEFAULT_MAX_INTERACTIONS = 20
DEFAULT_MAX_WAIT_TIMEOUT = 15000

ALLOWED_KEYS = {
    "Enter", "Tab", "Escape", "Space", "Backspace", "Delete",
    "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
    "Home", "End", "PageUp", "PageDown",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "Shift+Tab", "Control+a", "Control+c", "Control+v", "Control+x",
    "Control+z", "Control+y", "Alt+F4",
}


class BrowserSessionTool(Tool):
    """Tool for managing browser sessions."""

    def __init__(self, browser_manager, browser_enabled: bool = False):
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
        start = time.time()
        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(success=False, tool_name=self.name, error="; ".join(errors), execution_time=time.time() - start)
        action = arguments.get("action", "")
        try:
            if action == "create":
                return self._execute_create(start)
            elif action == "list":
                return self._execute_list(start)
            elif action == "close":
                return self._execute_close(arguments, start)
            else:
                return ToolResult(success=False, tool_name=self.name, error=f"Unknown action: {action}", execution_time=time.time() - start)
        except Exception as e:
            logger.error(f"Browser session tool error: {e}")
            return ToolResult(success=False, tool_name=self.name, error=f"Unexpected error: {str(e)}", execution_time=time.time() - start)

    def _execute_create(self, start: float) -> ToolResult:
        if self._manager.session_count >= self._manager.max_sessions:
            return ToolResult(success=False, tool_name=self.name, error=f"Session limit reached ({self._manager.max_sessions}). Close an existing session first.", execution_time=time.time() - start)
        manager = self._manager.create_session()
        if manager is None:
            return ToolResult(success=False, tool_name=self.name, error="Failed to create browser session", execution_time=time.time() - start)
        session_data = manager.session.to_dict()
        return ToolResult(success=True, tool_name=self.name, output=f"Browser session created: {manager.session_id}", execution_time=time.time() - start, metadata={"action": "create", "session_id": manager.session_id, "session": session_data})

    def _execute_list(self, start: float) -> ToolResult:
        sessions = self._manager.list_sessions()
        return ToolResult(success=True, tool_name=self.name, output=f"Found {len(sessions)} active session(s)", execution_time=time.time() - start, metadata={"action": "list", "sessions": sessions, "count": len(sessions)})

    def _execute_close(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        session_id = arguments.get("session_id", "")
        manager = self._manager.get_session(session_id)
        if manager is None:
            return ToolResult(success=False, tool_name=self.name, error=f"Session not found: {session_id}", execution_time=time.time() - start)
        closed = self._manager.close_session(session_id)
        if not closed:
            return ToolResult(success=False, tool_name=self.name, error=f"Failed to close session: {session_id}", execution_time=time.time() - start)
        return ToolResult(success=True, tool_name=self.name, output=f"Browser session closed: {session_id}", execution_time=time.time() - start, metadata={"action": "close", "session_id": session_id})


class BrowserNavigationTool(Tool):
    """Tool for safe browser URL navigation."""

    def __init__(self, browser_manager, browser_enabled: bool = False):
        self._manager = browser_manager
        self._enabled = browser_enabled

    @property
    def name(self) -> str:
        return "browser_navigation"

    @property
    def description(self) -> str:
        return "Navigate a browser page to a URL. Actions: navigate (go to an HTTP/HTTPS URL)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["navigate"], "description": "The navigation action to perform"},
                "session_id": {"type": "string", "description": "The browser session ID"},
                "url": {"type": "string", "description": "The URL to navigate to (http or https only)"},
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
        errors = []
        if not self._enabled:
            return False, ["Browser automation is disabled"]
        action = arguments.get("action", "")
        if action not in VALID_NAVIGATION_ACTIONS:
            errors.append(f"Invalid action: {action}. Must be one of: {', '.join(sorted(VALID_NAVIGATION_ACTIONS))}")
        session_id = arguments.get("session_id", "")
        if not session_id:
            errors.append("session_id is required")
        url = arguments.get("url", "")
        valid_url, url_errors = validate_url(url)
        if not valid_url:
            errors.extend(url_errors)
        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()
        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(success=False, tool_name=self.name, error="; ".join(errors), execution_time=time.time() - start)
        session_id = arguments.get("session_id", "")
        url = arguments.get("url", "")
        try:
            return self._execute_navigate(session_id, url, start)
        except Exception as e:
            logger.error(f"Browser navigation tool error: {e}")
            return ToolResult(success=False, tool_name=self.name, error=f"Unexpected error: {str(e)}", execution_time=time.time() - start)

    def _execute_navigate(self, session_id: str, url: str, start: float) -> ToolResult:
        session_mgr = self._manager.get_session(session_id)
        if session_mgr is None:
            return ToolResult(success=False, tool_name=self.name, error=f"Session not found: {session_id}", execution_time=time.time() - start)
        if session_mgr.is_closed:
            return ToolResult(success=False, tool_name=self.name, error=f"Session is closed: {session_id}", execution_time=time.time() - start)
        result = session_mgr.navigate(url)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(url)
        if result.get("success"):
            final_url = result.get("final_url", url)
            sanitized_final = sanitize_url_for_log(final_url)
            return ToolResult(success=True, tool_name=self.name, output=f"Navigated to {sanitized_final}", execution_time=exec_time, metadata={"action": "navigate", "session_id": session_id, "url": sanitized_url, "final_url": sanitized_final, "title": result.get("title", ""), "status": result.get("status"), "navigation_time_ms": result.get("navigation_time_ms", 0)})
        else:
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Navigation failed"), execution_time=exec_time, metadata={"action": "navigate", "session_id": session_id, "url": sanitized_url, "error_type": result.get("error_type", "unknown"), "navigation_time_ms": result.get("navigation_time_ms", 0)})


class BrowserPageReadTool(Tool):
    """Tool for reading text content from browser pages."""

    def __init__(self, browser_manager, browser_enabled: bool = False, max_page_text_chars: int = DEFAULT_MAX_PAGE_TEXT_CHARS, max_page_text_tokens: int = DEFAULT_MAX_PAGE_TEXT_TOKENS):
        self._manager = browser_manager
        self._enabled = browser_enabled
        self._max_chars = max_page_text_chars
        self._max_tokens = max_page_text_tokens

    @property
    def name(self) -> str:
        return "browser_page_read"

    @property
    def description(self) -> str:
        return "Read text content from a browser page. Actions: read (extract visible text from current page). Content is untrusted and wrapped with security markers."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["read"], "description": "The page read action to perform"},
                "session_id": {"type": "string", "description": "The browser session ID"},
                "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 20000)", "minimum": 100, "maximum": 50000},
                "max_tokens": {"type": "integer", "description": "Approximate maximum tokens (default: 5000)", "minimum": 25, "maximum": 12500},
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
        errors = []
        if not self._enabled:
            return False, ["Browser automation is disabled"]
        action = arguments.get("action", "")
        if action not in VALID_PAGE_READ_ACTIONS:
            errors.append(f"Invalid action: {action}. Must be one of: {', '.join(sorted(VALID_PAGE_READ_ACTIONS))}")
        session_id = arguments.get("session_id", "")
        if not session_id:
            errors.append("session_id is required")
        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()
        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(success=False, tool_name=self.name, error="; ".join(errors), execution_time=time.time() - start)
        session_id = arguments.get("session_id", "")
        max_chars = arguments.get("max_chars", self._max_chars)
        max_tokens = arguments.get("max_tokens", self._max_tokens)
        try:
            return self._execute_read(session_id, max_chars, max_tokens, start)
        except Exception as e:
            logger.error(f"Browser page read tool error: {e}")
            return ToolResult(success=False, tool_name=self.name, error=f"Unexpected error: {str(e)}", execution_time=time.time() - start)

    def _execute_read(self, session_id: str, max_chars: int, max_tokens: int, start: float) -> ToolResult:
        session_mgr = self._manager.get_session(session_id)
        if session_mgr is None:
            return ToolResult(success=False, tool_name=self.name, error=f"Session not found: {session_id}", execution_time=time.time() - start)
        if session_mgr.is_closed:
            return ToolResult(success=False, tool_name=self.name, error=f"Session is closed: {session_id}", execution_time=time.time() - start)
        result = session_mgr.read_page_text(max_chars=max_chars, max_tokens=max_tokens)
        exec_time = time.time() - start
        if result.get("error"):
            return ToolResult(success=False, tool_name=self.name, error=result["error"], execution_time=exec_time, metadata={"action": "read", "session_id": session_id, "code": result.get("code", "UNKNOWN")})
        wrapped_content = "[BEGIN UNTRUSTED WEBPAGE CONTENT]\n" + result['content'] + "\n[END UNTRUSTED WEBPAGE CONTENT]"
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        return ToolResult(success=True, tool_name=self.name, output=wrapped_content, execution_time=exec_time, metadata={"action": "read", "session_id": session_id, "truncated": result.get("truncated", False), "char_count": result.get("char_count", 0), "token_estimate": result.get("token_estimate", 0), "page_url": sanitized_url, "page_title": result.get("page_title", "")})


class BrowserInteractionTool(Tool):
    """Tool for browser page interaction.

    Stage 2.4.4 - Browser interaction.

    Supports inspect, click, fill, select, press, wait actions.
    No JavaScript execution. No page.evaluate().
    All content treated as untrusted.
    """

    def __init__(
        self,
        browser_manager,
        browser_enabled: bool = False,
        max_elements_returned: int = DEFAULT_MAX_ELEMENTS_RETURNED,
        max_input_text_length: int = DEFAULT_MAX_INPUT_TEXT_LENGTH,
        interaction_timeout: int = DEFAULT_INTERACTION_TIMEOUT,
        max_wait_timeout: int = DEFAULT_MAX_WAIT_TIMEOUT,
    ):
        self._manager = browser_manager
        self._enabled = browser_enabled
        self._max_elements = max_elements_returned
        self._max_input_length = max_input_text_length
        self._interaction_timeout = interaction_timeout
        self._max_wait_timeout = max_wait_timeout

    @property
    def name(self) -> str:
        return "browser_interaction"

    @property
    def description(self) -> str:
        return (
            "Interact with browser page elements. "
            "Actions: inspect (list interactive elements), "
            "click (click an element by index or selector), "
            "fill (enter text into an input/textarea), "
            "select (choose a dropdown option), "
            "press (press a keyboard key), "
            "wait (wait for a page state or condition). "
            "No JavaScript execution is available."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["inspect", "click", "fill", "select", "press", "wait"],
                    "description": "The interaction action to perform",
                },
                "session_id": {
                    "type": "string",
                    "description": "The browser session ID",
                },
                "index": {
                    "type": "integer",
                    "description": "Element index from a previous inspect result",
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for targeting an element",
                },
                "value": {
                    "type": "string",
                    "description": "Value for fill/select actions",
                },
                "key": {
                    "type": "string",
                    "description": "Key name for press action (e.g. Enter, Tab, Escape)",
                },
                "condition": {
                    "type": "string",
                    "enum": ["selector", "text", "load_state", "url_pattern"],
                    "description": "Wait condition type",
                },
                "condition_value": {
                    "type": "string",
                    "description": "Value for the wait condition",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Action timeout in milliseconds",
                    "minimum": 100,
                    "maximum": 30000,
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
                "action": {"type": "string"},
                "page_url": {"type": "string"},
                "page_title": {"type": "string"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["browser.interact"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 30.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not self._enabled:
            return False, ["Browser automation is disabled"]
        action = arguments.get("action", "")
        if action not in VALID_INTERACTION_ACTIONS:
            errors.append(f"Invalid action: {action}. Must be one of: {', '.join(sorted(VALID_INTERACTION_ACTIONS))}")
        session_id = arguments.get("session_id", "")
        if not session_id:
            errors.append("session_id is required")
        if action == "click":
            if "index" not in arguments and "selector" not in arguments:
                errors.append("Either index or selector is required for click action")
        if action == "fill":
            if "index" not in arguments and "selector" not in arguments:
                errors.append("Either index or selector is required for fill action")
            value = arguments.get("value", "")
            if not value:
                errors.append("value is required for fill action")
            elif len(value) > self._max_input_length:
                errors.append(f"value exceeds maximum length of {self._max_input_length} characters")
        if action == "select":
            if "index" not in arguments and "selector" not in arguments:
                errors.append("Either index or selector is required for select action")
            value = arguments.get("value", "")
            if not value:
                errors.append("value is required for select action")
        if action == "press":
            key = arguments.get("key", "")
            if not key:
                errors.append("key is required for press action")
            elif key not in ALLOWED_KEYS:
                errors.append(f"Unsupported key: {key}. Use a standard key name.")
        if action == "wait":
            condition = arguments.get("condition", "")
            if not condition:
                errors.append("condition is required for wait action")
            elif condition not in VALID_WAIT_CONDITIONS:
                errors.append(f"Invalid condition: {condition}. Must be one of: {', '.join(sorted(VALID_WAIT_CONDITIONS))}")
            condition_value = arguments.get("condition_value", "")
            if not condition_value:
                errors.append("condition_value is required for wait action")
        return (len(errors) == 0, errors)

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()
        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(success=False, tool_name=self.name, error="; ".join(errors), execution_time=time.time() - start)
        action = arguments.get("action", "")
        session_id = arguments.get("session_id", "")
        try:
            session_mgr = self._manager.get_session(session_id)
            if session_mgr is None:
                return ToolResult(success=False, tool_name=self.name, error=f"Session not found: {session_id}", execution_time=time.time() - start)
            if session_mgr.is_closed:
                return ToolResult(success=False, tool_name=self.name, error=f"Session is closed: {session_id}", execution_time=time.time() - start)
            if action == "inspect":
                return self._execute_inspect(session_mgr, arguments, start)
            elif action == "click":
                return self._execute_click(session_mgr, arguments, start)
            elif action == "fill":
                return self._execute_fill(session_mgr, arguments, start)
            elif action == "select":
                return self._execute_select(session_mgr, arguments, start)
            elif action == "press":
                return self._execute_press(session_mgr, arguments, start)
            elif action == "wait":
                return self._execute_wait(session_mgr, arguments, start)
            else:
                return ToolResult(success=False, tool_name=self.name, error=f"Unknown action: {action}", execution_time=time.time() - start)
        except Exception as e:
            logger.error(f"Browser interaction tool error: {e}")
            return ToolResult(success=False, tool_name=self.name, error=f"Unexpected error: {str(e)}", execution_time=time.time() - start)

    def _execute_inspect(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        max_elements = arguments.get("max_elements", self._max_elements)
        result = session_mgr.inspect_page(max_elements=max_elements)
        exec_time = time.time() - start
        if result.get("error"):
            return ToolResult(success=False, tool_name=self.name, error=result["error"], execution_time=exec_time, metadata={"action": "inspect", "code": result.get("code", "UNKNOWN")})
        elements = result.get("elements", [])
        output_lines = [f"[BEGIN UNTRUSTED WEBPAGE CONTENT]"]
        for elem in elements:
            parts = [f"index={elem['index']}", f"tag={elem['tag']}"]
            if elem.get("role"):
                parts.append(f"role={elem['role']}")
            if elem.get("text"):
                parts.append(f"text={elem['text']}")
            if elem.get("aria_label"):
                parts.append(f"aria-label={elem['aria_label']}")
            if elem.get("placeholder"):
                parts.append(f"placeholder={elem['placeholder']}")
            if elem.get("input_type"):
                parts.append(f"type={elem['input_type']}")
            if elem.get("name"):
                parts.append(f"name={elem['name']}")
            if elem.get("href"):
                parts.append(f"href={elem['href']}")
            if elem.get("value"):
                parts.append(f"value={elem['value']}")
            parts.append(f"visible={elem['visible']}")
            parts.append(f"enabled={elem['enabled']}")
            output_lines.append("  ".join(parts))
        output_lines.append("[END UNTRUSTED WEBPAGE CONTENT]")
        return ToolResult(
            success=True,
            tool_name=self.name,
            output="\n".join(output_lines),
            execution_time=exec_time,
            metadata={
                "action": "inspect",
                "count": result.get("count", 0),
                "truncated": result.get("truncated", False),
                "page_url": sanitize_url_for_log(result.get("page_url", "")),
                "page_title": result.get("page_title", ""),
            },
        )

    def _execute_click(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        index = arguments.get("index")
        selector = arguments.get("selector")
        timeout = arguments.get("timeout", self._interaction_timeout)
        result = session_mgr.click_element(index=index, selector=selector, timeout=timeout)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        metadata = {
            "action": "click",
            "index": index,
            "selector": selector,
            "page_url": sanitized_url,
        }
        if result.get("success"):
            return ToolResult(success=True, tool_name=self.name, output="Element clicked", execution_time=exec_time, metadata=metadata)
        else:
            metadata["code"] = result.get("code", "UNKNOWN")
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Click failed"), execution_time=exec_time, metadata=metadata)

    def _execute_fill(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        index = arguments.get("index")
        selector = arguments.get("selector")
        value = arguments.get("value", "")
        timeout = arguments.get("timeout", self._interaction_timeout)
        result = session_mgr.fill_field(value=value, index=index, selector=selector, timeout=timeout)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        metadata = {
            "action": "fill",
            "index": index,
            "selector": selector,
            "value": "[REDACTED]",
            "value_length": len(value),
            "page_url": sanitized_url,
        }
        if result.get("success"):
            return ToolResult(success=True, tool_name=self.name, output="Field filled", execution_time=exec_time, metadata=metadata)
        else:
            metadata["code"] = result.get("code", "UNKNOWN")
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Fill failed"), execution_time=exec_time, metadata=metadata)

    def _execute_select(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        index = arguments.get("index")
        selector = arguments.get("selector")
        value = arguments.get("value", "")
        timeout = arguments.get("timeout", self._interaction_timeout)
        result = session_mgr.select_option(value=value, index=index, selector=selector, timeout=timeout)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        metadata = {
            "action": "select",
            "index": index,
            "selector": selector,
            "selected_value": value,
            "page_url": sanitized_url,
        }
        if result.get("success"):
            return ToolResult(success=True, tool_name=self.name, output=f"Option selected: {value}", execution_time=exec_time, metadata=metadata)
        else:
            metadata["code"] = result.get("code", "UNKNOWN")
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Select failed"), execution_time=exec_time, metadata=metadata)

    def _execute_press(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        index = arguments.get("index")
        selector = arguments.get("selector")
        key = arguments.get("key", "")
        timeout = arguments.get("timeout", self._interaction_timeout)
        result = session_mgr.press_key(key=key, index=index, selector=selector, timeout=timeout)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        metadata = {
            "action": "press",
            "index": index,
            "selector": selector,
            "key": key,
            "page_url": sanitized_url,
        }
        if result.get("success"):
            return ToolResult(success=True, tool_name=self.name, output=f"Key pressed: {key}", execution_time=exec_time, metadata=metadata)
        else:
            metadata["code"] = result.get("code", "UNKNOWN")
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Key press failed"), execution_time=exec_time, metadata=metadata)

    def _execute_wait(self, session_mgr, arguments: Dict[str, Any], start: float) -> ToolResult:
        condition = arguments.get("condition", "")
        condition_value = arguments.get("condition_value", "")
        timeout = arguments.get("timeout", self._max_wait_timeout)
        timeout = min(timeout, self._max_wait_timeout)
        kwargs = {"timeout": timeout}
        if condition == "selector":
            kwargs["selector"] = condition_value
        elif condition == "text":
            kwargs["text"] = condition_value
        elif condition == "load_state":
            kwargs["state"] = condition_value
        elif condition == "url_pattern":
            kwargs["url_pattern"] = condition_value
        result = session_mgr.wait_for_state(**kwargs)
        exec_time = time.time() - start
        sanitized_url = sanitize_url_for_log(result.get("page_url", ""))
        metadata = {
            "action": "wait",
            "condition": condition,
            "page_url": sanitized_url,
        }
        if result.get("success"):
            return ToolResult(success=True, tool_name=self.name, output=f"Wait completed: {condition}", execution_time=exec_time, metadata=metadata)
        else:
            metadata["code"] = result.get("code", "UNKNOWN")
            return ToolResult(success=False, tool_name=self.name, error=result.get("error", "Wait failed"), execution_time=exec_time, metadata=metadata)
