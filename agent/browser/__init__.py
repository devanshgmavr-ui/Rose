"""Browser automation subsystem.

Stage 2.4.3 - Browser page reading support.

Provides Playwright-based browser session management
with isolated contexts, configurable security, safe
URL navigation, and page text extraction.
"""

from .models import BrowserSession, BrowserPage, SessionStatus
from .session import BrowserSessionManager
from .manager import BrowserManager
from .permissions import register_browser_permissions, BROWSER_PERMISSIONS, BROWSER_PERMISSION_SCOPES
from .policy import validate_url, sanitize_url_for_log
from .tools import BrowserSessionTool, BrowserNavigationTool, BrowserPageReadTool

__all__ = [
    "BrowserSession",
    "BrowserPage",
    "SessionStatus",
    "BrowserSessionManager",
    "BrowserManager",
    "BrowserSessionTool",
    "BrowserNavigationTool",
    "BrowserPageReadTool",
    "register_browser_permissions",
    "BROWSER_PERMISSIONS",
    "BROWSER_PERMISSION_SCOPES",
    "validate_url",
    "sanitize_url_for_log",
]
