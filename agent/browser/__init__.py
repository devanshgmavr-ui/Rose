"""Browser automation subsystem.

Stage 2.4.5 - Browser screenshot support.

Provides Playwright-based browser session management
with isolated contexts, configurable security, safe
URL navigation, page text extraction, controlled
page interaction, and screenshot capture.
"""

from .models import BrowserSession, BrowserPage, SessionStatus
from .session import BrowserSessionManager
from .manager import BrowserManager
from .permissions import register_browser_permissions, BROWSER_PERMISSIONS, BROWSER_PERMISSION_SCOPES
from .policy import validate_url, sanitize_url_for_log
from .tools import BrowserSessionTool, BrowserNavigationTool, BrowserPageReadTool, BrowserInteractionTool, BrowserScreenshotTool

__all__ = [
    "BrowserSession",
    "BrowserPage",
    "SessionStatus",
    "BrowserSessionManager",
    "BrowserManager",
    "BrowserSessionTool",
    "BrowserNavigationTool",
    "BrowserPageReadTool",
    "BrowserInteractionTool",
    "BrowserScreenshotTool",
    "register_browser_permissions",
    "BROWSER_PERMISSIONS",
    "BROWSER_PERMISSION_SCOPES",
    "validate_url",
    "sanitize_url_for_log",
]
