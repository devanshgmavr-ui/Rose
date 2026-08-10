"""Browser session management.

Stage 2.4.2 - Browser navigation support.

Manages a single isolated Playwright browser session.
Each session uses its own BrowserContext with no access
to user profiles, cookies, or credentials.
"""

import time
import logging
from typing import Optional, Any, Dict

from .models import BrowserSession, BrowserPage, SessionStatus

logger = logging.getLogger(__name__)


class BrowserSessionManager:
    """Manages a single Playwright browser session.

    Wraps Playwright Browser, BrowserContext, and initial Page
    into an isolated session with safe cleanup.
    """

    def __init__(
        self,
        session: BrowserSession,
        browser: Any,
        context: Any,
        page: Any,
    ):
        """Initialize session manager.

        Args:
            session: Session metadata model.
            browser: Playwright Browser object.
            context: Playwright BrowserContext object.
            page: Playwright Page object.
        """
        self._session = session
        self._browser = browser
        self._context = context
        self._page = page
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def session(self) -> BrowserSession:
        return self._session

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def page(self):
        """Get the current active page."""
        return self._page

    @property
    def context(self):
        """Get the browser context."""
        return self._context

    @property
    def browser(self):
        """Get the browser instance."""
        return self._browser

    def get_current_url(self) -> str:
        """Get current page URL safely."""
        if self._closed or not self._page:
            return ""
        try:
            return self._page.url
        except Exception:
            return ""

    def get_current_title(self) -> str:
        """Get current page title safely."""
        if self._closed or not self._page:
            return ""
        try:
            return self._page.title()
        except Exception:
            return ""

    def read_page_text(
        self, tab_index: int = 0,
        max_chars: int = 20000, max_tokens: int = 5000,
    ) -> Dict[str, Any]:
        """Read text content from a browser page.

        Stage 2.4.3 - Browser page reading.

        Extracts text content from a browser page using Playwright's
        inner_text() API. Content is returned as UNTRUSTED DATA with
        appropriate security warnings. Never written to audit logs.

        Args:
            tab_index: Tab index (0 = current page).
            max_chars: Maximum characters to return.
            max_tokens: Approximate maximum tokens to return.

        Returns:
            Dict with keys: content, truncated, char_count, token_estimate,
            page_url, page_title, content_wrapped.
        """
        if self._closed:
            return {"error": "Session is closed", "code": "SESSION_CLOSED"}

        if not self._page:
            return {"error": "No page available", "code": "NO_PAGE"}

        try:
            inner_text = self._page.inner_text("body")
            final_max_chars = min(max_chars, max_tokens * 4)

            truncated = len(inner_text) > final_max_chars
            if truncated:
                inner_text = inner_text[:final_max_chars]

            token_estimate = len(inner_text) // 4

            return {
                "content": inner_text,
                "truncated": truncated,
                "char_count": len(inner_text),
                "token_estimate": token_estimate,
                "page_url": self._page.url,
                "page_title": self._page.title(),
                "content_wrapped": True,
            }

        except Exception as e:
            return {"error": f"Failed to read page: {e}", "code": "READ_FAILED"}

    def navigate(self, url: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Navigate the current page to a URL.

        Args:
            url: The URL to navigate to.
            timeout: Optional navigation timeout in milliseconds.
                     If None, uses the context default.

        Returns:
            Dictionary with navigation result metadata.
        """
        if self._closed:
            return {"success": False, "error": "Session is closed"}

        if not self._page:
            return {"success": False, "error": "No page available"}

        try:
            nav_start = time.time()

            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout

            response = self._page.goto(url, **kwargs)

            nav_time_ms = int((time.time() - nav_start) * 1000)

            final_url = self._page.url
            title = ""
            try:
                title = self._page.title()
            except Exception:
                pass

            status = None
            if response:
                status = response.status

            self._session.current_url = final_url
            self._session.current_title = title

            return {
                "success": True,
                "url": url,
                "final_url": final_url,
                "title": title,
                "status": status,
                "navigation_time_ms": nav_time_ms,
            }

        except Exception as e:
            nav_time_ms = int((time.time() - nav_start) * 1000) if 'nav_start' in dir() else 0
            error_type = type(e).__name__
            error_msg = str(e)

            if "timeout" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Navigation timeout after {nav_time_ms}ms",
                    "error_type": "timeout",
                    "navigation_time_ms": nav_time_ms,
                }

            return {
                "success": False,
                "error": f"Navigation failed: {error_msg}",
                "error_type": error_type,
                "navigation_time_ms": nav_time_ms,
            }

    def update_state(self):
        """Update session metadata from Playwright state."""
        if self._closed:
            return
        self._session.current_url = self.get_current_url()
        self._session.current_title = self.get_current_title()
        try:
            pages = self._context.pages if self._context else []
            self._session.page_count = len(pages)
        except Exception:
            pass

    def close(self) -> bool:
        """Close the browser session and release resources.

        Returns:
            True if closed successfully, False if already closed.
        """
        if self._closed:
            return False

        self._closed = True
        self._session.status = SessionStatus.CLOSING

        try:
            if self._context:
                self._context.close()
        except Exception as e:
            logger.warning(f"Error closing context for {self.session_id}: {e}")

        try:
            if self._browser:
                self._browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser for {self.session_id}: {e}")

        self._session.status = SessionStatus.CLOSED
        self._session.page_count = 0
        logger.info(f"Browser session {self.session_id} closed")
        return True

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        if not self._closed:
            self.close()
