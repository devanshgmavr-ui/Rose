"""Browser lifecycle manager.

Stage 2.4.1 - Browser foundation.

Manages Playwright initialization, browser launch,
session creation, tracking, and cleanup.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List

from .models import BrowserSession, SessionStatus
from .session import BrowserSessionManager

logger = logging.getLogger(__name__)

DEFAULT_MAX_SESSIONS = 2
DEFAULT_MAX_TABS = 5
DEFAULT_NAVIGATION_TIMEOUT = 30000
DEFAULT_ACTION_TIMEOUT = 10000


class BrowserManager:
    """Manages Playwright browser lifecycle and sessions.

    Provides centralized control over browser creation,
    session tracking, and resource cleanup.
    """

    def __init__(
        self,
        headless: bool = True,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_tabs: int = DEFAULT_MAX_TABS,
        navigation_timeout: int = DEFAULT_NAVIGATION_TIMEOUT,
        action_timeout: int = DEFAULT_ACTION_TIMEOUT,
    ):
        """Initialize the browser manager.

        Args:
            headless: Run browser in headless mode.
            max_sessions: Maximum concurrent browser sessions.
            max_tabs: Maximum pages per session.
            navigation_timeout: Navigation timeout in milliseconds.
            action_timeout: Action timeout in milliseconds.
        """
        self._headless = headless
        self._max_sessions = max_sessions
        self._max_tabs = max_tabs
        self._navigation_timeout = navigation_timeout
        self._action_timeout = action_timeout

        self._playwright = None
        self._browser_type = None
        self._sessions: Dict[str, BrowserSessionManager] = {}
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def initialize(self) -> bool:
        """Initialize Playwright and prepare for browser launch.

        Returns:
            True if initialization successful, False otherwise.
        """
        if self._initialized:
            return True

        try:
            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser_type = self._playwright.chromium
            self._initialized = True
            logger.info("Browser manager initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser manager: {e}")
            return False

    def create_session(self) -> Optional[BrowserSessionManager]:
        """Create a new isolated browser session.

        Returns:
            BrowserSessionManager if successful, None otherwise.
        """
        with self._lock:
            if not self._initialized:
                logger.error("Browser manager not initialized")
                return None

            if len(self._sessions) >= self._max_sessions:
                logger.warning(
                    f"Session limit reached: {len(self._sessions)}/{self._max_sessions}"
                )
                return None

            try:
                browser = self._browser_type.launch(headless=self._headless)
                context = browser.new_context()
                context.set_default_navigation_timeout(self._navigation_timeout)
                context.set_default_timeout(self._action_timeout)
                page = context.new_page()

                session_model = BrowserSession(
                    session_id=BrowserSession.generate_id(),
                    created_at=time.time(),
                    headless=self._headless,
                    status=SessionStatus.ACTIVE,
                    page_count=1,
                )

                manager = BrowserSessionManager(
                    session=session_model,
                    browser=browser,
                    context=context,
                    page=page,
                )

                self._sessions[session_model.session_id] = manager
                logger.info(
                    f"Created browser session {session_model.session_id} "
                    f"(headless={self._headless})"
                )
                return manager

            except Exception as e:
                logger.error(f"Failed to create browser session: {e}")
                return None

    def get_session(self, session_id: str) -> Optional[BrowserSessionManager]:
        """Get a session by ID.

        Args:
            session_id: The session identifier.

        Returns:
            BrowserSessionManager if found, None otherwise.
        """
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with safe metadata.

        Returns:
            List of session metadata dictionaries.
        """
        with self._lock:
            result = []
            for manager in self._sessions.values():
                manager.update_state()
                result.append(manager.session.to_dict())
            return result

    def close_session(self, session_id: str) -> bool:
        """Close a specific session.

        Args:
            session_id: The session to close.

        Returns:
            True if closed, False if not found or already closed.
        """
        with self._lock:
            manager = self._sessions.get(session_id)
            if manager is None:
                return False

            closed = manager.close()
            if closed:
                del self._sessions[session_id]
            return closed

    def close_all(self) -> int:
        """Close all active sessions.

        Returns:
            Number of sessions closed.
        """
        with self._lock:
            count = 0
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                manager = self._sessions[session_id]
                if manager.close():
                    count += 1
            self._sessions.clear()
            logger.info(f"Closed {count} browser sessions")
            return count

    def _close_all_unlocked(self) -> int:
        """Close all sessions without acquiring the lock (internal)."""
        count = 0
        session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            manager = self._sessions[session_id]
            if manager.close():
                count += 1
        self._sessions.clear()
        logger.info(f"Closed {count} browser sessions")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get browser manager statistics.

        Returns:
            Dictionary with manager stats.
        """
        with self._lock:
            return {
                "initialized": self._initialized,
                "headless": self._headless,
                "active_sessions": len(self._sessions),
                "max_sessions": self._max_sessions,
                "max_tabs": self._max_tabs,
                "navigation_timeout": self._navigation_timeout,
                "action_timeout": self._action_timeout,
                "session_ids": list(self._sessions.keys()),
            }

    def health_check(self) -> Dict[str, Any]:
        """Check browser manager health.

        Returns:
            Health status dictionary.
        """
        return {
            "initialized": self._initialized,
            "headless": self._headless,
            "active_sessions": self.session_count,
            "max_sessions": self._max_sessions,
            "playwright_available": self._playwright is not None,
        }

    def shutdown(self):
        """Shutdown the browser manager and release all resources."""
        with self._lock:
            self._close_all_unlocked()

            self._browser_type = None
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception as e:
                    logger.warning(f"Error stopping Playwright: {e}")
                self._playwright = None

            self._initialized = False
            logger.info("Browser manager shutdown complete")
