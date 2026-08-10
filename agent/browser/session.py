"""Browser session management.

Stage 2.4.5 - Browser screenshot support.

Manages a single isolated Playwright browser session.
Each session uses its own BrowserContext with no access
to user profiles, cookies, or credentials.
"""

import time
import logging
from typing import Optional, Any, Dict, List

try:
    from PIL import Image
except ImportError:
    Image = None

from .models import BrowserSession, BrowserPage, SessionStatus

logger = logging.getLogger(__name__)

VALID_WAIT_STATES = {"load", "domcontentloaded", "networkidle", "commit"}


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
        return self._page

    @property
    def context(self):
        return self._context

    @property
    def browser(self):
        return self._browser

    def get_current_url(self) -> str:
        if self._closed or not self._page:
            return ""
        try:
            return self._page.url
        except Exception:
            return ""

    def get_current_title(self) -> str:
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

    def inspect_page(
        self, max_elements: int = 100, max_text_length: int = 200,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"error": "No page available", "code": "NO_PAGE"}
        try:
            elements = []
            selector = (
                "a, button, input, textarea, select, "
                "[role='button'], [role='link'], [role='tab'], "
                "[role='checkbox'], [role='radio'], [role='menuitem'], "
                "[onclick], [tabindex]"
            )
            locators = self._page.locator(selector).all()
            count = min(len(locators), max_elements)
            for i in range(count):
                loc = locators[i]
                try:
                    tag = loc.evaluate("el => el.tagName.toLowerCase()")
                    visible = loc.is_visible()
                    enabled = loc.is_enabled()
                    text = ""
                    try:
                        text = loc.inner_text()
                        if len(text) > max_text_length:
                            text = text[:max_text_length]
                    except Exception:
                        pass
                    aria_label = ""
                    try:
                        aria_label = loc.get_attribute("aria-label") or ""
                    except Exception:
                        pass
                    placeholder = ""
                    try:
                        placeholder = loc.get_attribute("placeholder") or ""
                    except Exception:
                        pass
                    input_type = ""
                    try:
                        input_type = loc.get_attribute("type") or ""
                    except Exception:
                        pass
                    name = ""
                    try:
                        name = loc.get_attribute("name") or ""
                    except Exception:
                        pass
                    href = ""
                    try:
                        href = loc.get_attribute("href") or ""
                    except Exception:
                        pass
                    role = ""
                    try:
                        role = loc.get_attribute("role") or ""
                    except Exception:
                        pass
                    value = ""
                    try:
                        if tag in ("input", "textarea"):
                            value = loc.input_value()
                            if len(value) > max_text_length:
                                value = value[:max_text_length]
                    except Exception:
                        pass
                    elements.append({
                        "index": i,
                        "tag": tag,
                        "role": role,
                        "text": text,
                        "aria_label": aria_label,
                        "placeholder": placeholder,
                        "input_type": input_type,
                        "name": name,
                        "href": href,
                        "value": value,
                        "visible": visible,
                        "enabled": enabled,
                    })
                except Exception:
                    continue
            return {
                "elements": elements,
                "count": len(elements),
                "truncated": len(locators) > max_elements,
                "page_url": self._page.url,
                "page_title": self._page.title(),
                "content_wrapped": True,
            }
        except Exception as e:
            return {"error": f"Failed to inspect page: {e}", "code": "INSPECT_FAILED"}

    def _resolve_locator(
        self, index: Optional[int] = None, selector: Optional[str] = None,
    ):
        if index is not None:
            interactive_selector = (
                "a, button, input, textarea, select, "
                "[role='button'], [role='link'], [role='tab'], "
                "[role='checkbox'], [role='radio'], [role='menuitem'], "
                "[onclick], [tabindex]"
            )
            locators = self._page.locator(interactive_selector).all()
            if index < 0 or index >= len(locators):
                return None
            return locators[index]
        if selector:
            return self._page.locator(selector).first
        return None

    def click_element(
        self, index: Optional[int] = None,
        selector: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            loc = self._resolve_locator(index, selector)
            if loc is None:
                return {"success": False, "error": "Could not resolve target element", "code": "INVALID_TARGET"}
            if not loc.is_visible():
                return {"success": False, "error": "Element is not visible", "code": "NOT_VISIBLE"}
            if not loc.is_enabled():
                return {"success": False, "error": "Element is disabled", "code": "DISABLED"}
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            loc.click(**kwargs)
            return {
                "success": True,
                "page_url": self._page.url,
                "page_title": self._page.title(),
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Click timed out", "code": "TIMEOUT"}
            if "detached" in error_msg.lower():
                return {"success": False, "error": "Element became detached from DOM", "code": "STALE_ELEMENT"}
            return {"success": False, "error": f"Click failed: {error_msg}", "code": "CLICK_FAILED"}

    def fill_field(
        self, value: str,
        index: Optional[int] = None,
        selector: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            loc = self._resolve_locator(index, selector)
            if loc is None:
                return {"success": False, "error": "Could not resolve target element", "code": "INVALID_TARGET"}
            if not loc.is_visible():
                return {"success": False, "error": "Element is not visible", "code": "NOT_VISIBLE"}
            if not loc.is_enabled():
                return {"success": False, "error": "Element is disabled", "code": "DISABLED"}
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            loc.fill(value, **kwargs)
            return {
                "success": True,
                "page_url": self._page.url,
                "page_title": self._page.title(),
                "value_length": len(value),
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Fill timed out", "code": "TIMEOUT"}
            if "detached" in error_msg.lower():
                return {"success": False, "error": "Element became detached from DOM", "code": "STALE_ELEMENT"}
            return {"success": False, "error": f"Fill failed: {error_msg}", "code": "FILL_FAILED"}

    def select_option(
        self, value: str,
        index: Optional[int] = None,
        selector: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            loc = self._resolve_locator(index, selector)
            if loc is None:
                return {"success": False, "error": "Could not resolve target element", "code": "INVALID_TARGET"}
            if not loc.is_visible():
                return {"success": False, "error": "Element is not visible", "code": "NOT_VISIBLE"}
            if not loc.is_enabled():
                return {"success": False, "error": "Element is disabled", "code": "DISABLED"}
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = timeout
            loc.select_option(value=value, **kwargs)
            return {
                "success": True,
                "page_url": self._page.url,
                "page_title": self._page.title(),
                "selected_value": value,
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Select timed out", "code": "TIMEOUT"}
            if "detached" in error_msg.lower():
                return {"success": False, "error": "Element became detached from DOM", "code": "STALE_ELEMENT"}
            return {"success": False, "error": f"Select failed: {error_msg}", "code": "SELECT_FAILED"}

    def press_key(
        self, key: str,
        index: Optional[int] = None,
        selector: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            if index is not None or selector is not None:
                loc = self._resolve_locator(index, selector)
                if loc is None:
                    return {"success": False, "error": "Could not resolve target element", "code": "INVALID_TARGET"}
                if not loc.is_visible():
                    return {"success": False, "error": "Element is not visible", "code": "NOT_VISIBLE"}
                kwargs = {}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                loc.press(key, **kwargs)
            else:
                self._page.keyboard.press(key)
            return {
                "success": True,
                "page_url": self._page.url,
                "page_title": self._page.title(),
                "key": key,
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Key press timed out", "code": "TIMEOUT"}
            if "detached" in error_msg.lower():
                return {"success": False, "error": "Element became detached from DOM", "code": "STALE_ELEMENT"}
            return {"success": False, "error": f"Key press failed: {error_msg}", "code": "PRESS_FAILED"}

    def wait_for_state(
        self, state: str = "load", timeout: Optional[int] = None,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        url_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            effective_timeout = timeout if timeout is not None else 15000
            effective_timeout = min(effective_timeout, 30000)
            if selector:
                self._page.wait_for_selector(selector, timeout=effective_timeout)
                return {"success": True, "condition": "selector", "page_url": self._page.url}
            if text:
                self._page.wait_for_selector(f"text={text}", timeout=effective_timeout)
                return {"success": True, "condition": "text", "page_url": self._page.url}
            if url_pattern:
                self._page.wait_for_url(f"**{url_pattern}**", timeout=effective_timeout)
                return {"success": True, "condition": "url_pattern", "page_url": self._page.url}
            if state in VALID_WAIT_STATES:
                self._page.wait_for_load_state(state=state, timeout=effective_timeout)
                return {"success": True, "condition": "load_state", "state": state, "page_url": self._page.url}
            return {"success": False, "error": f"Invalid wait condition: {state}", "code": "INVALID_CONDITION"}
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Wait timed out", "code": "TIMEOUT"}
            return {"success": False, "error": f"Wait failed: {error_msg}", "code": "WAIT_FAILED"}

    def screenshot_viewport(
        self,
        path: str,
        full_page: bool = False,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture a screenshot of the current viewport or full page.

        Args:
            path: File path to save the screenshot (PNG).
            full_page: If True, capture the full scrollable page.
            timeout: Optional timeout in milliseconds.

        Returns:
            Dict with success, path, width, height, size_bytes or error.
        """
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            kwargs = {"path": path, "full_page": full_page}
            if timeout is not None:
                kwargs["timeout"] = timeout
            self._page.screenshot(**kwargs)
            import os
            size_bytes = os.path.getsize(path)
            try:
                from PIL import Image
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                width = 0
                height = 0
            return {
                "success": True,
                "path": path,
                "type": "full_page" if full_page else "viewport",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Screenshot timed out", "code": "TIMEOUT"}
            return {"success": False, "error": f"Screenshot failed: {error_msg}", "code": "SCREENSHOT_FAILED"}

    def screenshot_element(
        self,
        path: str,
        index: Optional[int] = None,
        selector: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Capture a screenshot of a specific page element.

        Args:
            path: File path to save the screenshot (PNG).
            index: Element index from inspect_page results.
            selector: CSS selector for targeting an element.
            timeout: Optional timeout in milliseconds.

        Returns:
            Dict with success, path, width, height, size_bytes or error.
        """
        if self._closed:
            return {"success": False, "error": "Session is closed", "code": "SESSION_CLOSED"}
        if not self._page:
            return {"success": False, "error": "No page available", "code": "NO_PAGE"}
        try:
            loc = self._resolve_locator(index, selector)
            if loc is None:
                return {"success": False, "error": "Could not resolve target element", "code": "INVALID_TARGET"}
            if not loc.is_visible():
                return {"success": False, "error": "Element is not visible", "code": "NOT_VISIBLE"}
            kwargs = {"path": path}
            if timeout is not None:
                kwargs["timeout"] = timeout
            loc.screenshot(**kwargs)
            import os
            size_bytes = os.path.getsize(path)
            try:
                from PIL import Image
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                width = 0
                height = 0
            return {
                "success": True,
                "path": path,
                "type": "element",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
            }
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return {"success": False, "error": "Element screenshot timed out", "code": "TIMEOUT"}
            if "detached" in error_msg.lower():
                return {"success": False, "error": "Element became detached from DOM", "code": "STALE_ELEMENT"}
            return {"success": False, "error": f"Element screenshot failed: {error_msg}", "code": "SCREENSHOT_FAILED"}

    def update_state(self):
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
        if not self._closed:
            self.close()
