"""Rose Desktop UI - PySide6 application.

Phase 6 - Rose User Interface.

Provides:
- Main window with toolbar, sidebar, and status bar
- Chat panel with message display and input
- Task panel for autonomous task management
- Settings panel for configuration
- Confirmation dialog for dangerous actions
- Status bar showing agent health
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class UITheme(Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class UIConfig:
    theme: UITheme = UITheme.DARK
    window_width: int = 1200
    window_height: int = 800
    sidebar_width: int = 250
    show_sidebar: bool = True
    show_toolbar: bool = True
    show_statusbar: bool = True
    font_size: int = 14
    max_message_length: int = 10000
    auto_scroll: bool = True
    confirmation_timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme": self.theme.value,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "sidebar_width": self.sidebar_width,
            "show_sidebar": self.show_sidebar,
            "show_toolbar": self.show_toolbar,
            "show_statusbar": self.show_statusbar,
            "font_size": self.font_size,
            "auto_scroll": self.auto_scroll,
        }


@dataclass
class ChatMessageUI:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


class MessageDisplay:
    """Manages chat message display."""

    def __init__(self, max_messages: int = 500):
        self._messages: List[ChatMessageUI] = []
        self._max_messages = max_messages
        self._on_message_callbacks: List[Callable] = []

    def add_message(self, role: str, content: str, **kwargs) -> ChatMessageUI:
        msg = ChatMessageUI(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=kwargs.get("message_id", ""),
            metadata=kwargs.get("metadata", {}),
        )
        self._messages.append(msg)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]
        for cb in self._on_message_callbacks:
            try:
                cb(msg)
            except Exception:
                pass
        return msg

    def get_messages(self, limit: int = 100) -> List[ChatMessageUI]:
        return self._messages[-limit:]

    def get_all_messages(self) -> List[ChatMessageUI]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()

    def message_count(self) -> int:
        return len(self._messages)

    def on_message(self, callback: Callable):
        self._on_message_callbacks.append(callback)

    def search(self, query: str) -> List[ChatMessageUI]:
        q = query.lower()
        return [m for m in self._messages if q in m.content.lower()]

    def delete_message(self, message_id: str) -> bool:
        before = len(self._messages)
        self._messages = [m for m in self._messages if m.message_id != message_id]
        return len(self._messages) < before


class InputHandler:
    """Handles user input with validation."""

    def __init__(self, max_length: int = 10000):
        self._max_length = max_length
        self._history: List[str] = []
        self._history_index: int = -1
        self._submit_callbacks: List[Callable] = []
        self._is_processing: bool = False

    def validate(self, text: str) -> tuple:
        if not text or not text.strip():
            return False, "Empty message"
        if len(text) > self._max_length:
            return False, f"Message too long (max {self._max_length} chars)"
        if self._is_processing:
            return False, "Already processing a message"
        return True, ""

    def on_submit(self, callback: Callable):
        self._submit_callbacks.append(callback)

    def submit(self, text: str) -> tuple:
        valid, error = self.validate(text)
        if not valid:
            return False, error

        self._history.append(text)
        self._history_index = len(self._history)

        for cb in self._submit_callbacks:
            try:
                cb(text)
            except Exception as e:
                logger.warning(f"Submit callback error: {e}")
        return True, ""

    def set_processing(self, value: bool):
        self._is_processing = value

    def is_processing(self) -> bool:
        return self._is_processing

    def get_history(self) -> List[str]:
        return list(self._history)

    def navigate_history(self, direction: int) -> Optional[str]:
        if not self._history:
            return None
        new_index = self._history_index + direction
        new_index = max(-1, min(len(self._history) - 1, new_index))
        self._history_index = new_index
        if 0 <= self._history_index < len(self._history):
            return self._history[self._history_index]
        return ""

    def clear_history(self):
        self._history.clear()
        self._history_index = -1
