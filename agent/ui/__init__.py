"""Rose UI - PySide6 desktop interface."""

from .app import UIConfig, UITheme, ChatMessageUI, MessageDisplay, InputHandler
from .panels import (
    RoseUI, TaskPanel, TaskInfo, SettingsPanel, StatusBar,
    ConfirmationDialog, Sidebar,
)

__all__ = [
    "RoseUI", "UIConfig", "UITheme", "ChatMessageUI", "MessageDisplay", "InputHandler",
    "TaskPanel", "TaskInfo", "SettingsPanel", "StatusBar", "ConfirmationDialog", "Sidebar",
]
