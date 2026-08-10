"""OS automation and computer perception tools."""

from .screen import ScreenCaptureTool
from .system import SystemInfoTool
from .mouse import MouseTool
from .keyboard import KeyboardTool
from .windows import WindowTool
from .permissions import register_os_permissions, OS_PERMISSIONS, OS_PERMISSION_SCOPES

__all__ = [
    "ScreenCaptureTool",
    "SystemInfoTool",
    "MouseTool",
    "KeyboardTool",
    "WindowTool",
    "register_os_permissions",
    "OS_PERMISSIONS",
    "OS_PERMISSION_SCOPES",
]
