"""Window management tool using Windows ctypes API.

Stage 2.3.4 - Advanced window operations.

Provides safe window enumeration, state control, graceful close,
position/size manipulation. All mutations require confirmation and
are audited.
"""

import time
import ctypes
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict

from ..tools.base import Tool, ToolResult, ConfirmationLevel

logger = logging.getLogger(__name__)

MAX_WINDOWS_RETURNED = 100
WINDOW_ENUM_TIMEOUT = 5.0
MAX_WINDOW_ACTIONS_PER_REQUEST = 10

WM_CLOSE = 0x0010
SW_MINIMIZE = 6
SW_RESTORE = 9
SW_MAXIMIZE = 3

DEFAULT_MIN_WINDOW_WIDTH = 100
DEFAULT_MIN_WINDOW_HEIGHT = 100
DEFAULT_MAX_WINDOW_WIDTH = 4096
DEFAULT_MAX_WINDOW_HEIGHT = 4096

PROTECTED_CLASS_NAMES = {
    "Progman",
    "Shell_TrayWnd",
    "Shell_SecondaryTrayWnd",
    "WorkerW",
    "Button",
}

PROTECTED_TITLES = {"Program Manager"}

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


@dataclass
class WindowInfo:
    """Structured representation of a window's metadata."""
    hwnd: int
    title: str
    class_name: str
    pid: int
    visible: bool
    minimized: bool
    maximized: bool
    rect: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_window_title(hwnd: int) -> str:
    try:
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def _get_class_name(hwnd: int) -> str:
    try:
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value
    except Exception:
        return ""


def _get_window_pid(hwnd: int) -> int:
    try:
        pid = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value
    except Exception:
        return 0


def _get_window_rect(hwnd: int) -> Dict[str, int]:
    try:
        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return {
                "x": rect.left,
                "y": rect.top,
                "width": rect.right - rect.left,
                "height": rect.bottom - rect.top,
            }
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    except Exception:
        return {"x": 0, "y": 0, "width": 0, "height": 0}


def _is_window_visible(hwnd: int) -> bool:
    try:
        return user32.IsWindowVisible(hwnd) != 0
    except Exception:
        return False


def _is_iconic(hwnd: int) -> bool:
    try:
        return user32.IsIconic(hwnd) != 0
    except Exception:
        return False


def _is_zoomed(hwnd: int) -> bool:
    try:
        return user32.IsZoomed(hwnd) != 0
    except Exception:
        return False


def _is_protected_window(hwnd: int) -> bool:
    """Check if a window is a protected system window."""
    try:
        class_name = _get_class_name(hwnd)
        if class_name in PROTECTED_CLASS_NAMES:
            return True

        title = _get_window_title(hwnd)
        if title in PROTECTED_TITLES:
            return True

        if hwnd == user32.GetDesktopWindow():
            return True

        return False
    except Exception:
        return False


def _get_window_info(hwnd: int) -> WindowInfo:
    title = _get_window_title(hwnd)
    class_name = _get_class_name(hwnd)
    pid = _get_window_pid(hwnd)
    visible = _is_window_visible(hwnd)
    minimized = _is_iconic(hwnd)
    maximized = _is_zoomed(hwnd)
    rect = _get_window_rect(hwnd)

    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name=class_name,
        pid=pid,
        visible=visible,
        minimized=minimized,
        maximized=maximized,
        rect=rect,
    )


def enumerate_windows(max_windows: int = MAX_WINDOWS_RETURNED) -> List[WindowInfo]:
    """Enumerate all visible top-level windows.

    Args:
        max_windows: Maximum number of windows to return.

    Returns:
        List of WindowInfo objects for visible windows.
    """
    windows = []

    def callback(hwnd, lparam):
        if len(windows) >= max_windows:
            return False

        if not _is_window_visible(hwnd):
            return True

        try:
            info = _get_window_info(hwnd)
            if info.title:
                windows.append(info)
        except Exception as e:
            logger.debug(f"Skipping window {hwnd}: {e}")

        return True

    WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows(WNDPROC(callback), 0)

    return windows


class WindowTool(Tool):
    """Window management tool with enumeration and state control.

    Read-only actions: list, get_active (ALLOW)
    Mutation actions: activate, minimize, restore, maximize,
                      close, move, resize, set_bounds (REQUIRE_CONFIRMATION)
    """

    READ_ONLY_ACTIONS = {"list", "get_active"}
    MUTATION_ACTIONS = {"activate", "minimize", "restore", "maximize",
                        "close", "move", "resize", "set_bounds"}
    ALL_ACTIONS = READ_ONLY_ACTIONS | MUTATION_ACTIONS

    def __init__(
        self,
        enabled: bool = False,
        control_enabled: bool = False,
        close_enabled: bool = False,
        move_enabled: bool = False,
        resize_enabled: bool = False,
        min_width: int = DEFAULT_MIN_WINDOW_WIDTH,
        min_height: int = DEFAULT_MIN_WINDOW_HEIGHT,
        max_width: int = DEFAULT_MAX_WINDOW_WIDTH,
        max_height: int = DEFAULT_MAX_WINDOW_HEIGHT,
    ):
        self._enabled = enabled
        self._control_enabled = control_enabled
        self._close_enabled = close_enabled
        self._move_enabled = move_enabled
        self._resize_enabled = resize_enabled
        self._min_width = min_width
        self._min_height = min_height
        self._max_width = max_width
        self._max_height = max_height

    @property
    def name(self) -> str:
        return "window"

    @property
    def description(self) -> str:
        return "List, inspect, and control windows (activate, minimize, restore, maximize, close, move, resize)"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(self.ALL_ACTIONS),
                    "description": "Window action to perform",
                },
                "hwnd": {
                    "type": "integer",
                    "description": "Window handle (required for mutations)",
                },
                "filter_title": {
                    "type": "string",
                    "description": "Optional filter: return only windows containing this title",
                },
                "x": {
                    "type": "integer",
                    "description": "X coordinate for move/set_bounds",
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate for move/set_bounds",
                },
                "width": {
                    "type": "integer",
                    "description": "Width for resize/set_bounds",
                },
                "height": {
                    "type": "integer",
                    "description": "Height for resize/set_bounds",
                },
            },
            "required": ["action"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "windows": {"type": "array"},
                "count": {"type": "integer"},
                "active_window": {"type": "object"},
                "window": {"type": "object"},
                "action": {"type": "string"},
                "close_requested": {"type": "boolean"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["os.window"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        if self._control_enabled:
            return ConfirmationLevel.REQUIRE_CONFIRMATION
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return WINDOW_ENUM_TIMEOUT

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not self._enabled:
            return False, ["Window control is not enabled"]

        errors = []
        action = arguments.get("action")
        if action not in self.ALL_ACTIONS:
            errors.append(f"Invalid action: {action}. Must be one of {list(self.ALL_ACTIONS)}")

        if action in self.MUTATION_ACTIONS:
            if not self._control_enabled:
                errors.append(f"Window control is not enabled for mutation action: {action}")

            if action == "close" and not self._close_enabled:
                errors.append("Window close is not enabled")

            if action == "move" and not self._move_enabled:
                errors.append("Window move is not enabled")

            if action in ("resize", "set_bounds") and not self._resize_enabled:
                errors.append(f"Window resize is not enabled for action: {action}")

            hwnd = arguments.get("hwnd")
            if hwnd is None:
                errors.append(f"Window handle (hwnd) is required for action: {action}")
            elif not isinstance(hwnd, int) or hwnd <= 0:
                errors.append(f"Invalid window handle: {hwnd}. Must be a positive integer")

            if action in ("move", "set_bounds"):
                x = arguments.get("x")
                y = arguments.get("y")
                if x is None or y is None:
                    errors.append(f"Coordinates (x, y) are required for action: {action}")
                elif not isinstance(x, int) or not isinstance(y, int):
                    errors.append("Coordinates must be integers")

            if action in ("resize", "set_bounds"):
                width = arguments.get("width")
                height = arguments.get("height")
                if width is None or height is None:
                    errors.append(f"Dimensions (width, height) are required for action: {action}")
                elif not isinstance(width, int) or not isinstance(height, int):
                    errors.append("Dimensions must be integers")
                elif width < self._min_width or height < self._min_height:
                    errors.append(f"Dimensions must be at least {self._min_width}x{self._min_height}")
                elif width > self._max_width or height > self._max_height:
                    errors.append(f"Dimensions must be at most {self._max_width}x{self._max_height}")

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        action = arguments.get("action")

        try:
            if action == "list":
                return self._execute_list(arguments, start)
            elif action == "get_active":
                return self._execute_get_active(start)
            elif action in self.MUTATION_ACTIONS:
                return self._execute_mutation(action, arguments, start)
            else:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start,
                )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Window operation failed: {e}",
                execution_time=time.time() - start,
            )

    def _validate_hwnd(self, hwnd: Any) -> Tuple[bool, Optional[str]]:
        if hwnd is None:
            return False, "Window handle is required"
        if not isinstance(hwnd, int):
            return False, f"Window handle must be an integer, got {type(hwnd).__name__}"
        if hwnd <= 0:
            return False, f"Invalid window handle: {hwnd}"
        if not user32.IsWindow(hwnd):
            return False, f"Window handle {hwnd} does not reference an existing window"
        return True, None

    def _get_window_info_after_mutation(self, hwnd: int) -> WindowInfo:
        return _get_window_info(hwnd)

    def _execute_mutation(self, action: str, arguments: Dict[str, Any], start: float) -> ToolResult:
        if not arguments.get("confirmed"):
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Operation requires user confirmation. Set 'confirmed' to true to proceed with '{action}'.",
                execution_time=time.time() - start,
            )

        hwnd = arguments.get("hwnd")
        valid, error = self._validate_hwnd(hwnd)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=error,
                execution_time=time.time() - start,
            )

        if _is_protected_window(hwnd):
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Cannot modify protected system window {hwnd}",
                execution_time=time.time() - start,
            )

        if action == "activate":
            return self._execute_activate(hwnd, start)
        elif action == "minimize":
            return self._execute_minimize(hwnd, start)
        elif action == "restore":
            return self._execute_restore(hwnd, start)
        elif action == "maximize":
            return self._execute_maximize(hwnd, start)
        elif action == "close":
            return self._execute_close(hwnd, start)
        elif action == "move":
            return self._execute_move(hwnd, arguments, start)
        elif action == "resize":
            return self._execute_resize(hwnd, arguments, start)
        elif action == "set_bounds":
            return self._execute_set_bounds(hwnd, arguments, start)
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Unknown mutation action: {action}",
                execution_time=time.time() - start,
            )

    def _execute_activate(self, hwnd: int, start: float) -> ToolResult:
        try:
            result = user32.SetForegroundWindow(hwnd)
            if result == 0:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Failed to activate window {hwnd}: SetForegroundWindow returned 0",
                    execution_time=time.time() - start,
                )

            info = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Activated window: {info.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "activate",
                    "hwnd": hwnd,
                    "window": info.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to activate window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_minimize(self, hwnd: int, start: float) -> ToolResult:
        try:
            user32.ShowWindow(hwnd, SW_MINIMIZE)
            info = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Minimized window: {info.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "minimize",
                    "hwnd": hwnd,
                    "window": info.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to minimize window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_restore(self, hwnd: int, start: float) -> ToolResult:
        try:
            user32.ShowWindow(hwnd, SW_RESTORE)
            info = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Restored window: {info.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "restore",
                    "hwnd": hwnd,
                    "window": info.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to restore window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_maximize(self, hwnd: int, start: float) -> ToolResult:
        try:
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
            info = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Maximized window: {info.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "maximize",
                    "hwnd": hwnd,
                    "window": info.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to maximize window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_close(self, hwnd: int, start: float) -> ToolResult:
        try:
            info_before = self._get_window_info_after_mutation(hwnd)
            result = user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Close request sent to window: {info_before.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "close",
                    "hwnd": hwnd,
                    "title": info_before.title,
                    "close_requested": bool(result),
                    "previous_state": info_before.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to send close request to window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_move(self, hwnd: int, arguments: Dict[str, Any], start: float) -> ToolResult:
        try:
            x = arguments["x"]
            y = arguments["y"]
            info_before = self._get_window_info_after_mutation(hwnd)
            current_rect = info_before.rect

            move_result = user32.MoveWindow(
                hwnd, x, y, current_rect["width"], current_rect["height"], True
            )
            if not move_result:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"MoveWindow failed for window {hwnd}",
                    execution_time=time.time() - start,
                )

            info_after = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Moved window: {info_after.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "move",
                    "hwnd": hwnd,
                    "window": info_after.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to move window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_resize(self, hwnd: int, arguments: Dict[str, Any], start: float) -> ToolResult:
        try:
            width = arguments["width"]
            height = arguments["height"]
            info_before = self._get_window_info_after_mutation(hwnd)
            current_rect = info_before.rect

            move_result = user32.MoveWindow(
                hwnd, current_rect["x"], current_rect["y"], width, height, True
            )
            if not move_result:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"MoveWindow failed for window {hwnd}",
                    execution_time=time.time() - start,
                )

            info_after = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Resized window: {info_after.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "resize",
                    "hwnd": hwnd,
                    "window": info_after.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to resize window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_set_bounds(self, hwnd: int, arguments: Dict[str, Any], start: float) -> ToolResult:
        try:
            x = arguments["x"]
            y = arguments["y"]
            width = arguments["width"]
            height = arguments["height"]

            move_result = user32.MoveWindow(hwnd, x, y, width, height, True)
            if not move_result:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"MoveWindow failed for window {hwnd}",
                    execution_time=time.time() - start,
                )

            info_after = self._get_window_info_after_mutation(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Set bounds for window: {info_after.title}",
                execution_time=time.time() - start,
                metadata={
                    "action": "set_bounds",
                    "hwnd": hwnd,
                    "window": info_after.to_dict(),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to set bounds for window {hwnd}: {e}",
                execution_time=time.time() - start,
            )

    def _execute_list(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        filter_title = arguments.get("filter_title", "")
        windows = enumerate_windows(max_windows=MAX_WINDOWS_RETURNED)

        if filter_title:
            filter_lower = filter_title.lower()
            windows = [w for w in windows if filter_lower in w.title.lower()]

        window_dicts = [w.to_dict() for w in windows]
        execution_time = time.time() - start

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Found {len(window_dicts)} windows",
            execution_time=execution_time,
            metadata={
                "windows": window_dicts,
                "count": len(window_dicts),
                "filter": filter_title if filter_title else None,
            },
        )

    def _execute_get_active(self, start: float) -> ToolResult:
        try:
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    output="No active window",
                    execution_time=time.time() - start,
                    metadata={"active_window": None},
                )

            info = _get_window_info(hwnd)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Active window: {info.title}",
                execution_time=time.time() - start,
                metadata={"active_window": info.to_dict()},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Failed to get active window: {e}",
                execution_time=time.time() - start,
            )
