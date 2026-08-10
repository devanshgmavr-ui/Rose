"""Mouse control tool using Windows ctypes API."""

import time
import ctypes
import logging
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, ConfirmationLevel

logger = logging.getLogger(__name__)

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

WHEEL_DELTA = 120

MAX_MOUSE_ACTIONS_PER_REQUEST = 20
MOUSE_ACTION_TIMEOUT = 5.0
MAX_SCROLL_AMOUNT = 10
SCREEN_PADDING = 1


class MouseTool(Tool):
    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "mouse"

    @property
    def description(self) -> str:
        return "Control the mouse: move, click, double-click, right-click, scroll"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["position", "move", "click", "double_click", "right_click", "scroll"],
                    "description": "Mouse action to perform",
                },
                "x": {"type": "integer", "description": "X coordinate for move/click"},
                "y": {"type": "integer", "description": "Y coordinate for move/click"},
                "scroll_amount": {
                    "type": "integer",
                    "description": "Scroll amount (positive=up, negative=down)",
                    "default": 1,
                },
            },
            "required": ["action"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "position": {"type": "object"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["os.mouse"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return MOUSE_ACTION_TIMEOUT

    def _get_screen_dimensions(self) -> Tuple[int, int]:
        try:
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            return width, height
        except Exception:
            return 1920, 1080

    def _get_cursor_pos(self) -> Tuple[int, int]:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        user32 = ctypes.windll.user32
        pt = POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def _set_cursor_pos(self, x: int, y: int) -> bool:
        user32 = ctypes.windll.user32
        result = user32.SetCursorPos(x, y)
        return result != 0

    def _mouse_event(self, flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> None:
        ctypes.windll.user32.mouse_event(flags, dx, dy, data, 0)

    def _get_active_window_title(self) -> str:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            return buf.value or "(no title)"
        except Exception:
            return "(unknown)"

    def _validate_coordinates(self, x: int, y: int, screen_w: int, screen_h: int) -> List[str]:
        errors = []
        if not isinstance(x, int) or not isinstance(y, int):
            errors.append("Coordinates must be integers")
            return errors
        if x < SCREEN_PADDING or x >= screen_w - SCREEN_PADDING:
            errors.append(f"X coordinate {x} is outside screen bounds (0-{screen_w - 1})")
        if y < SCREEN_PADDING or y >= screen_h - SCREEN_PADDING:
            errors.append(f"Y coordinate {y} is outside screen bounds (0-{screen_h - 1})")
        return errors

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not self._enabled:
            return False, ["Mouse control is not enabled"]

        errors = []
        action = arguments.get("action")
        valid_actions = ["position", "move", "click", "double_click", "right_click", "scroll"]
        if action not in valid_actions:
            errors.append(f"Invalid action: {action}. Must be one of {valid_actions}")
            return False, errors

        screen_w, screen_h = self._get_screen_dimensions()

        if action in ("move", "click", "double_click", "right_click"):
            x = arguments.get("x")
            y = arguments.get("y")
            if x is None or y is None:
                errors.append(f"Action '{action}' requires x and y coordinates")
            else:
                errors.extend(self._validate_coordinates(x, y, screen_w, screen_h))

        if action == "scroll":
            scroll_amount = arguments.get("scroll_amount", 1)
            if not isinstance(scroll_amount, int):
                errors.append("scroll_amount must be an integer")
            elif abs(scroll_amount) > MAX_SCROLL_AMOUNT:
                errors.append(f"scroll_amount {scroll_amount} exceeds maximum of {MAX_SCROLL_AMOUNT}")

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
            if action == "position":
                return self._execute_position(start)
            elif action == "move":
                return self._execute_move(arguments, start)
            elif action == "click":
                return self._execute_click(arguments, start, "left")
            elif action == "double_click":
                return self._execute_click(arguments, start, "double")
            elif action == "right_click":
                return self._execute_click(arguments, start, "right")
            elif action == "scroll":
                return self._execute_scroll(arguments, start)
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
                error=f"Mouse operation failed: {e}",
                execution_time=time.time() - start,
            )

    def _execute_position(self, start: float) -> ToolResult:
        x, y = self._get_cursor_pos()
        screen_w, screen_h = self._get_screen_dimensions()
        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Cursor at ({x}, {y})",
            execution_time=time.time() - start,
            metadata={
                "position": {"x": x, "y": y},
                "screen": {"width": screen_w, "height": screen_h},
            },
        )

    def _execute_move(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        x, y = arguments["x"], arguments["y"]
        success = self._set_cursor_pos(x, y)
        if not success:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="Failed to move cursor",
                execution_time=time.time() - start,
            )
        new_x, new_y = self._get_cursor_pos()
        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Cursor moved to ({new_x}, {new_y})",
            execution_time=time.time() - start,
            metadata={"position": {"x": new_x, "y": new_y}},
        )

    def _execute_click(self, arguments: Dict[str, Any], start: float, click_type: str) -> ToolResult:
        x, y = arguments["x"], arguments["y"]
        self._set_cursor_pos(x, y)
        time.sleep(0.01)

        if click_type in ("left", "double"):
            self._mouse_event(MOUSEEVENTF_LEFTDOWN)
            self._mouse_event(MOUSEEVENTF_LEFTUP)
            if click_type == "double":
                time.sleep(0.03)
                self._mouse_event(MOUSEEVENTF_LEFTDOWN)
                self._mouse_event(MOUSEEVENTF_LEFTUP)
        elif click_type == "right":
            self._mouse_event(MOUSEEVENTF_RIGHTDOWN)
            self._mouse_event(MOUSEEVENTF_RIGHTUP)

        time.sleep(0.01)
        active_window = self._get_active_window_title()

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"{click_type} click at ({x}, {y})",
            execution_time=time.time() - start,
            metadata={
                "position": {"x": x, "y": y},
                "click_type": click_type,
                "active_window": active_window,
            },
        )

    def _execute_scroll(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        scroll_amount = arguments.get("scroll_amount", 1)
        wheel_delta = scroll_amount * WHEEL_DELTA
        self._mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_delta)
        time.sleep(0.01)

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Scrolled by {scroll_amount} (delta={wheel_delta})",
            execution_time=time.time() - start,
            metadata={"scroll_amount": scroll_amount, "wheel_delta": wheel_delta},
        )
