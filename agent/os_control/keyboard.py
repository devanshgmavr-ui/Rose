"""Keyboard control tool using Windows ctypes API."""

import time
import ctypes
import logging
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, ConfirmationLevel

logger = logging.getLogger(__name__)

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
INPUT_KEYBOARD = 1
WHEEL_DELTA = 120

MAX_KEYBOARD_ACTIONS_PER_REQUEST = 20
MAX_TYPED_TEXT_LENGTH = 1000
MAX_HOTKEY_KEYS = 4
KEYBOARD_ACTION_TIMEOUT = 5.0

VIRTUAL_KEY_MAP = {
    "ENTER": 0x0D,
    "RETURN": 0x0D,
    "TAB": 0x09,
    "SPACE": 0x20,
    "BACKSPACE": 0x08,
    "DELETE": 0x2E,
    "DEL": 0x2E,
    "INSERT": 0x2D,
    "INS": 0x2D,
    "HOME": 0x24,
    "END": 0x23,
    "PAGEUP": 0x21,
    "PAGE_DOWN": 0x22,
    "PGUP": 0x21,
    "PGDN": 0x22,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "ESCAPE": 0x1B,
    "ESC": 0x1B,
    "CAPSLOCK": 0x14,
    "NUMLOCK": 0x90,
    "SCROLLLOCK": 0x91,
    "PRINTSCREEN": 0x2C,
    "PRTSC": 0x2C,
    "F1": 0x70,
    "F2": 0x71,
    "F3": 0x72,
    "F4": 0x73,
    "F5": 0x74,
    "F6": 0x75,
    "F7": 0x76,
    "F8": 0x77,
    "F9": 0x78,
    "F10": 0x79,
    "F11": 0x7A,
    "F12": 0x7B,
    "CTRL": 0x11,
    "CONTROL": 0x11,
    "ALT": 0x12,
    "MENU": 0x12,
    "SHIFT": 0x10,
    "WIN": 0x5B,
    "LWIN": 0x5B,
    "RWIN": 0x5C,
    "COMMAND": 0x5B,
}

RESTRICTED_COMBINATIONS = [
    frozenset(["CTRL", "ALT", "DELETE"]),
    frozenset(["CTRL", "ALT", "DEL"]),
    frozenset(["CTRL", "SHIFT", "ESC"]),
    frozenset(["CTRL", "SHIFT", "ESCAPE"]),
    frozenset(["ALT", "F4"]),
]

RESTRICTED_SINGLE_KEYS = []

ALPHANUMERIC_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " !@#$%^&*()_+-=[]{}|;':\",./<>?~`"
    "\t\n\r"
)


class KeyboardTool(Tool):
    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "keyboard"

    @property
    def description(self) -> str:
        return "Control the keyboard: type text, press keys, use hotkey combinations"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["type", "press", "hotkey"],
                    "description": "Keyboard action to perform",
                },
                "text": {"type": "string", "description": "Text to type (for 'type' action)"},
                "key": {"type": "string", "description": "Key to press (for 'press' action)"},
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys for hotkey combination (for 'hotkey' action)",
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
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["os.keyboard"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return KEYBOARD_ACTION_TIMEOUT

    def _send_unicode_char(self, char: str) -> None:
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("ki", KEYBDINPUT),
                ("padding", ctypes.c_ubyte * 8),
            ]

        inputs = (INPUT * 2)()
        inputs[0].type = INPUT_KEYBOARD
        inputs[0].ki.wVk = 0
        inputs[0].ki.wScan = ord(char)
        inputs[0].ki.dwFlags = KEYEVENTF_UNICODE

        inputs[1].type = INPUT_KEYBOARD
        inputs[1].ki.wVk = 0
        inputs[1].ki.wScan = ord(char)
        inputs[1].ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))

    def _send_virtual_key(self, vk_code: int, up: bool = False) -> None:
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("ki", KEYBDINPUT),
                ("padding", ctypes.c_ubyte * 8),
            ]

        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp.ki.wVk = vk_code
        inp.ki.dwFlags = KEYEVENTF_KEYUP if up else 0

        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def _get_active_window_title(self) -> str:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            return buf.value or "(no title)"
        except Exception:
            return "(unknown)"

    def _is_restricted_combination(self, keys: List[str]) -> Tuple[bool, str]:
        normalized = set(k.upper() for k in keys)
        for restricted in RESTRICTED_COMBINATIONS:
            if restricted.issubset(normalized):
                return True, "+".join(sorted(restricted))
        return False, ""

    def _normalize_key(self, key: str) -> str:
        upper = key.upper()
        if upper in VIRTUAL_KEY_MAP:
            return upper
        if len(key) == 1:
            return key
        return upper

    def _get_vk_code(self, key: str) -> int:
        normalized = self._normalize_key(key)
        if normalized in VIRTUAL_KEY_MAP:
            return VIRTUAL_KEY_MAP[normalized]
        if len(normalized) == 1:
            return ord(normalized.upper())
        raise ValueError(f"Unknown key: {key}")

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if not self._enabled:
            return False, ["Keyboard control is not enabled"]

        errors = []
        action = arguments.get("action")
        valid_actions = ["type", "press", "hotkey"]
        if action not in valid_actions:
            errors.append(f"Invalid action: {action}. Must be one of {valid_actions}")
            return False, errors

        if action == "type":
            text = arguments.get("text", "")
            if not isinstance(text, str):
                errors.append("Text must be a string")
            elif len(text) == 0:
                errors.append("Text cannot be empty")
            elif len(text) > MAX_TYPED_TEXT_LENGTH:
                errors.append(f"Text length {len(text)} exceeds maximum of {MAX_TYPED_TEXT_LENGTH}")

        elif action == "press":
            key = arguments.get("key")
            if key is None:
                errors.append("Press action requires a 'key' argument")
            else:
                normalized = self._normalize_key(key)
                if normalized not in VIRTUAL_KEY_MAP and len(key) != 1:
                    errors.append(f"Unknown key: {key}. Use F1-F12, ENTER, TAB, etc.")
                if normalized in RESTRICTED_SINGLE_KEYS:
                    errors.append(f"Key '{key}' is restricted")

        elif action == "hotkey":
            keys = arguments.get("keys", [])
            if not isinstance(keys, list):
                errors.append("Keys must be an array")
            elif len(keys) == 0:
                errors.append("Keys array cannot be empty")
            elif len(keys) > MAX_HOTKEY_KEYS:
                errors.append(f"Hotkey has {len(keys)} keys, maximum is {MAX_HOTKEY_KEYS}")
            else:
                restricted, combo = self._is_restricted_combination(keys)
                if restricted:
                    errors.append(f"Restricted key combination: {combo}")
                for k in keys:
                    normalized = self._normalize_key(k)
                    if normalized not in VIRTUAL_KEY_MAP and len(k) != 1:
                        errors.append(f"Unknown key in hotkey: {k}")

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
            if action == "type":
                return self._execute_type(arguments, start)
            elif action == "press":
                return self._execute_press(arguments, start)
            elif action == "hotkey":
                return self._execute_hotkey(arguments, start)
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
                error=f"Keyboard operation failed: {e}",
                execution_time=time.time() - start,
            )

    def _execute_type(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        text = arguments.get("text", "")
        for char in text:
            self._send_unicode_char(char)
            time.sleep(0.005)

        active_window = self._get_active_window_title()
        display_text = text[:50] + "..." if len(text) > 50 else text

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Typed: {repr(display_text)}",
            execution_time=time.time() - start,
            metadata={
                "characters_typed": len(text),
                "active_window": active_window,
            },
        )

    def _execute_press(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        key = arguments["key"]
        vk_code = self._get_vk_code(key)

        self._send_virtual_key(vk_code, up=False)
        time.sleep(0.01)
        self._send_virtual_key(vk_code, up=True)
        time.sleep(0.01)

        active_window = self._get_active_window_title()

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Pressed: {key.upper()}",
            execution_time=time.time() - start,
            metadata={
                "key": key.upper(),
                "vk_code": vk_code,
                "active_window": active_window,
            },
        )

    def _execute_hotkey(self, arguments: Dict[str, Any], start: float) -> ToolResult:
        keys = arguments.get("keys", [])
        vk_codes = [self._get_vk_code(k) for k in keys]

        for vk in vk_codes:
            self._send_virtual_key(vk, up=False)
            time.sleep(0.005)

        time.sleep(0.01)

        for vk in reversed(vk_codes):
            self._send_virtual_key(vk, up=True)
            time.sleep(0.005)

        active_window = self._get_active_window_title()
        combo = "+".join(k.upper() for k in keys)

        return ToolResult(
            success=True,
            tool_name=self.name,
            output=f"Hotkey: {combo}",
            execution_time=time.time() - start,
            metadata={
                "keys": [k.upper() for k in keys],
                "active_window": active_window,
            },
        )
