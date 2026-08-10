"""System information tool using ctypes."""

import os
import sys
import time
import platform
import logging
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel

logger = logging.getLogger(__name__)

MAX_OUTPUT_LENGTH = 5000


class SystemInfoTool(Tool):
    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return "Get safe system information (OS, screen, CPU, memory, active window)"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["all", "os", "screen", "cursor", "active_window", "cpu", "memory"],
                    "description": "Type of information to retrieve",
                    "default": "all",
                },
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "info": {"type": "object"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["os.system_info"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return 10.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        info_type = arguments.get("info_type", "all")
        valid_types = ["all", "os", "screen", "cursor", "active_window", "cpu", "memory"]
        if info_type not in valid_types:
            errors.append(f"Invalid info_type: {info_type}. Must be one of {valid_types}")
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

        info_type = arguments.get("info_type", "all")

        try:
            info = {}

            if info_type in ("all", "os"):
                info["os"] = self._get_os_info()

            if info_type in ("all", "screen"):
                info["screen"] = self._get_screen_info()

            if info_type in ("all", "cursor"):
                info["cursor"] = self._get_cursor_info()

            if info_type in ("all", "active_window"):
                info["active_window"] = self._get_active_window_info()

            if info_type in ("all", "cpu"):
                info["cpu"] = self._get_cpu_info()

            if info_type in ("all", "memory"):
                info["memory"] = self._get_memory_info()

            output_text = str(info)
            if len(output_text) > MAX_OUTPUT_LENGTH:
                output_text = output_text[:MAX_OUTPUT_LENGTH] + "...[truncated]"

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=output_text,
                execution_time=time.time() - start,
                metadata=info,
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"System info retrieval failed: {e}",
                execution_time=time.time() - start,
            )

    def _get_os_info(self) -> Dict[str, Any]:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

    def _get_screen_info(self) -> Dict[str, Any]:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            monitors = user32.GetSystemMetrics(80)
            return {
                "width": width,
                "height": height,
                "monitor_count": monitors,
            }
        except Exception:
            return {"error": "Could not retrieve screen info"}

    def _get_cursor_info(self) -> Dict[str, Any]:
        try:
            import ctypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            user32 = ctypes.windll.user32
            pt = POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            return {"x": pt.x, "y": pt.y}
        except Exception:
            return {"error": "Could not retrieve cursor info"}

    def _get_active_window_info(self) -> Dict[str, Any]:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            title = buf.value

            return {
                "handle": hwnd,
                "title": title if title else "(no title)",
            }
        except Exception:
            return {"error": "Could not retrieve active window info"}

    def _get_cpu_info(self) -> Dict[str, Any]:
        info = {
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_arch": platform.architecture()[0],
        }
        return info

    def _get_memory_info(self) -> Dict[str, Any]:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = MEMORYSTATUSEX()
            mem_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))

            return {
                "memory_load_percent": mem_status.dwMemoryLoad,
                "total_physical_bytes": mem_status.ullTotalPhys,
                "available_physical_bytes": mem_status.ullAvailPhys,
                "total_physical_gb": round(mem_status.ullTotalPhys / (1024**3), 2),
                "available_physical_gb": round(mem_status.ullAvailPhys / (1024**3), 2),
            }
        except Exception:
            return {"error": "Could not retrieve memory info"}
