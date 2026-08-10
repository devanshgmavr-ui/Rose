"""Application launcher for launching apps with confirmation and safety.

Stage 5.1 - Controlled Application Launching.
"""

import subprocess
import os
import time
import shutil
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .base import Tool, ToolResult, Permission, ConfirmationLevel
from .permissions import PermissionManager
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class LaunchMode(Enum):
    DETACHED = "detached"
    WAIT = "wait"
    INTERACTIVE = "interactive"


@dataclass
class LaunchResult:
    success: bool
    app_name: str
    pid: Optional[int] = None
    error: str = ""
    launch_time: float = 0.0
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "app_name": self.app_name,
            "pid": self.pid,
            "error": self.error,
            "launch_time": self.launch_time,
            "exit_code": self.exit_code,
        }

    def to_text(self) -> str:
        if self.success:
            s = f"Launched {self.app_name}"
            if self.pid:
                s += f" (PID: {self.pid})"
            return s
        return f"Failed to launch {self.app_name}: {self.error}"


class AppLauncher:
    SAFE_EXTENSIONS = {".exe", ".bat", ".cmd", ".com", ".msc", ".ps1", ".py", ".pyw"}

    def __init__(
        self,
        permission_manager: Optional[PermissionManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        workspace_dir: Optional[str] = None,
        allowed_apps: Optional[List[str]] = None,
        blocked_apps: Optional[List[str]] = None,
        default_timeout: float = 30.0,
    ):
        self._perm = permission_manager or PermissionManager()
        self._audit = audit_logger or AuditLogger()
        self._workspace = workspace_dir or os.getcwd()
        self._allowed = allowed_apps
        self._blocked = blocked_apps or []
        self._default_timeout = default_timeout
        self._running: Dict[int, subprocess.Popen] = {}

    def launch(
        self,
        app_path: str,
        args: Optional[List[str]] = None,
        working_dir: Optional[str] = None,
        timeout: Optional[float] = None,
        mode: LaunchMode = LaunchMode.DETACHED,
        env: Optional[Dict[str, str]] = None,
        confirm: bool = True,
    ) -> LaunchResult:
        start = time.time()

        if not self._validate_path(app_path):
            return LaunchResult(
                success=False, app_name=app_path,
                error="Invalid or blocked application path",
            )

        if not os.path.exists(app_path):
            return LaunchResult(
                success=False, app_name=app_path, error="Application not found",
            )

        try:
            effective_timeout = timeout or self._default_timeout
            cmd = [app_path] + (args or [])
            kwargs: Dict[str, Any] = {"cwd": working_dir or self._workspace}

            if env:
                merged_env = os.environ.copy()
                merged_env.update(env)
                kwargs["env"] = merged_env

            if mode == LaunchMode.WAIT:
                proc = subprocess.run(
                    cmd, timeout=effective_timeout,
                    capture_output=True, text=True, **kwargs,
                )
                elapsed = time.time() - start
                return LaunchResult(
                    success=proc.returncode == 0, app_name=app_path,
                    error=proc.stderr if proc.returncode != 0 else "",
                    launch_time=elapsed, exit_code=proc.returncode,
                )

            if mode == LaunchMode.INTERACTIVE:
                proc = subprocess.Popen(cmd, **kwargs)
                self._running[proc.pid] = proc
                return LaunchResult(
                    success=True, app_name=app_path,
                    pid=proc.pid, launch_time=time.time() - start,
                )

            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd, creationflags=creation_flags,
                start_new_session=True, **kwargs,
            )
            self._running[proc.pid] = proc
            return LaunchResult(
                success=True, app_name=app_path,
                pid=proc.pid, launch_time=time.time() - start,
            )

        except subprocess.TimeoutExpired:
            return LaunchResult(
                success=False, app_name=app_path,
                error=f"Timeout after {effective_timeout}s",
                launch_time=time.time() - start,
            )
        except Exception as e:
            return LaunchResult(
                success=False, app_name=app_path,
                error=str(e), launch_time=time.time() - start,
            )

    def launch_by_name(
        self, app_name: str, args: Optional[List[str]] = None, **kwargs
    ) -> LaunchResult:
        resolved = shutil.which(app_name)
        if not resolved:
            return LaunchResult(
                success=False, app_name=app_name,
                error=f"Application '{app_name}' not found in PATH",
            )
        return self.launch(resolved, args=args, **kwargs)

    def is_running(self, pid: int) -> bool:
        proc = self._running.get(pid)
        if not proc:
            return False
        return proc.poll() is None

    def get_exit_code(self, pid: int) -> Optional[int]:
        proc = self._running.get(pid)
        if not proc:
            return None
        return proc.poll()

    def terminate(self, pid: int, timeout: float = 5.0) -> bool:
        proc = self._running.get(pid)
        if not proc:
            return False
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
            self._running.pop(pid, None)
            return True
        except Exception:
            return False

    def list_running(self) -> List[Dict[str, Any]]:
        result = []
        for pid, proc in list(self._running.items()):
            if proc.poll() is None:
                result.append({"pid": pid, "running": True})
            else:
                result.append({"pid": pid, "running": False, "exit_code": proc.returncode})
        return result

    def _validate_path(self, app_path: str) -> bool:
        if not app_path or not isinstance(app_path, str):
            return False
        ext = Path(app_path).suffix.lower()
        if ext and ext not in self.SAFE_EXTENSIONS:
            return False
        normalized = os.path.normpath(app_path)
        if any(blocked in normalized.lower() for blocked in self._blocked):
            return False
        if self._allowed:
            app_lower = os.path.basename(app_path).lower()
            if not any(app_lower == a.lower() for a in self._allowed):
                return False
        return True


class LaunchAppTool(Tool):
    @property
    def name(self) -> str:
        return "launch_app"

    @property
    def description(self) -> str:
        return "Launch an application with safety confirmation"

    @property
    def category(self) -> str:
        return "os_automation"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "app_path": {"type": "string", "description": "Path to the application"},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Arguments"},
                "working_dir": {"type": "string", "description": "Working directory"},
                "timeout": {"type": "number", "description": "Timeout in seconds"},
                "mode": {"type": "string", "enum": ["detached", "wait", "interactive"]},
            },
            "required": ["app_path"],
        }

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self.parameters

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "app_name": {"type": "string"},
                "pid": {"type": ["integer", "null"]},
                "error": {"type": ["string", "null"]},
                "launch_time": {"type": "number"},
                "exit_code": {"type": ["integer", "null"]},
            },
        }

    @property
    def required_permissions(self) -> List[Permission]:
        return [Permission.APP_LAUNCH]

    def validate(self, **kwargs) -> bool:
        return bool(kwargs.get("app_path"))

    def execute(self, **kwargs) -> ToolResult:
        app_path = kwargs.get("app_path", "")
        if not app_path:
            return ToolResult(success=False, tool_name=self.name, error="app_path is required")

        launcher = AppLauncher()
        result = launcher.launch(
            app_path,
            args=kwargs.get("args"),
            working_dir=kwargs.get("working_dir"),
            timeout=kwargs.get("timeout"),
            mode=LaunchMode(kwargs.get("mode", "detached")),
        )

        return ToolResult(
            success=result.success,
            tool_name=self.name,
            output=result.to_dict(),
            error=result.error if not result.success else None,
        )
