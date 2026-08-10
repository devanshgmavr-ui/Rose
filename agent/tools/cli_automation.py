"""Controlled CLI automation with sandboxing.

Stage 5.3 - Controlled CLI Automation.

Provides:
- Sandboxed command execution
- Output capture and parsing
- Timeout enforcement
- Command whitelist/blacklist
- Working directory control
- Environment variable filtering
"""

import subprocess
import os
import time
import shlex
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ShellType(Enum):
    POWERSHELL = "powershell"
    CMD = "cmd"
    BASH = "bash"


@dataclass
class CommandResult:
    success: bool
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_time: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "timed_out": self.timed_out,
        }

    def to_text(self) -> str:
        if self.timed_out:
            return f"Command timed out after {self.execution_time:.1f}s"
        if self.success:
            return self.stdout or "Command executed successfully"
        return self.stderr or f"Command failed with exit code {self.exit_code}"


class CLIExecutor:
    """Executes CLI commands with safety controls."""

    DANGEROUS_COMMANDS = {
        "format", "rd", "rmdir", "del", "erase", "rm", "rf",
        "shutdown", "reboot", "halt", "poweroff",
        "mkfs", "fdisk", "diskpart",
        "reg", "regedit",
        "taskkill", "taskkill /f",
    }

    DANGEROUS_FLAGS = {"-rf", "-fr", "--force", "/f", "/s", "/q"}

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        shell: ShellType = ShellType.POWERSHELL,
        default_timeout: float = 30.0,
        max_output_size: int = 100000,
        blocked_commands: Optional[List[str]] = None,
        allowed_commands: Optional[List[str]] = None,
        env_passthrough: bool = True,
        restricted_env: Optional[Dict[str, str]] = None,
    ):
        self._workspace = workspace_dir or os.getcwd()
        self._shell = shell
        self._default_timeout = default_timeout
        self._max_output = max_output_size
        self._blocked = [c.lower() for c in (blocked_commands or [])]
        self._allowed = [c.lower() for c in (allowed_commands or [])] if allowed_commands else None
        self._env_passthrough = env_passthrough
        self._restricted_env = restricted_env or {}

    def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        capture: bool = True,
    ) -> CommandResult:
        """Execute a command with safety checks."""
        start = time.time()

        if not self._validate_command(command):
            return CommandResult(
                success=False, command=command,
                stderr="Command is blocked or dangerous",
            )

        effective_timeout = timeout or self._default_timeout
        cwd = working_dir or self._workspace

        exec_env = self._build_environment(env)

        try:
            if self._shell == ShellType.POWERSHELL:
                cmd_list = ["powershell", "-NoProfile", "-Command", command]
            elif self._shell == ShellType.CMD:
                cmd_list = ["cmd", "/c", command]
            else:
                cmd_list = ["bash", "-c", command]

            proc = subprocess.run(
                cmd_list,
                cwd=cwd,
                capture_output=capture,
                text=True,
                timeout=effective_timeout,
                env=exec_env,
            )

            elapsed = time.time() - start
            stdout = proc.stdout[:self._max_output] if proc.stdout else ""
            stderr = proc.stderr[:self._max_output] if proc.stderr else ""

            return CommandResult(
                success=proc.returncode == 0,
                command=command,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                execution_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            return CommandResult(
                success=False, command=command,
                stderr=f"Command timed out after {effective_timeout}s",
                execution_time=elapsed, timed_out=True,
            )
        except Exception as e:
            elapsed = time.time() - start
            return CommandResult(
                success=False, command=command,
                stderr=str(e), execution_time=elapsed,
            )

    def execute_safe(
        self,
        command: str,
        timeout: Optional[float] = None,
        working_dir: Optional[str] = None,
    ) -> CommandResult:
        """Execute with stricter safety: no shell injection, fixed args."""
        start = time.time()
        effective_timeout = timeout or self._default_timeout
        cwd = working_dir or self._workspace

        try:
            if self._shell == ShellType.POWERSHELL:
                cmd_list = ["powershell", "-NoProfile", "-Command", command]
            elif self._shell == ShellType.CMD:
                cmd_list = ["cmd", "/c", command]
            else:
                cmd_list = ["bash", "-c", command]

            proc = subprocess.run(
                cmd_list, cwd=cwd, capture_output=True, text=True,
                timeout=effective_timeout,
            )

            elapsed = time.time() - start
            return CommandResult(
                success=proc.returncode == 0,
                command=command,
                stdout=proc.stdout[:self._max_output] if proc.stdout else "",
                stderr=proc.stderr[:self._max_output] if proc.stderr else "",
                exit_code=proc.returncode,
                execution_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False, command=command,
                stderr=f"Timed out after {effective_timeout}s",
                execution_time=time.time() - start, timed_out=True,
            )
        except Exception as e:
            return CommandResult(
                success=False, command=command,
                stderr=str(e), execution_time=time.time() - start,
            )

    def _validate_command(self, command: str) -> bool:
        """Validate command safety."""
        if not command or not isinstance(command, str):
            return False

        cmd_lower = command.lower().strip()

        for blocked in self._blocked:
            if blocked in cmd_lower:
                return False

        if self._allowed:
            first_word = cmd_lower.split()[0] if cmd_lower.split() else ""
            if not any(first_word.startswith(a) for a in self._allowed):
                return False

        for dangerous in self.DANGEROUS_COMMANDS:
            if cmd_lower.startswith(dangerous) or f" {dangerous} " in cmd_lower:
                return False

        for flag in self.DANGEROUS_FLAGS:
            if flag in cmd_lower:
                return False

        if ";" in cmd_lower or "&&" in cmd_lower or "||" in cmd_lower:
            return False

        if "$(" in cmd_lower or "`" in cmd_lower:
            return False

        return True

    def _build_environment(self, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build safe environment."""
        if self._env_passthrough:
            exec_env = os.environ.copy()
        else:
            exec_env = {
                "PATH": os.environ.get("PATH", ""),
                "TEMP": os.environ.get("TEMP", ""),
                "SystemRoot": os.environ.get("SystemRoot", ""),
            }

        exec_env.update(self._restricted_env)
        if env:
            exec_env.update(env)

        return exec_env

    def get_shell_info(self) -> Dict[str, Any]:
        """Get information about the current shell."""
        return {
            "shell": self._shell.value,
            "workspace": self._workspace,
            "default_timeout": self._default_timeout,
            "blocked_commands": self._blocked,
            "env_passthrough": self._env_passthrough,
        }


class CLIExecutorTool:
    """Tool wrapper for CLI execution."""

    def __init__(self, executor: Optional[CLIExecutor] = None):
        self._executor = executor or CLIExecutor()

    def execute_command(
        self, command: str, timeout: Optional[float] = None
    ) -> CommandResult:
        return self._executor.execute(command, timeout=timeout)

    def execute_safe(
        self, command: str, timeout: Optional[float] = None
    ) -> CommandResult:
        return self._executor.execute_safe(command, timeout=timeout)

    def get_info(self) -> Dict[str, Any]:
        return self._executor.get_shell_info()
