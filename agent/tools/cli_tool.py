"""Restricted CLI tool with allowlist - disabled by default."""

import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import Tool, ToolResult, Permission, ConfirmationLevel


class CLITool(Tool):
    """Restricted command execution with allowlist. Disabled by default."""

    def __init__(self, workspace_dir: str = "workspace"):
        self._workspace = Path(workspace_dir).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._timeout = 15.0
        self._max_output_size = 50000
        self._enabled = False
        self._allowed_commands = {
            "dir": {"windows": "dir", "linux": "ls"},
            "ls": {"windows": "dir", "linux": "ls"},
            "cat": {"windows": "type", "linux": "cat"},
            "type": {"windows": "type", "linux": "cat"},
            "echo": {"windows": "echo", "linux": "echo"},
            "pwd": {"windows": "cd", "linux": "pwd"},
            "cd": {"windows": "cd", "linux": "pwd"},
            "find": {"windows": "where", "linux": "find"},
            "where": {"windows": "where", "linux": "which"},
            "tree": {"windows": "tree", "linux": "tree"},
            "head": {"windows": "more", "linux": "head"},
            "wc": {"windows": "find /c", "linux": "wc"},
        }
        self._blocked_patterns = [
            "&", "|", ";", ">", "<", "`", "$(", "${",
            "powershell", "cmd", "bash", "sh",
            "rm ", "del ", "rmdir", "format ",
            "reg ", "net ", "taskkill", "shutdown",
            "curl", "wget", "invoke-", "iwr",
        ]

    @property
    def name(self) -> str:
        return "cli"

    @property
    def description(self) -> str:
        return "Execute restricted CLI commands. Only allowlisted commands are permitted. DISABLED by default."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command arguments",
                },
            },
            "required": ["command"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "return_code": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.COMMAND_EXECUTE]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.DENY

    @property
    def timeout(self) -> float:
        return self._timeout

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def is_enabled(self) -> bool:
        return self._enabled

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        if not self._enabled:
            errors.append("CLI tool is disabled by default. Enable it explicitly.")
            return False, errors

        command = arguments.get("command")
        if not command:
            errors.append("Missing required argument: command")
            return False, errors

        if not self._is_allowed_command(command):
            errors.append(f"Command not in allowlist: {command}")
            return False, errors

        full_cmd = command + " " + " ".join(arguments.get("args", []))
        if self._contains_dangerous_patterns(full_cmd):
            errors.append("Command contains dangerous patterns")
            return False, errors

        args = arguments.get("args", [])
        for arg in args:
            if ".." in arg:
                errors.append(f"Path traversal not allowed: {arg}")
                return False, errors

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()

        if not self._enabled:
            return ToolResult(
                success=False, tool_name=self.name,
                error="CLI tool is disabled by default",
                execution_time=time.time() - start,
            )

        command = arguments["command"]
        args = arguments.get("args", [])

        mapped_cmd = self._map_command(command)
        full_args = [mapped_cmd] + args

        try:
            result = subprocess.run(
                full_args,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=str(self._workspace),
                shell=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            stdout = result.stdout
            stderr = result.stderr
            truncated = False

            if len(stdout) > self._max_output_size:
                stdout = stdout[: self._max_output_size] + "\n...[truncated]"
                truncated = True
            if len(stderr) > self._max_output_size:
                stderr = stderr[: self._max_output_size] + "\n...[truncated]"
                truncated = True

            return ToolResult(
                success=result.returncode == 0,
                tool_name=self.name,
                output=stdout,
                error=stderr,
                execution_time=time.time() - start,
                truncated=truncated,
                metadata={"return_code": result.returncode},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"Command timed out after {self._timeout}s",
                execution_time=time.time() - start,
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"Command not found: {mapped_cmd}",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=str(e),
                execution_time=time.time() - start,
            )

    def _is_allowed_command(self, command: str) -> bool:
        return command.lower() in self._allowed_commands

    def _map_command(self, command: str) -> str:
        mapping = self._allowed_commands.get(command.lower(), {})
        if os.name == "nt":
            return mapping.get("windows", command)
        return mapping.get("linux", command)

    def _contains_dangerous_patterns(self, cmd: str) -> bool:
        cmd_lower = cmd.lower()
        for pattern in self._blocked_patterns:
            if pattern.lower() in cmd_lower:
                return True
        if "\n" in cmd or "\r" in cmd:
            return True
        return False

    def get_allowed_commands(self) -> List[str]:
        return list(self._allowed_commands.keys())
