"""Restricted filesystem tool - workspace boundary only."""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import Tool, ToolResult, Permission, ConfirmationLevel


class FilesystemTool(Tool):
    """Filesystem tool restricted to workspace directory."""

    def __init__(self, workspace_dir: str = "workspace"):
        self._workspace = Path(workspace_dir).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._max_read_size = 100000
        self._max_write_size = 500000
        self._max_list_items = 1000

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def description(self) -> str:
        return "Read, write, and list files within the workspace directory only."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "write"],
                },
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["action"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "files": {"type": "array"},
                "size": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.FILESYSTEM_READ, Permission.FILESYSTEM_WRITE]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 10.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        action = arguments.get("action")
        if not action:
            errors.append("Missing required argument: action")
        elif action not in ("list", "read", "write"):
            errors.append(f"Invalid action: {action}")

        if action in ("read", "write"):
            path = arguments.get("path")
            if not path:
                errors.append(f"Missing required argument: path for {action}")
            elif not self._is_safe_path(path):
                errors.append(f"Path outside workspace: {path}")

        if action == "write":
            content = arguments.get("content")
            if content is None:
                errors.append("Missing required argument: content for write")
            elif isinstance(content, str) and len(content) > self._max_write_size:
                errors.append(f"Content exceeds max write size ({self._max_write_size} bytes)")

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        start = time.time()
        action = arguments.get("action")

        try:
            if action == "list":
                return self._list_files(arguments, start)
            elif action == "read":
                return self._read_file(arguments, start)
            elif action == "write":
                return self._write_file(arguments, start)
            else:
                return ToolResult(
                    success=False, tool_name=self.name,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start,
                )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=str(e), execution_time=time.time() - start,
            )

    def _is_safe_path(self, path_str: str) -> bool:
        try:
            path = (self._workspace / path_str).resolve()
            if not str(path).startswith(str(self._workspace)):
                return False
            if ".." in Path(path_str).parts:
                return False
            return True
        except Exception:
            return False

    def _resolve_safe(self, path_str: str) -> Path:
        path = (self._workspace / path_str).resolve()
        if not str(path).startswith(str(self._workspace)):
            raise ValueError(f"Path escapes workspace: {path_str}")
        return path

    def _list_files(self, args: Dict, start: float) -> ToolResult:
        rel_path = args.get("path", ".")
        if not self._is_safe_path(rel_path):
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"Path outside workspace: {rel_path}",
                execution_time=time.time() - start,
            )
        target = self._resolve_safe(rel_path)
        if not target.exists():
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"Path not found: {rel_path}",
                execution_time=time.time() - start,
            )
        items = []
        try:
            for i, item in enumerate(sorted(target.iterdir())):
                if i >= self._max_list_items:
                    break
                rel = str(item.relative_to(self._workspace)).replace("\\", "/")
                items.append({
                    "name": item.name,
                    "path": rel,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                })
        except PermissionError:
            return ToolResult(
                success=False, tool_name=self.name,
                error="Permission denied",
                execution_time=time.time() - start,
            )

        output = f"Found {len(items)} items in '{rel_path}':\n"
        for item in items:
            prefix = "  [DIR] " if item["type"] == "directory" else "       "
            output += f"{prefix}{item['path']}\n"

        return ToolResult(
            success=True, tool_name=self.name,
            output=output, metadata={"items": items},
            execution_time=time.time() - start,
        )

    def _read_file(self, args: Dict, start: float) -> ToolResult:
        rel_path = args["path"]
        target = self._resolve_safe(rel_path)
        if not target.exists():
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"File not found: {rel_path}",
                execution_time=time.time() - start,
            )
        if target.is_dir():
            return ToolResult(
                success=False, tool_name=self.name,
                error=f"Is a directory: {rel_path}",
                execution_time=time.time() - start,
            )
        try:
            size = target.stat().st_size
            truncated = size > self._max_read_size
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(self._max_read_size)
            return ToolResult(
                success=True, tool_name=self.name,
                output=content, truncated=truncated,
                metadata={"path": rel_path, "size": size},
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=str(e), execution_time=time.time() - start,
            )

    def _write_file(self, args: Dict, start: float) -> ToolResult:
        rel_path = args["path"]
        content = args["content"]
        target = self._resolve_safe(rel_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                success=True, tool_name=self.name,
                output=f"Successfully wrote {len(content)} bytes to {rel_path}",
                metadata={"path": rel_path, "bytes_written": len(content)},
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False, tool_name=self.name,
                error=str(e), execution_time=time.time() - start,
            )
