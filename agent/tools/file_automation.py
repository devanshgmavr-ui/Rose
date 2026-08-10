"""Advanced file automation with content operations.

Stage 5.2 - Advanced File Automation.

Provides:
- File content reading/writing
- Search and replace
- File copying/moving
- Directory operations
- File watching
- Content indexing
"""

import os
import shutil
import time
import logging
import hashlib
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class FileOperation(Enum):
    READ = "read"
    WRITE = "write"
    APPEND = "append"
    COPY = "copy"
    MOVE = "move"
    DELETE = "delete"
    SEARCH = "search"
    REPLACE = "replace"
    LIST_DIR = "list_dir"
    MAKE_DIR = "make_dir"
    GET_INFO = "get_info"


@dataclass
class FileResult:
    success: bool
    operation: str
    path: str
    output: Any = None
    error: str = ""
    bytes_written: int = 0
    files_affected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "path": self.path,
            "output": self.output,
            "error": self.error,
            "bytes_written": self.bytes_written,
            "files_affected": self.files_affected,
        }

    def to_text(self) -> str:
        if self.success:
            if self.output:
                return str(self.output)
            return f"{self.operation} on {self.path} completed"
        return f"{self.operation} failed: {self.error}"


@dataclass
class FileInfo:
    name: str
    path: str
    size: int
    modified: float
    is_dir: bool
    extension: str = ""
    mime_type: str = ""
    checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "modified": self.modified,
            "is_dir": self.is_dir,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "checksum": self.checksum,
        }


class FileAutomator:
    """Advanced file operations with safety controls."""

    MAX_FILE_SIZE_MB = 100
    MAX_READ_SIZE_MB = 50
    MAX_WRITE_SIZE_MB = 50
    MAX_DIR_DEPTH = 10

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        max_file_size_mb: float = 100,
        allowed_extensions: Optional[List[str]] = None,
        blocked_extensions: Optional[List[str]] = None,
    ):
        self._workspace = workspace_dir or os.getcwd()
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._allowed_ext = allowed_extensions
        self._blocked_ext = blocked_extensions or []

    def read_file(
        self, path: str, encoding: str = "utf-8", max_size: Optional[int] = None
    ) -> FileResult:
        """Read file contents."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "read", path, error="File not found")
            if os.path.isdir(full_path):
                return FileResult(False, "read", path, error="Path is a directory")

            size = os.path.getsize(full_path)
            limit = max_size or self._max_file_size
            if size > limit:
                return FileResult(
                    False, "read", path,
                    error=f"File too large: {size} bytes (limit: {limit})",
                )

            with open(full_path, "r", encoding=encoding) as f:
                content = f.read()

            return FileResult(True, "read", path, output=content)

        except Exception as e:
            return FileResult(False, "read", path, error=str(e))

    def write_file(
        self, path: str, content: str, encoding: str = "utf-8", append: bool = False
    ) -> FileResult:
        """Write content to file."""
        try:
            full_path = self._resolve_path(path)
            dir_path = os.path.dirname(full_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            size = len(content.encode(encoding))
            if size > self._max_file_size:
                return FileResult(
                    False, "write" if not append else "append", path,
                    error=f"Content too large: {size} bytes",
                )

            mode = "a" if append else "w"
            with open(full_path, mode, encoding=encoding) as f:
                f.write(content)

            return FileResult(
                True, "write" if not append else "append", path,
                bytes_written=size,
            )

        except Exception as e:
            return FileResult(False, "write" if not append else "append", path, error=str(e))

    def copy_file(self, src: str, dst: str, overwrite: bool = False) -> FileResult:
        """Copy a file."""
        try:
            src_path = self._resolve_path(src)
            dst_path = self._resolve_path(dst)

            if not os.path.exists(src_path):
                return FileResult(False, "copy", src, error="Source not found")

            if os.path.exists(dst_path) and not overwrite:
                return FileResult(False, "copy", src, error="Destination exists")

            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

            shutil.copy2(src_path, dst_path)
            return FileResult(True, "copy", src, output=dst)

        except Exception as e:
            return FileResult(False, "copy", src, error=str(e))

    def move_file(self, src: str, dst: str, overwrite: bool = False) -> FileResult:
        """Move a file."""
        try:
            src_path = self._resolve_path(src)
            dst_path = self._resolve_path(dst)

            if not os.path.exists(src_path):
                return FileResult(False, "move", src, error="Source not found")

            if os.path.exists(dst_path) and not overwrite:
                return FileResult(False, "move", src, error="Destination exists")

            dst_dir = os.path.dirname(dst_path)
            if dst_dir and not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)

            shutil.move(src_path, dst_path)
            return FileResult(True, "move", src, output=dst)

        except Exception as e:
            return FileResult(False, "move", src, error=str(e))

    def delete_file(self, path: str) -> FileResult:
        """Delete a file."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "delete", path, error="File not found")

            if os.path.isdir(full_path):
                shutil.rmtree(full_path)
            else:
                os.remove(full_path)

            return FileResult(True, "delete", path)

        except Exception as e:
            return FileResult(False, "delete", path, error=str(e))

    def list_directory(
        self, path: str, pattern: str = "*", recursive: bool = False
    ) -> FileResult:
        """List directory contents."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "list_dir", path, error="Directory not found")
            if not os.path.isdir(full_path):
                return FileResult(False, "list_dir", path, error="Path is not a directory")

            p = Path(full_path)
            if recursive:
                items = list(p.rglob(pattern))
            else:
                items = list(p.glob(pattern))

            result_list = []
            for item in items:
                info = FileInfo(
                    name=item.name,
                    path=str(item),
                    size=item.stat().st_size if item.is_file() else 0,
                    modified=item.stat().st_mtime,
                    is_dir=item.is_dir(),
                    extension=item.suffix.lower() if item.is_file() else "",
                )
                result_list.append(info.to_dict())

            return FileResult(
                True, "list_dir", path,
                output=result_list, files_affected=len(result_list),
            )

        except Exception as e:
            return FileResult(False, "list_dir", path, error=str(e))

    def make_directory(self, path: str) -> FileResult:
        """Create a directory."""
        try:
            full_path = self._resolve_path(path)
            os.makedirs(full_path, exist_ok=True)
            return FileResult(True, "make_dir", path)

        except Exception as e:
            return FileResult(False, "make_dir", path, error=str(e))

    def get_file_info(self, path: str) -> FileResult:
        """Get file information."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "get_info", path, error="File not found")

            stat = os.stat(full_path)
            p = Path(full_path)
            info = FileInfo(
                name=p.name,
                path=str(p),
                size=stat.st_size,
                modified=stat.st_mtime,
                is_dir=os.path.isdir(full_path),
                extension=p.suffix.lower() if os.path.isfile(full_path) else "",
            )

            return FileResult(True, "get_info", path, output=info.to_dict())

        except Exception as e:
            return FileResult(False, "get_info", path, error=str(e))

    def search_in_file(
        self, path: str, pattern: str, max_results: int = 100
    ) -> FileResult:
        """Search for pattern in file content."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "search", path, error="File not found")

            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            matches = []
            for i, line in enumerate(lines, 1):
                if pattern.lower() in line.lower():
                    matches.append({"line": i, "content": line.rstrip()})
                    if len(matches) >= max_results:
                        break

            return FileResult(
                True, "search", path,
                output=matches, files_affected=len(matches),
            )

        except Exception as e:
            return FileResult(False, "search", path, error=str(e))

    def replace_in_file(
        self, path: str, old: str, new: str, count: int = 0
    ) -> FileResult:
        """Replace text in file."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "replace", path, error="File not found")

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content, replacements = content.replace(old, new), content.count(old)
            if count > 0:
                new_content = content.replace(old, new, count)
                replacements = min(replacements, count)

            if replacements == 0:
                return FileResult(False, "replace", path, error="Pattern not found")

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return FileResult(
                True, "replace", path, files_affected=replacements,
            )

        except Exception as e:
            return FileResult(False, "replace", path, error=str(e))

    def get_checksum(self, path: str, algorithm: str = "sha256") -> FileResult:
        """Calculate file checksum."""
        try:
            full_path = self._resolve_path(path)
            if not os.path.exists(full_path):
                return FileResult(False, "checksum", path, error="File not found")

            h = hashlib.new(algorithm)
            with open(full_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)

            return FileResult(True, "checksum", path, output=h.hexdigest())

        except Exception as e:
            return FileResult(False, "checksum", path, error=str(e))

    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to workspace."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self._workspace, path))

    def _validate_extension(self, path: str) -> bool:
        """Validate file extension."""
        ext = Path(path).suffix.lower()
        if self._blocked_ext and ext in self._blocked_ext:
            return False
        if self._allowed_ext and ext not in self._allowed_ext:
            return False
        return True
