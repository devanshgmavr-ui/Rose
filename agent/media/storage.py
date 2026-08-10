"""Secure media storage with path validation and size limits."""

import os
import re
import time
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from .base import MediaType, MEDIA_TYPE_EXTENSIONS

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_IMAGE_SIZE = 20 * 1024 * 1024
MAX_VIDEO_SIZE = 100 * 1024 * 1024
SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9_\-. ]+$')


class MediaStorage:
    def __init__(self, workspace_dir: str = "workspace"):
        self._workspace = Path(workspace_dir).resolve()
        self._media_dir = self._workspace / "media"
        self._images_dir = self._media_dir / "images"
        self._videos_dir = self._media_dir / "videos"
        self._temp_dir = self._media_dir / "temporary"
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for d in [self._media_dir, self._images_dir, self._videos_dir, self._temp_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def workspace_path(self) -> Path:
        return self._workspace

    @property
    def media_path(self) -> Path:
        return self._media_dir

    @property
    def images_path(self) -> Path:
        return self._images_dir

    @property
    def videos_path(self) -> Path:
        return self._videos_dir

    @property
    def temp_path(self) -> Path:
        return self._temp_dir

    def _is_within_workspace(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self._workspace)
            return True
        except ValueError:
            return False

    def validate_path(self, path: str) -> Tuple[bool, List[str]]:
        errors = []
        if not path or not path.strip():
            errors.append("Path is empty")
            return False, errors

        p = Path(path)

        if p.is_absolute():
            if not self._is_within_workspace(p):
                errors.append(f"Absolute path outside workspace: {path}")
                return False, errors

        resolved = (self._workspace / p).resolve()
        if not self._is_within_workspace(resolved):
            errors.append(f"Path resolves outside workspace: {path}")
            return False, errors

        parts = p.parts
        for part in parts:
            if part in ("..", "~"):
                errors.append(f"Path traversal detected: {part}")
                return False, errors

        return True, []

    def validate_file_type(self, filename: str, expected_type: Optional[MediaType] = None) -> Tuple[bool, List[str]]:
        errors = []
        if not filename:
            errors.append("Filename is empty")
            return False, errors

        ext = Path(filename).suffix.lower()
        if not ext:
            errors.append("File has no extension")
            return False, errors

        if expected_type:
            allowed = MEDIA_TYPE_EXTENSIONS.get(expected_type, [])
            if ext not in allowed:
                errors.append(f"Extension {ext} not valid for {expected_type.value}")
                return False, errors

        return True, []

    def validate_file_size(self, size_bytes: int, media_type: Optional[MediaType] = None) -> Tuple[bool, List[str]]:
        errors = []
        if size_bytes <= 0:
            errors.append("File size must be positive")
            return False, errors

        limit = MAX_FILE_SIZE
        if media_type == MediaType.IMAGE:
            limit = MAX_IMAGE_SIZE
        elif media_type == MediaType.VIDEO:
            limit = MAX_VIDEO_SIZE

        if size_bytes > limit:
            errors.append(f"File size {size_bytes} exceeds limit {limit}")
            return False, errors

        return True, []

    def sanitize_filename(self, filename: str) -> str:
        if not filename:
            return f"media_{uuid.uuid4().hex[:8]}"

        name = Path(filename).stem
        ext = Path(filename).suffix

        safe_name = re.sub(r'[^\w\s\-]', '', name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('_')

        if not safe_name:
            safe_name = f"media_{uuid.uuid4().hex[:8]}"

        if len(safe_name) > 100:
            safe_name = safe_name[:100]

        return f"{safe_name}{ext}"

    def get_storage_path(self, media_type: MediaType, filename: str) -> Path:
        safe_name = self.sanitize_filename(filename)
        if media_type == MediaType.IMAGE:
            return self._images_dir / safe_name
        elif media_type == MediaType.VIDEO:
            return self._videos_dir / safe_name
        else:
            return self._temp_dir / safe_name

    def store_file(self, source_path: str, media_type: MediaType, filename: Optional[str] = None) -> Tuple[bool, str, List[str]]:
        errors = []
        source = Path(source_path)

        if not source.exists():
            errors.append(f"Source file not found: {source_path}")
            return False, "", errors

        if not source.is_file():
            errors.append(f"Source is not a file: {source_path}")
            return False, "", errors

        file_size = source.stat().st_size
        valid, size_errors = self.validate_file_size(file_size, media_type)
        if not valid:
            return False, "", size_errors

        actual_name = filename or source.name
        dest = self.get_storage_path(media_type, actual_name)

        try:
            shutil.copy2(str(source), str(dest))
            logger.info(f"Stored {source_path} -> {dest}")
            return True, str(dest), []
        except Exception as e:
            errors.append(f"Storage failed: {e}")
            return False, "", errors

    def store_bytes(self, data: bytes, media_type: MediaType, filename: str) -> Tuple[bool, str, List[str]]:
        errors = []

        valid, size_errors = self.validate_file_size(len(data), media_type)
        if not valid:
            return False, "", size_errors

        dest = self.get_storage_path(media_type, filename)

        try:
            dest.write_bytes(data)
            logger.info(f"Stored {len(data)} bytes -> {dest}")
            return True, str(dest), []
        except Exception as e:
            errors.append(f"Storage failed: {e}")
            return False, "", errors

    def delete_file(self, filepath: str) -> Tuple[bool, List[str]]:
        errors = []
        p = Path(filepath)

        if not self._is_within_workspace(p.resolve()):
            errors.append("Cannot delete file outside workspace")
            return False, errors

        if not p.exists():
            return True, []

        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(str(p))
            return True, []
        except Exception as e:
            errors.append(f"Delete failed: {e}")
            return False, errors

    def get_file_info(self, filepath: str) -> Optional[dict]:
        p = Path(filepath)
        if not p.exists():
            return None

        stat = p.stat()
        return {
            "path": str(p),
            "name": p.name,
            "extension": p.suffix,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
        }

    def list_media(self, media_type: Optional[MediaType] = None) -> List[dict]:
        results = []
        directories = []

        if media_type is None or media_type == MediaType.IMAGE:
            directories.append(self._images_dir)
        if media_type is None or media_type == MediaType.VIDEO:
            directories.append(self._videos_dir)

        for d in directories:
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        info = self.get_file_info(str(f))
                        if info:
                            results.append(info)

        return results

    def cleanup_temp(self, max_age_seconds: float = 3600) -> int:
        removed = 0
        if not self._temp_dir.exists():
            return removed

        now = time.time()
        for f in self._temp_dir.iterdir():
            if f.is_file():
                age = now - f.stat().st_mtime
                if age > max_age_seconds:
                    try:
                        f.unlink()
                        removed += 1
                    except Exception:
                        pass

        return removed

    def get_storage_stats(self) -> dict:
        image_count = len(list(self._images_dir.iterdir())) if self._images_dir.exists() else 0
        video_count = len(list(self._videos_dir.iterdir())) if self._videos_dir.exists() else 0
        temp_count = len(list(self._temp_dir.iterdir())) if self._temp_dir.exists() else 0

        def _dir_size(d: Path) -> int:
            if not d.exists():
                return 0
            return sum(f.stat().st_size for f in d.iterdir() if f.is_file())

        return {
            "media_dir": str(self._media_dir),
            "image_count": image_count,
            "video_count": video_count,
            "temp_count": temp_count,
            "image_bytes": _dir_size(self._images_dir),
            "video_bytes": _dir_size(self._videos_dir),
            "temp_bytes": _dir_size(self._temp_dir),
        }
