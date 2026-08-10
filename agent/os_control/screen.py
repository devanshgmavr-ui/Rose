"""Screen capture tool using PIL.ImageGrab."""

import os
import time
import uuid
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel

logger = logging.getLogger(__name__)

MAX_SCREENSHOT_WIDTH = 4096
MAX_SCREENSHOT_HEIGHT = 4096
MAX_SCREENSHOT_SIZE_BYTES = 20 * 1024 * 1024


class ScreenCaptureTool(Tool):
    def __init__(self, workspace_dir: str = "workspace"):
        self._workspace = Path(workspace_dir).resolve()
        self._media_dir = self._workspace / "media" / "screenshots"
        self._media_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "screen_capture"

    @property
    def description(self) -> str:
        return "Capture a screenshot of the current screen"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "region": {
                    "type": "object",
                    "description": "Optional region to capture: {x, y, width, height}",
                    "properties": {
                        "x": {"type": "integer", "default": 0},
                        "y": {"type": "integer", "default": 0},
                        "width": {"type": "integer", "default": 0},
                        "height": {"type": "integer", "default": 0},
                    },
                },
                "filename": {
                    "type": "string",
                    "description": "Optional filename for the screenshot",
                },
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "file_size": {"type": "integer"},
                "format": {"type": "string"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return ["os.screen_capture"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return 15.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []

        region = arguments.get("region")
        if region is not None:
            if not isinstance(region, dict):
                errors.append("region must be a dictionary with x, y, width, height")
            else:
                for key in ("x", "y", "width", "height"):
                    val = region.get(key, 0)
                    if not isinstance(val, int):
                        errors.append(f"region.{key} must be an integer")

                width = region.get("width", 0)
                height = region.get("height", 0)
                if width < 0 or height < 0:
                    errors.append("region dimensions must be non-negative")
                if width > MAX_SCREENSHOT_WIDTH or height > MAX_SCREENSHOT_HEIGHT:
                    errors.append(
                        f"region dimensions exceed maximum: {width}x{height} > "
                        f"{MAX_SCREENSHOT_WIDTH}x{MAX_SCREENSHOT_HEIGHT}"
                    )

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

        try:
            from PIL import ImageGrab

            region = arguments.get("region")
            bbox = None
            if region and region.get("width", 0) > 0 and region.get("height", 0) > 0:
                bbox = (
                    region.get("x", 0),
                    region.get("y", 0),
                    region.get("x", 0) + region["width"],
                    region.get("y", 0) + region["height"],
                )

            screenshot = ImageGrab.grab(bbox=bbox)

            width, height = screenshot.size
            if width > MAX_SCREENSHOT_WIDTH or height > MAX_SCREENSHOT_HEIGHT:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Screenshot too large: {width}x{height}",
                    execution_time=time.time() - start,
                )

            filename = arguments.get("filename")
            if not filename:
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}_{uuid.uuid4().hex[:6]}.png"

            if not filename.endswith(".png"):
                filename += ".png"

            safe_name = self._sanitize_filename(filename)
            output_path = self._media_dir / safe_name

            screenshot.save(str(output_path), "PNG")

            file_size = output_path.stat().st_size
            if file_size > MAX_SCREENSHOT_SIZE_BYTES:
                output_path.unlink()
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=f"Screenshot file too large: {file_size} bytes",
                    execution_time=time.time() - start,
                )

            execution_time = time.time() - start

            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Screenshot captured: {output_path}",
                execution_time=execution_time,
                metadata={
                    "path": str(output_path),
                    "width": width,
                    "height": height,
                    "file_size": file_size,
                    "format": "png",
                    "timestamp": time.time(),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Screen capture failed: {e}",
                execution_time=time.time() - start,
            )

    def _sanitize_filename(self, filename: str) -> str:
        import re
        name = Path(filename).stem
        ext = Path(filename).suffix
        safe = re.sub(r'[^\w\-]', '_', name)
        safe = re.sub(r'_+', '_', safe).strip('_')
        if not safe:
            safe = f"screenshot_{uuid.uuid4().hex[:8]}"
        if len(safe) > 100:
            safe = safe[:100]
        return f"{safe}{ext}"
