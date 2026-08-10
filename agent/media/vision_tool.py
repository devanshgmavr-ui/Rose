"""Vision analysis tool for the agent tool system.

Stage 3.1 - Vision Analysis.

Provides vision analysis capabilities with workspace-boundary
validation and untrusted content markers.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel
from .analyzer import VisionAnalyzer
from .storage import MediaStorage

logger = logging.getLogger(__name__)


class VisionAnalyzeTool(Tool):
    """Tool for analyzing images and screenshots."""

    def __init__(
        self,
        vision_analyzer: VisionAnalyzer,
        media_storage: MediaStorage,
        vision_enabled: bool = True,
        max_image_size_mb: int = 20,
        workspace_dir: str = "workspace",
    ):
        self._analyzer = vision_analyzer
        self._storage = media_storage
        self._enabled = vision_enabled
        self._max_image_size_mb = max_image_size_mb
        self._workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "vision_analyze"

    @property
    def description(self) -> str:
        return "Analyze images and screenshots to understand visual content, objects, text, and UI elements"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["analyze", "describe"],
                    "description": "Action to perform: 'analyze' for structured output, 'describe' for text summary",
                    "default": "analyze",
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "Specific question about the image (optional)",
                    "default": "Analyze this image in detail",
                },
            },
            "required": ["image_path"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "detected_elements": {"type": "array"},
                "image_width": {"type": "integer"},
                "image_height": {"type": "integer"},
                "metadata": {"type": "object"},
            },
        }

    @property
    def required_permissions(self) -> list:
        if not self._enabled:
            return []
        return ["vision.analyze"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 60.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tool arguments."""
        errors = []

        if not self._enabled:
            errors.append("Vision analysis is disabled by configuration")
            return False, errors

        action = arguments.get("action", "analyze")
        valid_actions = ["analyze", "describe"]
        if action not in valid_actions:
            errors.append(f"Invalid action: {action}. Must be one of {valid_actions}")

        image_path = arguments.get("image_path", "")
        if not image_path:
            errors.append("image_path is required")
        elif not os.path.exists(image_path):
            errors.append(f"Image file not found: {image_path}")
        elif not os.path.isfile(image_path):
            errors.append(f"Path is not a file: {image_path}")
        else:
            ext = Path(image_path).suffix.lower()
            allowed = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]
            if ext not in allowed:
                errors.append(f"Unsupported image format: {ext}")

            try:
                size = os.path.getsize(image_path)
            except OSError:
                size = 0

            max_size_bytes = self._max_image_size_mb * 1024 * 1024
            if size > max_size_bytes:
                errors.append(
                    f"Image too large: {size} bytes "
                    f"(max {max_size_bytes} bytes / {self._max_image_size_mb} MB)"
                )

            if size == 0:
                errors.append("Image file is empty")

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute vision analysis."""
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        action = arguments.get("action", "analyze")
        image_path = arguments["image_path"]
        prompt = arguments.get("prompt", "Analyze this image in detail")

        try:
            if action == "describe":
                output = self._analyzer.describe_image(image_path)
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    output=output,
                    execution_time=time.time() - start,
                    metadata={"action": "describe", "image_path": image_path},
                )
            else:
                result = self._analyzer.analyze(
                    image_path=image_path,
                    prompt=prompt,
                    workspace_root=self._workspace_dir,
                )

                if result.success:
                    output = result.to_text()
                    return ToolResult(
                        success=True,
                        tool_name=self.name,
                        output=output,
                        execution_time=result.execution_time,
                        metadata={
                            "action": "analyze",
                            "image_path": image_path,
                            "image_width": result.image_width,
                            "image_height": result.image_height,
                            "element_count": len(result.detected_elements),
                        },
                    )
                else:
                    return ToolResult(
                        success=False,
                        tool_name=self.name,
                        error=result.error,
                        execution_time=result.execution_time,
                    )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Vision analysis failed: {e}",
                execution_time=time.time() - start,
            )
