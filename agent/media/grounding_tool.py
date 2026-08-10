"""Visual grounding tool for the agent tool system.

Stage 3.2 - Visual Grounding.

Translates vision analysis results into grounded targets
with coordinates for mouse/keyboard actions.
Does NOT perform actions directly - targets go through
ToolRouter -> PermissionManager -> ActionTool.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel
from .grounding import VisualGrounder, GroundingResult, GroundedTarget
from .analyzer import VisionAnalyzer
from .storage import MediaStorage

logger = logging.getLogger(__name__)


class VisualGroundingTool(Tool):
    """Tool for grounding vision results into actionable targets."""

    def __init__(
        self,
        vision_analyzer: VisionAnalyzer,
        visual_grounder: VisualGrounder,
        media_storage: MediaStorage,
        vision_enabled: bool = True,
        grounding_enabled: bool = True,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        self._analyzer = vision_analyzer
        self._grounder = visual_grounder
        self._storage = media_storage
        self._vision_enabled = vision_enabled
        self._grounding_enabled = grounding_enabled
        self._screen_width = screen_width
        self._screen_height = screen_height

    @property
    def name(self) -> str:
        return "visual_ground"

    @property
    def description(self) -> str:
        return "Ground vision results into actionable coordinates and targets"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ground", "validate"],
                    "description": "Action to perform: 'ground' to find targets, 'validate' to check target safety",
                    "default": "ground",
                },
                "image_path": {
                    "type": "string",
                    "description": "Path to the image/screenshot to analyze",
                },
                "target": {
                    "type": "string",
                    "description": "Description of the target element to find (optional, finds all if empty)",
                },
                "screen_width": {
                    "type": "integer",
                    "description": "Screen width in pixels",
                    "default": 1920,
                },
                "screen_height": {
                    "type": "integer",
                    "description": "Screen height in pixels",
                    "default": 1080,
                },
            },
            "required": ["image_path"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "targets": {"type": "array"},
                "target_count": {"type": "integer"},
                "screen_width": {"type": "integer"},
                "screen_height": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        if not self._vision_enabled or not self._grounding_enabled:
            return []
        return ["vision.analyze"]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        if not self._vision_enabled or not self._grounding_enabled:
            return ConfirmationLevel.DENY
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 60.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate tool arguments."""
        errors = []

        if not self._vision_enabled:
            errors.append("Vision analysis is disabled by configuration")
            return False, errors

        if not self._grounding_enabled:
            errors.append("Visual grounding is disabled by configuration")
            return False, errors

        action = arguments.get("action", "ground")
        valid_actions = ["ground", "validate"]
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

        return len(errors) == 0, errors

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute visual grounding."""
        start = time.time()

        valid, errors = self.validate(arguments)
        if not valid:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error="; ".join(errors),
                execution_time=time.time() - start,
            )

        action = arguments.get("action", "ground")
        image_path = arguments["image_path"]
        target = arguments.get("target")
        sw = arguments.get("screen_width", self._screen_width)
        sh = arguments.get("screen_height", self._screen_height)

        try:
            vision_result = self._analyzer.analyze(
                image_path=image_path,
                prompt=f"Analyze UI elements for: {target or 'all elements'}",
            )

            if not vision_result.success:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=vision_result.error,
                    execution_time=time.time() - start,
                )

            grounding_result = self._grounder.ground(
                vision_result=vision_result,
                target_description=target,
                screen_width=sw,
                screen_height=sh,
            )

            if not grounding_result.success:
                return ToolResult(
                    success=False,
                    tool_name=self.name,
                    error=grounding_result.error,
                    execution_time=time.time() - start,
                )

            if action == "validate":
                return self._validate_targets(grounding_result, start)

            output = grounding_result.to_text()
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=output,
                execution_time=time.time() - start,
                metadata={
                    "action": "ground",
                    "target_count": len(grounding_result.targets),
                    "screen_width": grounding_result.screen_width,
                    "screen_height": grounding_result.screen_height,
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=f"Grounding failed: {e}",
                execution_time=time.time() - start,
            )

    def _validate_targets(
        self, result: GroundingResult, start: float
    ) -> ToolResult:
        """Validate grounded targets."""
        all_valid = True
        validation_results = []

        for target in result.targets:
            is_valid, errors = self._grounder.validate_target(target)
            validation_results.append({
                "description": target.description,
                "is_valid": is_valid,
                "errors": errors,
            })
            if not is_valid:
                all_valid = False

        output_lines = [
            "[BEGIN UNTRUSTED GROUNDING VALIDATION]"
        ]
        for vr in validation_results:
            status = "VALID" if vr["is_valid"] else "INVALID"
            output_lines.append(f"Target: {vr['description']} - {status}")
            if vr["errors"]:
                for err in vr["errors"]:
                    output_lines.append(f"  Issue: {err}")
        output_lines.append("[END UNTRUSTED GROUNDING VALIDATION]")

        return ToolResult(
            success=True,
            tool_name=self.name,
            output="\n".join(output_lines),
            execution_time=time.time() - start,
            metadata={
                "action": "validate",
                "all_valid": all_valid,
                "validation_results": validation_results,
            },
        )
