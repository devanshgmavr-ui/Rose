"""Media tools for integration with the agent tool system."""

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..tools.base import Tool, ToolResult, Permission, ConfirmationLevel
from .base import MediaType, MediaRequest, MediaResult
from .router import MediaRouter
from .storage import MediaStorage

logger = logging.getLogger(__name__)


class ImageAnalyzeTool(Tool):
    def __init__(self, media_router: MediaRouter):
        self._router = media_router

    @property
    def name(self) -> str:
        return "image_analyze"

    @property
    def description(self) -> str:
        return "Analyze an image to understand its content, objects, and context"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze",
                },
                "prompt": {
                    "type": "string",
                    "description": "Specific question about the image (optional)",
                    "default": "Describe this image in detail",
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
                "analysis": {"type": "object"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.FILESYSTEM_READ]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return 60.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
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

            size = os.path.getsize(image_path)
            max_size = 20 * 1024 * 1024
            if size > max_size:
                errors.append(f"Image too large: {size} bytes (max {max_size})")

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

        request = MediaRequest(
            media_type=MediaType.IMAGE,
            input_path=arguments["image_path"],
            prompt=arguments.get("prompt", "Describe this image in detail"),
        )

        result = self._router.route(request)

        if result.success:
            output_text = self._format_analysis(result)
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=output_text,
                execution_time=result.execution_time,
                metadata=result.metadata,
            )
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result.error,
                execution_time=result.execution_time,
            )

    def _format_analysis(self, result: MediaResult) -> str:
        if not result.output:
            return "Analysis complete"

        meta = result.metadata
        lines = [f"Image Analysis: {meta.get('file_name', 'unknown')}"]
        lines.append(f"Size: {meta.get('file_size', 0)} bytes")
        lines.append(f"Format: {meta.get('file_extension', 'unknown')}")
        lines.append(f"Description: {meta.get('description', 'N/A')}")

        if meta.get('analysis_prompt'):
            lines.append(f"Prompt: {meta['analysis_prompt']}")

        return "\n".join(lines)


class ImageGenerateTool(Tool):
    def __init__(self, media_router: MediaRouter):
        self._router = media_router

    @property
    def name(self) -> str:
        return "image_generate"

    @property
    def description(self) -> str:
        return "Generate an image from a text prompt"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the image to generate",
                },
                "width": {
                    "type": "integer",
                    "description": "Image width in pixels",
                    "default": 512,
                },
                "height": {
                    "type": "integer",
                    "description": "Image height in pixels",
                    "default": 512,
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional)",
                },
                "output_filename": {
                    "type": "string",
                    "description": "Output filename (optional)",
                },
            },
            "required": ["prompt"],
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
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.FILESYSTEM_WRITE]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 120.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        prompt = arguments.get("prompt", "")
        if not prompt or not prompt.strip():
            errors.append("prompt is required")

        width = arguments.get("width", 512)
        height = arguments.get("height", 512)

        if not isinstance(width, int) or width < 0:
            errors.append(f"Invalid width: {width}")
        elif width > 2048:
            errors.append(f"Width exceeds maximum: {width}")

        if not isinstance(height, int) or height < 0:
            errors.append(f"Invalid height: {height}")
        elif height > 2048:
            errors.append(f"Height exceeds maximum: {height}")

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

        filename = arguments.get("output_filename")
        if not filename:
            filename = f"generated_{uuid.uuid4().hex[:8]}.png"

        output_path = str(
            Path("workspace") / "media" / "images" / filename
        )

        request = MediaRequest(
            media_type=MediaType.IMAGE,
            prompt=arguments["prompt"],
            width=arguments.get("width", 512),
            height=arguments.get("height", 512),
            seed=arguments.get("seed"),
            output_path=output_path,
        )

        result = self._router.route(request)

        if result.success and result.output:
            output = result.output
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=f"Image generated: {output.path}\nSize: {output.width}x{output.height}\nFile: {output.file_size} bytes",
                execution_time=result.execution_time,
                metadata={
                    "path": output.path,
                    "width": output.width,
                    "height": output.height,
                    "file_size": output.file_size,
                },
            )
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result.error,
                execution_time=result.execution_time,
            )


class VideoGenerateTool(Tool):
    def __init__(self, media_router: MediaRouter):
        self._router = media_router

    @property
    def name(self) -> str:
        return "video_generate"

    @property
    def description(self) -> str:
        return "Generate a video from a text prompt"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text description of the video to generate",
                },
                "width": {
                    "type": "integer",
                    "description": "Video width in pixels",
                    "default": 256,
                },
                "height": {
                    "type": "integer",
                    "description": "Video height in pixels",
                    "default": 256,
                },
                "duration": {
                    "type": "number",
                    "description": "Video duration in seconds",
                    "default": 5.0,
                },
                "num_frames": {
                    "type": "integer",
                    "description": "Number of frames to generate",
                    "default": 30,
                },
                "output_filename": {
                    "type": "string",
                    "description": "Output filename (optional)",
                },
            },
            "required": ["prompt"],
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "duration": {"type": "number"},
                "file_size": {"type": "integer"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.FILESYSTEM_WRITE]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.REQUIRE_CONFIRMATION

    @property
    def timeout(self) -> float:
        return 180.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        prompt = arguments.get("prompt", "")
        if not prompt or not prompt.strip():
            errors.append("prompt is required")

        width = arguments.get("width", 256)
        height = arguments.get("height", 256)
        duration = arguments.get("duration", 5.0)
        num_frames = arguments.get("num_frames", 30)

        if not isinstance(width, int) or width < 0:
            errors.append(f"Invalid width: {width}")
        elif width > 1024:
            errors.append(f"Width exceeds maximum: {width}")

        if not isinstance(height, int) or height < 0:
            errors.append(f"Invalid height: {height}")
        elif height > 1024:
            errors.append(f"Height exceeds maximum: {height}")

        if duration <= 0:
            errors.append(f"Duration must be positive: {duration}")
        elif duration > 30.0:
            errors.append(f"Duration exceeds maximum: {duration}s")

        if num_frames < 1:
            errors.append(f"num_frames must be at least 1: {num_frames}")
        elif num_frames > 150:
            errors.append(f"num_frames exceeds maximum: {num_frames}")

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

        filename = arguments.get("output_filename")
        if not filename:
            filename = f"video_{uuid.uuid4().hex[:8]}.mp4"

        output_path = str(
            Path("workspace") / "media" / "videos" / filename
        )

        request = MediaRequest(
            media_type=MediaType.VIDEO,
            prompt=arguments["prompt"],
            width=arguments.get("width", 256),
            height=arguments.get("height", 256),
            duration=arguments.get("duration", 5.0),
            num_frames=arguments.get("num_frames", 30),
            output_path=output_path,
        )

        result = self._router.route(request)

        if result.success and result.output:
            output = result.output
            return ToolResult(
                success=True,
                tool_name=self.name,
                output=(
                    f"Video generated: {output.path}\n"
                    f"Size: {output.width}x{output.height}\n"
                    f"Duration: {output.duration}s\n"
                    f"File: {output.file_size} bytes"
                ),
                execution_time=result.execution_time,
                metadata={
                    "path": output.path,
                    "width": output.width,
                    "height": output.height,
                    "duration": output.duration,
                    "file_size": output.file_size,
                },
            )
        else:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=result.error,
                execution_time=result.execution_time,
            )


class MediaInfoTool(Tool):
    def __init__(self, media_storage: MediaStorage):
        self._storage = media_storage

    @property
    def name(self) -> str:
        return "media_info"

    @property
    def description(self) -> str:
        return "Get information about stored media files"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "info", "stats"],
                    "description": "Action to perform",
                    "default": "stats",
                },
                "media_type": {
                    "type": "string",
                    "enum": ["image", "video"],
                    "description": "Filter by media type (for list action)",
                },
                "filepath": {
                    "type": "string",
                    "description": "File path (for info action)",
                },
            },
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {"type": "array"},
                "stats": {"type": "object"},
                "info": {"type": "object"},
            },
        }

    @property
    def required_permissions(self) -> list:
        return [Permission.FILESYSTEM_READ]

    @property
    def confirmation_level(self) -> ConfirmationLevel:
        return ConfirmationLevel.ALLOW

    @property
    def timeout(self) -> float:
        return 10.0

    def validate(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        action = arguments.get("action", "stats")
        valid_actions = ["list", "info", "stats"]
        if action not in valid_actions:
            errors.append(f"Invalid action: {action}. Must be one of {valid_actions}")

        if action == "info" and not arguments.get("filepath"):
            errors.append("filepath is required for info action")

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

        action = arguments.get("action", "stats")

        try:
            if action == "stats":
                stats = self._storage.get_storage_stats()
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    output=f"Storage stats: {stats}",
                    execution_time=time.time() - start,
                    metadata=stats,
                )

            elif action == "list":
                media_type_str = arguments.get("media_type")
                media_type = MediaType(media_type_str) if media_type_str else None
                files = self._storage.list_media(media_type)
                return ToolResult(
                    success=True,
                    tool_name=self.name,
                    output=f"Found {len(files)} media files",
                    execution_time=time.time() - start,
                    metadata={"files": files},
                )

            elif action == "info":
                filepath = arguments["filepath"]
                info = self._storage.get_file_info(filepath)
                if info:
                    return ToolResult(
                        success=True,
                        tool_name=self.name,
                        output=f"File info: {info}",
                        execution_time=time.time() - start,
                        metadata=info,
                    )
                else:
                    return ToolResult(
                        success=False,
                        tool_name=self.name,
                        error=f"File not found: {filepath}",
                        execution_time=time.time() - start,
                    )

        except Exception as e:
            return ToolResult(
                success=False,
                tool_name=self.name,
                error=str(e),
                execution_time=time.time() - start,
            )

        return ToolResult(
            success=False,
            tool_name=self.name,
            error=f"Unknown action: {action}",
            execution_time=time.time() - start,
        )
